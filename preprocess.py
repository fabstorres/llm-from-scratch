import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def read_batches(files, batch_size=512):
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            yield [text or "" for text in batch.column("text").to_pylist()]


def serialize_to_bin_file(folder_path, output_path=None):
    folder = Path(folder_path)
    files = sorted(folder.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    output = Path(output_path) if output_path else folder / "openwebtext.bin"
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    # Same tokenizer as train.py: sorted characters mapped to integer IDs.
    chars = set()
    progress = tqdm(
        total=total, desc="Building vocabulary", unit=" docs", mininterval=5
    )
    for texts in read_batches(files):
        for text in texts:
            chars.update(text)
        progress.update(len(texts))
    progress.close()

    vocab = sorted(chars)
    stoi = {char: token_id for token_id, char in enumerate(vocab)}
    if len(stoi) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Character vocabulary does not fit in uint16")

    metadata_path = output.with_suffix(output.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {"dtype": "uint16", "vocab": vocab},
            metadata_file,
            ensure_ascii=False,
        )

    lookup = np.zeros(max(map(ord, chars)) + 1, dtype=np.uint16)
    for char, token_id in stoi.items():
        lookup[ord(char)] = token_id

    progress = tqdm(total=total, desc="Writing tokens", unit=" docs", mininterval=5)
    with output.open("wb") as bin_file:
        for texts in read_batches(files):
            text = "".join(texts)
            codepoints = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
            lookup[codepoints].tofile(bin_file)
            progress.update(len(texts))
    progress.close()

    print(f"Wrote {output}")
    print(f"Wrote {metadata_path}")
    print(f"Vocabulary size: {len(stoi):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", nargs="?", default="data/openwebtext")
    parser.add_argument("--output")
    args = parser.parse_args()
    serialize_to_bin_file(args.folder_path, args.output)
