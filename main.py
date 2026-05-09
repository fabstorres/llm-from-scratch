shakespeare = open("./data/shakespeare.txt").read()
chars = sorted(set(shakespeare))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(id: list[int]) -> str:
    return "".join([itos[i] for i in id])


encoding = encode("Hello, World!")
print(encoding)
decoding = decode(encoding)
print(decoding)
