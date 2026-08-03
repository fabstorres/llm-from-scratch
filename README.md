# Train an LLM From Scratch

> **Original work and attribution:** This repository began with the workshop created by [Angelos Perivolaropoulos](https://github.com/angelos-p) in [`angelos-p/llm-from-scratch`](https://github.com/angelos-p/llm-from-scratch). It was detached from the fork network because my work increasingly diverged through additional model experiments, analysis, and large-dataset tooling—not to erase the project’s origin or attribution.

This repository documents my progression from the original Shakespeare workshop model toward training small GPT models on larger datasets. It contains the original workshop material, a runnable reference implementation, three analyzed training runs, and the preprocessing pipeline developed for the OpenWebText experiment.

## Experiment Progression

All three completed experiments use the same decoder-only GPT implementation and were trained for 5,000 optimizer steps. The runs differ in model capacity, dataset, vocabulary size, and batch size.

| Run | Dataset | Character vocabulary | Architecture | Batch | Parameters | Best recorded validation loss | Result |
|-----|---------|---------------------:|--------------|------:|-----------:|------------------------------:|--------|
| **model-01** | Shakespeare | 65 | 6 layers, 6 heads, width 384 | 64 | 10,770,816 | 1.5748 at step 1,400 | Strong overfitting after the minimum |
| **model-02** | Shakespeare | 65 | 3 layers, 4 heads, width 112 | 64 | 492,128 | 1.7436 at step 4,900 | Continued generalization through the run |
| **model-03** | OpenWebText | 30,993 | 3 layers, 4 heads, width 112 | 16 | 3,956,064 | 2.3595 at step 4,400 | Large improvement followed by a late plateau |

Validation is evaluated every 100 steps, so the table reports the best **recorded** validation value rather than a continuously observed minimum. Loss values for the Shakespeare and OpenWebText runs are not directly comparable because the datasets and prediction vocabularies are substantially different.

The experiment artifacts live locally under `model-01/`, `model-02/`, and `model-03/`. Checkpoints and loss logs are intentionally ignored by Git, while their analysis is maintained in [`analysis.ipynb`](analysis.ipynb).

### Model 01: The Original Workshop Model

Model-01 is the first model created by following the original workshop. It uses the workshop’s default medium configuration:

```text
Dataset:       Shakespeare
Vocabulary:    65 characters
Architecture:  6 layers / 6 heads / 384 embedding dimensions
Context:       256 characters
Batch size:    64
Parameters:    10,770,816
Steps:         5,000
```

The model learned the training set aggressively. Its best recorded validation loss was **1.5748 at step 1,400**, but training loss continued falling while validation loss rose to **3.9216 at step 4,900**. This is the clearest overfitting run in the project: the model had enough capacity to increasingly memorize the approximately 1.1-million-character Shakespeare corpus instead of continuing to generalize.

### Model 02: Testing a Smaller Model

The workshop suggests that a model around 0.5M parameters may be better matched to the small Shakespeare dataset. Model-02 was my attempt to reproduce and analyze that claim while keeping the dataset, tokenizer, context length, optimizer, batch size, and training duration consistent with model-01.

The workshop uses 2 layers, 2 heads, and width 128 as an illustrative ~0.5M configuration. The actual model-02 experiment instead uses a custom architecture that reaches approximately the same parameter target:

```text
Dataset:       Shakespeare
Vocabulary:    65 characters
Architecture:  3 layers / 4 heads / 112 embedding dimensions
Context:       256 characters
Batch size:    64
Parameters:    492,128
Steps:         5,000
```

Model-02 learned more slowly than model-01, but its validation loss continued trending downward throughout the run. Its best and last recorded validation value was **1.7436 at step 4,900**, without the large train/validation divergence seen in model-01.

This run supports the practical lesson that model capacity needs to be matched to the amount of available data. The larger model reached a lower validation loss earlier, but the smaller model generalized much more consistently over the full training run.

### Model 03: Scaling the Dataset

Model-03 explores the other side of that relationship: instead of making the model smaller for Shakespeare, what happens if the small model is given substantially more data?

The experiment uses the transformer-body configuration from model-02 and attempts to train on [Skylion007/OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), an open replication of the WebText dataset used in the development of GPT-2.

#### Preparing a dataset that does not fit in RAM

The downloaded Parquet shards totaled roughly **24 GB** locally and could not be loaded into RAM as one in-memory token array. To make step-based random sampling possible, I added [`preprocess.py`](preprocess.py).

The preprocessing pipeline:

1. recursively discovers Parquet files;
2. reads their `text` columns in batches;
3. builds a sorted vocabulary from every distinct character;
4. converts each Unicode character to a `uint16` token ID;
5. writes the tokens to a `.bin` file; and
6. writes the dtype and vocabulary to an adjacent `.bin.json` metadata file.

[`train.py`](train.py) detects `.bin` input and opens it with a NumPy memory map. This keeps the full token dataset on disk while copying only sampled training and validation batches into RAM and GPU memory.

```bash
uv run python preprocess.py data/openwebtext --output data/openwebtext.bin
uv run python train.py data/openwebtext.bin
```

The raw OpenWebText data and generated binary files are not included in this repository.

#### The character-vocabulary problem

Using the original character tokenizer on OpenWebText expanded the vocabulary from **65 Shakespeare characters to 30,993 distinct Unicode characters**. Although model-03 kept model-02’s 3-layer, 4-head, width-112 transformer body, the much larger tied token-embedding/output matrix increased the complete model from **492,128 to 3,956,064 parameters**.

The first batch-64 attempt ran out of CUDA memory on my **NVIDIA GeForce RTX 3060 Ti with 8 GB of VRAM**. Reducing the batch size to 16 allowed the experiment to run:

```text
Dataset:       locally prepared OpenWebText
Vocabulary:    30,993 characters
Architecture:  3 layers / 4 heads / 112 embedding dimensions
Context:       256 characters
Batch size:    16
Parameters:    3,956,064
Steps:         5,000
```

Model-03 clearly learned during the run. Validation loss fell from **10.5259 at initialization to 2.3595 at step 4,400**. The last recorded value was **2.3629 at step 4,900**, a very small increase from the minimum. I interpret this as a late plateau rather than the severe overfitting seen in model-01.

This was only a short exploratory run. Five thousand random-batch steps expose the model to a tiny fraction of a dataset this large, so model-03 should not be considered a fully trained OpenWebText model.

#### Current hypothesis

My current hypothesis is that model-03 has the opposite capacity mismatch from model-01:

- model-01 had too many model parameters for the amount of training data;
- model-03 may have too many vocabulary entries for the useful capacity of its small transformer body.

The 30,993-character vocabulary includes rare Unicode characters that may receive very little training signal while still consuming embedding capacity and expanding the next-token prediction space. The vocabulary increase alone accounts for most of the jump from the 0.49M model to the 3.96M model.

This is a working theory, not a conclusion established by the current run. The model may also benefit from substantially more optimization steps, improved document boundaries, or a different validation split.

### Model 04: Planned Next Steps

For model-04, I am considering two approaches:

1. **Build a subword tokenizer** with a controlled vocabulary of approximately **8K or 16K tokens**.
2. **Filter the OpenWebText input** to remove non-English or rare Unicode content before building the vocabulary.

The goal is to test whether reducing the prediction vocabulary makes better use of the available model capacity and 8 GB VRAM budget. These are planned experiments; neither approach has been validated yet.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- macOS, Linux, or Windows
- Apple Silicon GPU (MPS), NVIDIA GPU (CUDA), or CPU
- [Jupyter Notebook](https://jupyter.org/) to view and run [`analysis.ipynb`](analysis.ipynb)

Training selects MPS, CUDA, or CPU automatically. Jupyter is only needed for the analysis notebook, not for training or generation.

## Quick Start

Clone the repository and install its dependencies:

```bash
git clone https://github.com/fabstorres/llm-from-scratch.git
cd llm-from-scratch
uv sync
```

### Train on Shakespeare

```bash
uv run python train.py data/shakespeare.txt
```

The data path is optional; without one, `train.py` uses `./data/shakespeare.txt`. The current executable configuration is the 3-layer, 4-head, width-112 architecture used by model-02, with batch size 16.

During training, the script:

- evaluates validation loss every 100 steps;
- prints a generated sample every 100 steps;
- saves `checkpoint_<step>.pt` every 1,000 steps;
- saves the final model as `checkpoint_final.pt`; and
- writes training history to `loss_log.json`.

### Generate Text

```bash
uv run python generate.py checkpoint_final.pt \
  --prompt "To be or not" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 40 \
  --seed 42
```

| Option | Description |
|--------|-------------|
| `--prompt` | Starting text for generation |
| `--max_new_tokens` | Number of tokens to generate |
| `--temperature` | Sampling temperature; lower values are more deterministic |
| `--top_k` | Sample only from the `k` most likely next tokens |
| `--seed` | Optional random seed for reproducible sampling |

## Current Data Pipeline

Plain-text files are loaded into memory and tokenized directly. Paths ending in `.bin` use the memory-mapped dataset loader and require adjacent metadata at `<dataset>.bin.json`.

For a folder of Parquet files containing a `text` column:

```bash
uv run python preprocess.py data/openwebtext
uv run python train.py data/openwebtext/openwebtext.bin
```

By default, preprocessing writes:

```text
data/openwebtext/openwebtext.bin
data/openwebtext/openwebtext.bin.json
```

OpenWebText, generated `.bin` files, checkpoints, loss logs, plots, and model artifacts are intentionally excluded from Git and must be downloaded or generated locally.

## Tokenization

The current runnable pipeline is character-level for both plain-text and Parquet datasets. This keeps the implementation transparent but scales poorly when a large web corpus contains tens of thousands of distinct Unicode characters.

BPE or another subword tokenizer is not yet integrated into `train.py` or `preprocess.py`. Adding a bounded tokenizer is one of the planned model-04 experiments.

## Workshop: Build It Step by Step

The original guided workshop remains available under [`docs/`](docs/). To recreate the pipeline yourself without modifying the reference implementation:

```bash
uv sync
mkdir scratchpad
cd scratchpad
```

Work through the material in order:

| Part | What You'll Write | Concepts |
|------|-------------------|----------|
| [Part 1: Tokenization](docs/01-tokenization.md) | Character-level tokenizer | Character encoding, vocabulary construction, subword-tokenization trade-offs |
| [Part 2: The Transformer](docs/02-the-transformer.md) | Full GPT model architecture | Embeddings, self-attention, layer normalization, MLP blocks |
| [Part 3: The Training Loop](docs/03-training-loop.md) | Complete training pipeline | Loss functions, AdamW, gradient clipping, learning-rate scheduling |
| [Part 4: Text Generation](docs/04-text-generation.md) | Inference and sampling | Temperature, top-k, autoregressive decoding |
| [Part 5: Putting It All Together](docs/05-putting-it-together.md) | Train on real data and experiment | Loss curves, scaling experiments, next steps |
| [Part 6: Competition](docs/06-competition.md) | Train the best AI poet | Find datasets, scale up, submit your best poem |

The original workshop says that no previous machine-learning experience is required. In my experience, however, it was difficult to follow without some background knowledge, so I recommend learning the fundamentals before starting. [3Blue1Brown's Neural Networks playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) is an excellent visual introduction to neural networks, gradient descent, and backpropagation. You should also be comfortable reading Python code.

## Architecture: GPT at a Glance

```text
Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  characters → dataset-specific token IDs
└────────┬────────┘
         ▼
┌─────────────────┐
│  Token Embed +  │  token IDs → vectors (n_embd dimensions)
│  Position Embed │  + positional information
└────────┬────────┘
         ▼
┌─────────────────┐
│  Transformer    │  × n_layer
│  Block:         │
│  ┌────────────┐ │
│  │ LayerNorm  │ │
│  │ Self-Attn  │ │  n_head parallel attention heads
│  │ + Residual │ │
│  ├────────────┤ │
│  │ LayerNorm  │ │
│  │ MLP (FFN)  │ │  expand 4×, GELU, project back
│  │ + Residual │ │
│  └────────────┘ │
└────────┬────────┘
         ▼
┌─────────────────┐
│   LayerNorm     │
│ Linear → logits │  vocabulary scores for the next token
└─────────────────┘
```

## References and Acknowledgements

- [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch) — the original workshop and codebase from which this repository evolved
- [Skylion007/OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — the dataset used for the model-03 experiment
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [GPT-2 paper (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — language models as unsupervised multitask learners
