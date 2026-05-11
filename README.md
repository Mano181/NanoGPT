# NanoGPT

A minimal, educational implementation of a character-level GPT (Generative Pre-trained Transformer) built from scratch using PyTorch. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Overview

This repository walks through building a transformer-based language model step by step — starting from a simple bigram model and progressing to a small but complete GPT. The model is trained on the Tiny Shakespeare dataset and learns to generate Shakespeare-like text character by character.

## Files

| File | Description |
|------|-------------|
| `bigram.py` | Baseline bigram language model extended with multi-head self-attention |
| `microGPT.py` | Full transformer implementation with stacked blocks, feed-forward layers, and layer normalization |
| `input.txt` | Tiny Shakespeare dataset (~1MB of Shakespeare plays used for training) |

## Architecture

### `bigram.py` — Bigram + Multi-Head Attention

A simple model that uses token and positional embeddings fed through multi-head self-attention, then a linear projection to vocabulary logits.

- Token + positional embeddings
- Multi-head causal self-attention (4 heads)
- Single linear language model head

### `microGPT.py` — Full Transformer

A complete GPT-style transformer with:

- Token + positional embeddings
- Stacked Transformer blocks, each containing:
  - Multi-head causal self-attention
  - Position-wise feed-forward network
  - Residual connections + Layer Normalization
- Final layer norm before the language model head

## Hyperparameters

| Parameter | `bigram.py` | `microGPT.py` |
|-----------|-------------|---------------|
| `batch_size` | 32 | 32 |
| `block_size` | 8 | 8 |
| `n_embd` | 27 | 32 |
| `max_iters` | 30,000 | 30,000 |
| `learning_rate` | 1e-2 | 1e-3 |
| `eval_interval` | 1,000 | 1,000 |

## Requirements

- Python 3.8+
- PyTorch 2.0+

Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Download the dataset

The dataset is already included as `input.txt`. To re-download it:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Train the bigram model

```bash
python bigram.py
```

### Train the full micro-GPT

```bash
python microGPT.py
```

Training prints loss every 1,000 steps, then generates 500 characters of Shakespeare-style text at the end.

### Example output (after training)

```
WARWICK:
The king is come; deal mildly with his youth;
For young hot colts being rag'd do rage the more.
...
```

## How It Works

1. **Tokenization** — each unique character in the dataset becomes a token (character-level vocabulary, ~65 tokens for Shakespeare).
2. **Encoding/Decoding** — simple lookup tables map characters ↔ integers.
3. **Training** — cross-entropy loss on next-character prediction; optimized with AdamW.
4. **Generation** — autoregressive sampling: at each step, sample the next character from the predicted probability distribution and append it to the context.

## Learning Path

This repo is structured as a learning progression:

```
bigram.py  →  microGPT.py
   │               │
Simple           Full GPT
attention        transformer
```

Start with `bigram.py` to understand self-attention basics, then move to `microGPT.py` to see how transformer blocks compose into a complete model.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
