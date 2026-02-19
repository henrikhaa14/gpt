# GPT Language Model

A minimal GPT (Generative Pre-trained Transformer) implementation built from scratch in PyTorch. This project includes both training and text generation, using the GPT-2 tokenizer from [tiktoken](https://github.com/openai/tiktoken).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Text Generation](#text-generation)
- [How It Works](#how-it-works)

---

## Overview

This project implements a decoder-only transformer language model (GPT-style) that can be trained on any text corpus and then used interactively to generate text continuations from a given prompt. It is designed for learning and experimentation rather than production use.

---

## Architecture

The model follows the standard GPT architecture:

```
Input Token IDs
       │
       ▼
┌──────────────┐
│ Token Embed  │  Maps each token ID to a dense vector
│  + Pos Embed │  Adds positional information
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│   Transformer Block (×N)     │
│                              │
│  ┌────────────────────────┐  │
│  │ Layer Norm             │  │
│  │ Multi-Head Self-Attn   │  │  Tokens attend to all previous tokens
│  │ + Residual Connection  │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ Layer Norm             │  │
│  │ Feed-Forward Network   │  │  Position-wise MLP (expand → GELU → contract)
│  │ + Residual Connection  │  │
│  └────────────────────────┘  │
│                              │
└──────────────┬───────────────┘
               │
               ▼
        ┌─────────────┐
        │  Layer Norm  │
        │  Linear Head │  Projects back to vocabulary logits
        └──────┬──────┘
               │
               ▼
     Next-token probabilities
```

### Components

| Component | Description |
|---|---|
| **SelfAttention** | Multi-head self-attention with causal masking. Computes Q, K, V via a single fused linear projection, applies scaled dot-product attention, and masks future positions to preserve autoregressive ordering. |
| **FeedForward** | Two-layer MLP with GELU activation. Expands the embedding dimension by 4× then contracts back, allowing the model to learn complex per-position transformations. |
| **TransformerBlock** | Combines attention and feed-forward sub-layers with pre-norm (LayerNorm before each sub-layer) and residual connections for stable training. |
| **GPT** | The full model. Stacks token embeddings, learned positional embeddings, N transformer blocks, a final LayerNorm, and a linear output head mapping to vocabulary logits. Also includes the `generate()` method for autoregressive text generation. |

---

## Project Structure

```
.
├── main.py              # Interactive text generation (inference)
├── train.py             # Model training script
├── parameters.json      # All model & training hyperparameters
├── README.md
├── inputs/
│   ├── input_01.txt     # Custom training text
│   └── input_02.txt
└── models/
    ├── model_01.pt      # Pre-trained model weights
    └── model_02.pt
```

---

## Configuration

All hyperparameters are centralized in `parameters.json`:

### Model Parameters

| Parameter | Default | Description |
|---|---|---|
| `vocab_size` | 50,257 | Vocabulary size (matches GPT-2 BPE tokenizer) |
| `embedding_dim` | 512 | Dimensionality of token and positional embeddings |
| `num_heads` | 3 | Number of attention heads in each self-attention layer |
| `num_layers` | 3 | Number of stacked transformer blocks |
| `max_seq_len` | 512 | Maximum context window (number of tokens the model can see at once) |

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 32 | Batch size (currently unused — training processes one sequence at a time) |
| `seq_len` | 512 | Sequence length for each training example |
| `learning_rate` | 1e-4 | AdamW optimizer learning rate |
| `num_epochs` | 3 | Number of full passes through the training data |

### Generation Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_new_tokens` | 100 | Number of tokens to generate per prompt |
| `temperature` | 0.7 | Controls randomness: lower values → more deterministic, higher → more creative |
| `top_k` | 50 | Restricts sampling to the top K most likely tokens at each step |

---

## Training

### Prerequisites

```bash
pip install torch tiktoken tqdm
```

### Running Training

```bash
python train.py
```

You will be prompted for:
1. **Input file name** — the name (without `.txt`) of a text file in the `inputs/` directory (e.g., `tiny-shakespeare`)
2. **Model name** — the name for the saved model (saved to `models/<name>.pt`)

### Training Process

1. The input text file is loaded and tokenized using the GPT-2 BPE tokenizer.
2. The tokenized text is split into non-overlapping sequences of length `seq_len + 1`.
3. For each sequence, the first `seq_len` tokens serve as **input** and the last `seq_len` tokens (shifted by one) serve as **target**.
4. The model is trained with **cross-entropy loss** and the **AdamW optimizer**.
5. A progress bar (via `tqdm`) shows per-batch loss; average loss is printed at the end of each epoch.
6. After all epochs, model weights are saved to `models/<model_name>.pt`.

---

## Text Generation

### Running Inference

```bash
python main.py
```

You will be prompted for:
1. **Model name** — the name of a trained model in the `models/` directory (without `.pt`)

Then enter text prompts interactively. Type `quit` to exit.

### Generation Process

1. The prompt is tokenized into token IDs using the GPT-2 tokenizer.
2. The model autoregressively generates one token at a time:
   - The current token sequence (cropped to `max_seq_len`) is fed through the model.
   - Logits for the last position are divided by `temperature`.
   - **Top-k filtering** zeroes out all but the top K most likely tokens.
   - A token is sampled from the resulting probability distribution.
   - The sampled token is appended and the process repeats.
3. The full sequence (prompt + generated tokens) is decoded back to text and printed.

### Example

```
Prompt: To be, or not to be
```

The model will continue the text based on what it learned from training data.

---

## How It Works

### Tokenization

This project uses OpenAI's **tiktoken** library with the `gpt2` encoding, a byte-pair encoding (BPE) tokenizer with a 50,257-token vocabulary. This is the same tokenizer used by GPT-2, providing subword tokenization that balances vocabulary size with sequence length.

### Self-Attention Mechanism

Each attention head computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q$, $K$, $V$ are the query, key, and value matrices projected from the input, and $d_k$ is the head dimension. A **causal mask** is applied before the softmax so that each token can only attend to itself and preceding tokens — this is what makes the model autoregressive.

### Pre-Norm Residual Connections

Each sub-layer follows the pattern:

$$x = x + \text{SubLayer}(\text{LayerNorm}(x))$$

Applying LayerNorm *before* the sub-layer (pre-norm) is more stable for training compared to the original post-norm formulation.

### Temperature and Top-k Sampling

During generation:
- **Temperature** ($\tau$): Logits are divided by $\tau$ before softmax. Values below 1.0 sharpen the distribution (more confident picks), values above 1.0 flatten it (more random).
- **Top-k**: Only the $k$ highest-probability tokens are considered; the rest are masked to $-\infty$. This prevents sampling from the long tail of unlikely tokens.
