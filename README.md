# GPT Language Model

A minimal GPT (Generative Pre-trained Transformer) implementation built from scratch in PyTorch. This project includes both training and text generation, using the GPT-2 tokenizer from [tiktoken](https://github.com/openai/tiktoken).

---

## Overview

This project implements a decoder-only transformer language model (GPT-style) that can be trained on any text corpus and then used interactively to generate text continuations from a given prompt. It is designed for learning and experimentation rather than production use.

---

## Configuration (`parameters.json`)

All hyperparameters for the model, training, and text generation are centralized in **`parameters.json`**. Before you train or generate text, simply adjust the values in this file. The code reads this single source dynamically, so you don't need to change any Python code to experiment with different model sizes or training configurations.

A quick look at the available settings:
- **Model Size:** `vocab_size`, `embedding_dim`, `num_heads`, `num_layers`, `max_seq_len`
- **Training:** `batch_size`, `seq_len`, `learning_rate`, `num_epochs`
- **Generation:** `max_new_tokens`, `temperature`, `top_k`

---

## Project Structure

```text
.
+-- main.py              # Interactive text generation (inference)
+-- train.py             # Model training script
+-- parameters.json      # Central configuration for all hyperparameters
+-- inputs/              # Directory for raw training text files
+-- models/              # Saved model weights
```

---

## Usage

### 1. Training

Ensure your text data is inside the `inputs/` folder (e.g., `books.txt`). Adjust your model architecture and training hyperparameters in `parameters.json`.

```bash
python train.py
```

You will be prompted for:
1. **Input text name** (e.g., `books`, do not include the `.txt` extension)
2. **Model save name** (e.g., `my_gpt_model`)

### 2. Generating Text

Once you have trained heavily, you can chat with or prompt the model. Check the generation settings like `temperature` and `top_k` in `parameters.json`.

```bash
python main.py
```

You'll be asked to provide the name of the saved model (e.g., `my_gpt_model` from the `models/` directory) and you can start prompting!
