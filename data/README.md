# Dataset

**Source:** [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
**License:** CC-BY-4.0
**Size:** ~10,000 human-written instruction/response pairs (9,500 train / 500 test)

## Why this dataset

This project's training config (LoRA rank, learning rate, `use_gradient_checkpointing="unsloth"`)
was benchmarked against Unsloth's published Gemma 4 fine-tuning recipe, which trains on
`mlabonne/FineTome-100k`. This project deliberately trains on a **different** dataset —
`no_robots` — so the results here are an independent measurement of the SFT/quantization
effect, not a reproduction of someone else's numbers on their own data.

`no_robots` was chosen over other common alternatives (Alpaca, Dolly-15k, OASST) because:

- It's **human-written**, not model-distilled — cleaner signal for measuring instruction-following
  gains without also measuring "did it learn to sound like GPT-4."
- It's **small enough** (~10K rows) to train multiple epochs on a free Colab T4 within a single
  session, which matters given the project's free-GPU-quota constraint.
- It ships a held-out **test split**, used as the eval split during training for loss tracking
  (separate from the downstream `lm-evaluation-harness` benchmark suite).

## No raw data is committed

`no_robots` is loaded directly from the Hugging Face Hub at training time via
[`src/slm_prod/data.py`](../src/slm_prod/data.py) — nothing is vendored into this repo.

## Formatting

Each example is a `messages` list (`system`/`user`/`assistant` turns) rendered through the
Gemma 4 chat template before tokenization. See `format_example()` in
[`src/slm_prod/data.py`](../src/slm_prod/data.py) for the exact transform.
