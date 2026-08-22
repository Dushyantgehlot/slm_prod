# slm_prod

**Measuring what SFT and quantization actually do to a small language model — end to end, on a free Colab GPU.**

This project takes [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B) (Google DeepMind, released April 2026 — 8B total / 4.5B effective parameters, multimodal text+image+audio) through a full lifecycle:

```
 base model  ──SFT (QLoRA)──▶  fine-tuned model  ──GPTQ (4-bit)──▶  quantized model
      │                             │                                    │
      └─────────────────────────────┴──────────────  lm-evaluation-harness  ────────────┘
                                              (same benchmark suite, all three checkpoints)
```

The deliverable isn't "a fine-tuned model" — it's a **before/after/after-that comparison table** that isolates the effect of each step: how much does instruction tuning move the needle, and how much quality do you give up (or not) by quantizing to 4-bit for deployment.

## Why this exists

Most portfolio SFT projects stop at "I fine-tuned a model." This one is built to answer the question a reviewer actually asks: *does it work, and what did each stage cost you?* Every stage is benchmarked independently and the results are reproducible end-to-end on free-tier compute — no paid cloud GPU required.

## Pipeline

| Stage | What happens | Tool | Where |
|---|---|---|---|
| 1. Data prep | Format [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) into the Gemma 4 chat template | `src/slm_prod/data.py` | [`notebooks/01_sft_train_qlora.ipynb`](notebooks/01_sft_train_qlora.ipynb) |
| 2. SFT | QLoRA (4-bit NF4, rank-16 adapters) fine-tune of `google/gemma-4-E4B` | Unsloth + TRL `SFTTrainer` | [`src/slm_prod/train_sft.py`](src/slm_prod/train_sft.py) |
| 3. Merge | Fold LoRA adapters back into full-precision weights | PEFT | [`src/slm_prod/merge_lora.py`](src/slm_prod/merge_lora.py) |
| 4. Quantize | 4-bit GPTQ quantization of the merged SFT model | GPTQModel | [`src/slm_prod/quantize_gptq.py`](src/slm_prod/quantize_gptq.py) |
| 5. Evaluate | Same benchmark suite run against base / SFT / GPTQ checkpoints | EleutherAI `lm-evaluation-harness` | [`eval/run_eval.sh`](eval/run_eval.sh) |
| 6. Analyze | Aggregate results into a comparison table + plots | pandas / matplotlib | [`notebooks/04_results_analysis.ipynb`](notebooks/04_results_analysis.ipynb) |

## Why this dataset, why this model

- **Model — `google/gemma-4-E4B`**: the smaller of the two dense Gemma 4 variants (8B total, 4.5B effective params via per-layer embeddings), chosen as the target model for this project. Documented to fine-tune on ~10GB VRAM with QLoRA — comfortably inside a free Colab T4's 16GB.
- **Dataset — `HuggingFaceH4/no_robots`**: a small (~10K), human-written, permissively-licensed (CC-BY-4.0) instruction dataset. Chosen deliberately *instead of* `mlabonne/FineTome-100k` (the dataset used in the reference Unsloth Gemma 4 recipe this project's training config was benchmarked against) so results here reflect an independent run rather than a reproduction.
- **Training library — Unsloth**: chosen specifically because it roughly halves VRAM and training time versus vanilla PEFT/bitsandbytes, which is the difference between fitting in a free Colab session and not.
- **LoRA targeting — regex, not a plain module list**: E4B is multimodal (text + image + audio). Its vision/audio towers reuse the same leaf names (`q_proj`, `k_proj`, ...) as the language model but wrap them in a custom `Gemma4ClippableLinear` module that PEFT can't adapt — a plain `target_modules=["q_proj", ...]` list matches those too and PEFT's type-check rejects the match before any exclusion rule can apply. `configs/lora.yaml` instead uses a full-path regex scoped to `language_model.layers.*`, matching only the text tower's real `nn.Linear` projections. See the comment in that file for the PEFT version this requires. Since the vision/audio towers are never trained, `configs/model.yaml` also sets `text_only: true` so Unsloth skips loading them at all — on a T4 (no bf16 support) leaving them loaded in float32 fallback is enough VRAM pressure to trigger spurious CPU/disk offload errors from bnb's 4-bit quantizer.
- **Quantization — GPTQModel**: AutoGPTQ is deprecated as of 2026; GPTQModel is the maintained successor and the backend `transformers` now expects for GPTQ checkpoints.

## Results

> Populate this table by running `notebooks/04_results_analysis.ipynb` after all three eval passes complete. Numbers below are placeholders until a run is recorded.

| Benchmark | Base (`gemma-4-E4B`) | + SFT (QLoRA) | + GPTQ (4-bit) | Δ SFT | Δ Quantization |
|---|---|---|---|---|---|
| MMLU (5-shot) | — | — | — | — | — |
| ARC-Challenge (25-shot) | — | — | — | — | — |
| HellaSwag (10-shot) | — | — | — | — | — |
| GSM8K (5-shot) | — | — | — | — | — |
| TruthfulQA (0-shot) | — | — | — | — | — |
| IFEval (0-shot) | — | — | — | — | — |
| **VRAM (inference)** | — | — | — | — | — |
| **Latency (tok/s)** | — | — | — | — | — |

Full raw results land in [`eval/results/`](eval/results/); the narrative writeup goes in [`reports/results_summary.md`](reports/results_summary.md).

## Project structure

```
slm_prod/
├── configs/              # YAML configs for model, LoRA, SFT, GPTQ, eval tasks
├── data/                 # dataset card + preprocessing notes (no raw data committed)
├── notebooks/            # Colab-first pipeline, run in numeric order
├── src/slm_prod/         # importable, testable versions of the notebook logic
├── eval/                 # lm-eval-harness wrapper, task list, results (jsonl gitignored)
├── reports/               # human-readable writeup + figures
├── scripts/               # one-off utilities (push to HF Hub, etc.)
└── tests/                 # unit tests for data formatting logic
```

Notebooks are the primary interface (this is a free-Colab-GPU project — reproducibility on someone else's Google account matters more than a CLI). `src/slm_prod/` holds the same logic as plain, testable Python modules the notebooks import, so nothing important lives only inside a notebook cell.

## Reproducing this

1. Open [`notebooks/00_setup_colab.ipynb`](notebooks/00_setup_colab.ipynb) in Colab, select a T4 GPU runtime, run all cells (installs deps, verifies GPU + VRAM).
2. Run [`notebooks/01_sft_train_qlora.ipynb`](notebooks/01_sft_train_qlora.ipynb) — trains and pushes the LoRA adapter (and merged model) to your HF Hub account.
3. Run [`notebooks/02_merge_and_quantize_gptq.ipynb`](notebooks/02_merge_and_quantize_gptq.ipynb) — merges the adapter and produces a 4-bit GPTQ checkpoint.
4. Run [`notebooks/03_eval_base_sft_gptq.ipynb`](notebooks/03_eval_base_sft_gptq.ipynb) — runs `lm-evaluation-harness` against all three checkpoints.
5. Run [`notebooks/04_results_analysis.ipynb`](notebooks/04_results_analysis.ipynb) — builds the comparison table/plots and fills in `reports/results_summary.md`.

Equivalent CLI entrypoints exist under `src/slm_prod/` and `eval/run_eval.sh` for running outside Colab (see `Makefile`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN (and WANDB_API_KEY if used)
```

`requirements-colab.txt` is the trimmed set actually installed inside the notebooks (Colab ships some deps preinstalled).

## License

Code in this repo: [MIT](LICENSE). Gemma 4 weights are distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (Apache 2.0-style, with acceptable-use conditions) — review before redistributing fine-tuned weights.
