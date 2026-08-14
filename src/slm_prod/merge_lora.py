"""Merge a trained LoRA adapter back into full-precision base weights.

GPTQ quantization operates on plain dense weights, not adapters, so this
step has to run between SFT and quantize_gptq.py.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from slm_prod.utils import load_config


def main():
    model_cfg = load_config("model.yaml")
    sft_cfg = load_config("sft.yaml")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_cfg["adapter_dir"])

    merged = PeftModel.from_pretrained(base_model, sft_cfg["adapter_dir"])
    merged = merged.merge_and_unload()

    merged_dir = Path(sft_cfg["merged_dir"])
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
