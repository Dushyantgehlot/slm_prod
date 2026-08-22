"""QLoRA SFT of google/gemma-4-E4B on HuggingFaceH4/no_robots.

Runs standalone (`python -m slm_prod.train_sft`) or from
notebooks/01_sft_train_qlora.ipynb. Designed to fit a free Colab T4 (16GB).
"""

from pathlib import Path

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from slm_prod.data import load_sft_dataset
from slm_prod.utils import load_config


def main():
    model_cfg = load_config("model.yaml")
    lora_cfg = load_config("lora.yaml")
    sft_cfg = load_config("sft.yaml")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_id"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,  # let Unsloth pick the best dtype for the detected GPU
        load_in_4bit=model_cfg["load_in_4bit"],
        text_only=model_cfg["text_only"],
    )
    if model_cfg.get("chat_template"):
        tokenizer.chat_template = FastLanguageModel.get_chat_template(
            tokenizer, model_cfg["chat_template"]
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
    )

    train_dataset = load_sft_dataset(tokenizer, split=sft_cfg["dataset_split"])
    eval_dataset = load_sft_dataset(tokenizer, split=sft_cfg["eval_split"])

    output_dir = Path(sft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        num_train_epochs=sft_cfg["num_train_epochs"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        weight_decay=sft_cfg["weight_decay"],
        optim=sft_cfg["optim"],
        logging_steps=sft_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=sft_cfg["eval_steps"],
        save_steps=sft_cfg["save_steps"],
        save_total_limit=sft_cfg["save_total_limit"],
        seed=sft_cfg["seed"],
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        packing=False,
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    adapter_dir = sft_cfg["adapter_dir"]
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    if sft_cfg.get("push_to_hub") and sft_cfg.get("hub_model_id"):
        model.push_to_hub(sft_cfg["hub_model_id"])
        tokenizer.push_to_hub(sft_cfg["hub_model_id"])
        print(f"Pushed adapter to hub: {sft_cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
