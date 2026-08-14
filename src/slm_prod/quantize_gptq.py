"""4-bit GPTQ quantization of the merged SFT model, via GPTQModel.

AutoGPTQ is deprecated as of 2026 — GPTQModel is the maintained successor
and the backend `transformers` expects for GPTQ checkpoints. Calibrating on
the same distribution the model was fine-tuned on (no_robots) keeps the
quantization error representative of the model's actual deployment inputs.
"""

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

from slm_prod.utils import load_config


def load_calibration_texts(cfg: dict) -> list[str]:
    dataset = load_dataset(cfg["calibration_dataset_id"], split=cfg["calibration_split"])
    dataset = dataset.shuffle(seed=42).select(range(cfg["num_calibration_samples"]))
    return [
        "\n".join(f"{m['role']}: {m['content']}" for m in ex["messages"])
        for ex in dataset
    ]


def main():
    gptq_cfg = load_config("gptq.yaml")

    quantize_config = QuantizeConfig(
        bits=gptq_cfg["bits"],
        group_size=gptq_cfg["group_size"],
        desc_act=gptq_cfg["desc_act"],
        sym=gptq_cfg["sym"],
        damp_percent=gptq_cfg["damp_percent"],
    )

    model = GPTQModel.load(gptq_cfg["input_dir"], quantize_config)

    calibration_texts = load_calibration_texts(gptq_cfg)
    model.quantize(calibration_texts, batch_size=1)

    model.save(gptq_cfg["output_dir"])
    print(f"GPTQ 4-bit model saved to {gptq_cfg['output_dir']}")

    if gptq_cfg.get("push_to_hub") and gptq_cfg.get("hub_model_id"):
        model.push_to_hub(gptq_cfg["hub_model_id"])
        print(f"Pushed GPTQ model to hub: {gptq_cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
