from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config from configs/ by filename, e.g. load_config('sft.yaml')."""
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)
