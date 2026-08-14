"""Load and format HuggingFaceH4/no_robots for SFT.

See data/README.md for why this dataset was chosen.
"""

from datasets import Dataset, load_dataset


def format_example(example: dict, tokenizer) -> dict:
    """Render one no_robots `messages` example through the model's chat template.

    no_robots rows already look like: {"messages": [{"role": ..., "content": ...}, ...]}
    which is exactly what apply_chat_template expects — no reshaping needed.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_sft_dataset(tokenizer, split: str = "train") -> Dataset:
    """Load a no_robots split and format it into a single `text` column ready for SFTTrainer."""
    dataset = load_dataset("HuggingFaceH4/no_robots", split=split)
    return dataset.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=dataset.column_names,
    )
