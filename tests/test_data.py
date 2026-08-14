from unittest.mock import MagicMock

from slm_prod.data import format_example


def test_format_example_calls_chat_template_with_messages():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered chat text>"

    example = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }

    result = format_example(example, tokenizer)

    tokenizer.apply_chat_template.assert_called_once_with(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    assert result == {"text": "<rendered chat text>"}
