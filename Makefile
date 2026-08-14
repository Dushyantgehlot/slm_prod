.PHONY: install train merge quantize eval-base eval-sft eval-gptq eval-all test

install:
	pip install -e .
	pip install -r requirements.txt

train:
	python -m slm_prod.train_sft

merge:
	python -m slm_prod.merge_lora

quantize:
	python -m slm_prod.quantize_gptq

eval-base:
	eval/run_eval.sh base "pretrained=google/gemma-4-E4B,dtype=bfloat16"

eval-sft:
	eval/run_eval.sh sft "pretrained=models/sft-gemma-4-e4b-no_robots-merged,dtype=bfloat16"

eval-gptq:
	eval/run_eval.sh gptq "pretrained=models/sft-gemma-4-e4b-no_robots-gptq4bit"

eval-all: eval-base eval-sft eval-gptq

test:
	pytest tests/ -v
