#!/usr/bin/env bash
# Run the shared lm-evaluation-harness task suite against one model checkpoint.
#
# Usage:
#   eval/run_eval.sh <label> <model_args> [limit]
#
# Examples:
#   eval/run_eval.sh base  "pretrained=google/gemma-4-E4B,dtype=bfloat16"
#   eval/run_eval.sh sft   "pretrained=google/gemma-4-E4B,peft=adapters/sft-gemma-4-e4b-no_robots,dtype=bfloat16"
#   eval/run_eval.sh gptq  "pretrained=models/sft-gemma-4-e4b-no_robots-gptq4bit"
#   eval/run_eval.sh base  "pretrained=google/gemma-4-E4B,dtype=bfloat16" 20   # smoke test, 20 examples/task
#
# `label` becomes the results subdirectory: eval/results/<label>/results.json
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <label> <model_args> [limit]" >&2
  exit 1
fi

LABEL="$1"
MODEL_ARGS="$2"
LIMIT="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="$(paste -sd, "$SCRIPT_DIR/tasks.txt")"
OUTPUT_DIR="$SCRIPT_DIR/results/$LABEL"
mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --device cuda:0 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR" \
  "${LIMIT_ARGS[@]}"

echo "Results written to $OUTPUT_DIR"
