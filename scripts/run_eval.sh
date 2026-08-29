#!/bin/bash
# 使い方: run_eval.sh MODEL [ja|en] [API_CONCURRENCY] [python への追加引数...]
# 例: run_eval.sh Qwen/Qwen3-4B ja 10 --enable_thinking
set -euo pipefail
MODEL=$1
shift || exit 1

LANGUAGE=ja
API_CONCURRENCY=10

if [[ "${1:-}" == "ja" || "${1:-}" == "en" ]]; then
  LANGUAGE=$1
  shift
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    API_CONCURRENCY=$1
    shift
  fi
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  API_CONCURRENCY=$1
  shift
  if [[ "${1:-}" == "ja" || "${1:-}" == "en" ]]; then
    LANGUAGE=$1
    shift
  fi
fi

# source .venv/bin/activate
# uv pip install --upgrade transformers
uv run python src/evaluation/eval_multi_prompts.py \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --tensor_parallel_size 4 \
    --swap_statements \
    --api_concurrency "$API_CONCURRENCY" \
    "$@"
# 追加引数の例: --num 10 --prompt_files eval_prompt/en/prompt_v1.txt --enable_thinking
