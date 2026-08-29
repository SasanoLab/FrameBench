#!/bin/bash
set -euo pipefail

MODEL="${1:-gpt-4.1-mini}"
LANGUAGE="${LANGUAGE:-ja}"
MAX_ITEMS="${MAX_ITEMS:-5}"
DATA_ROOT="${DATA_ROOT:-data}"
QA_FILE="${QA_FILE:-data/${LANGUAGE}/${MODEL}/step2/qa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/${LANGUAGE}/${MODEL}/step3/}"

# 人手で記述修正を行った場合は、qa_fileを以下のように変更
# --qa_file data/ja/<model>/step2+annotation/text_corrected_qa.jsonl \

args=(
  --qa_file "$QA_FILE"
  --output_dir "$OUTPUT_DIR"
  --data_root "$DATA_ROOT"
  --model "$MODEL"
  --language "$LANGUAGE"
)

if [[ -n "$MAX_ITEMS" ]]; then
  args+=(--max_items "$MAX_ITEMS")
fi

uv run python src/generation/3_gen_additional_sentences.py \
  "${args[@]}"