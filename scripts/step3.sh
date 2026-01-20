#!/bin/bash
MODEL=$1

# 人手で記述修正を行った場合は、qa_fileを以下のように変更
# --qa_file data/ja/<model>/step2+annotation/text_corrected_qa.jsonl \

uv run python src/3_gen_additional_sentences.py \
  --qa_file data/ja/<model>/step2/qa.jsonl \
  --output_dir data/ja/<model>/step3/ \
  --model $MODEL \
  --max_items 5 # 全件処理するにはここをコメントアウト