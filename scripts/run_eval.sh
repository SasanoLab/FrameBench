#!/bin/bash
MODEL=$1

uv run python src/evaluation/eval_multi_prompts.py \
    --model $MODEL \
    --dataset cl-nagoya/jFrameBench \
    --prompt_files eval_prompt/prompt_v1.txt \
    --num 10 # 全件処理するにはここをコメントアウト