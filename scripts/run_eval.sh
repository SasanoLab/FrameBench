#!/bin/bash
MODEL=$1
source .venv/bin/activate  
uv pip install --upgrade transformers   
python src/evaluation/eval_multi_prompts.py \
    --model $MODEL \
    --dataset cl-nagoya/jFrameBench \
    --prompt_files eval_prompt/prompt_v1.txt \
    --tensor_parallel_size 4 \
    --enable_thinking \
    --num 10 # 全件処理するにはここをコメントアウト