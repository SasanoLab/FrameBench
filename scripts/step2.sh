# !/bin/bash
MODEL=$1

uv run python src/2_generate_frame_qa.py \
    --model $MODEL \
    --language ja \
    --prompt_file src/prompts/make_qa_ja.toml \
    --num 10 # 全件処理するにはここをコメントアウト