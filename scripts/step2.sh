# !/bin/bash
set -euo pipefail

MODEL="${1:-gpt-4.1-mini}"
LANGUAGE="${LANGUAGE:-ja}"
NUM="${NUM:-10}"
NUM_PAIRS="${NUM_PAIRS:-2}"
DATA_ROOT="${DATA_ROOT:-data}"
PROMPT_FILE="${PROMPT_FILE:-src/generation/prompts/make_qa_${LANGUAGE}.toml}"

args=(
    --model "$MODEL"
    --language "$LANGUAGE"
    --prompt_file "$PROMPT_FILE"
    --data_root "$DATA_ROOT"
    --num_pairs "$NUM_PAIRS"
)

if [[ -n "$NUM" ]]; then
    args+=(--num "$NUM")
fi

uv run python src/generation/2_generate_frame_qa.py \
    "${args[@]}"