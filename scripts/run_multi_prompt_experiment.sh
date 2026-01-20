#!/bin/bash
# 複数プロンプト×複数モデルの実験スクリプト
set -e

# === vLLMモデル一覧 ===
# 実行したいモデルのコメントを外す
VLLM_MODELS=(
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-32B"
    # "llm-jp/llm-jp-3.1-1.8b-instruct4"
    # "llm-jp/llm-jp-3.1-13b-instruct4"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # "google/gemma-3-1b-it"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-it"
    # "openai/gpt-oss-20b"
    # "openai/gpt-oss-120b"
    # "sbintuitions/sarashina2.2-3b-instruct-v0.1"
)

# === OpenAIモデル一覧 ===
OPENAI_MODELS=(
    # "gpt-5"
    # "gpt-5-nano"
    # "gpt-4o"
    # "gpt-4o-mini"
    # "o4-mini"
)

# === プロンプト一覧 ===
# 実行したいプロンプトのコメントを外す
PROMPT_FILES=(
    # "eval_prompt/prompt_v1.txt"
    # "eval_prompt/prompt_v2.txt"
    # "eval_prompt/prompt_v3.txt"
    # "eval_prompt/prompt_v4.txt"
    # "eval_prompt/prompt_v5.txt"
    "eval_prompt/prompt_v6.txt"
    "eval_prompt/prompt_v7.txt"
    "eval_prompt/prompt_v8.txt"
    "eval_prompt/prompt_v9.txt"
)

# === vLLM設定 ===
VLLM_TENSOR_PARALLEL_SIZE=4
# VLLM_MAX_MODEL_LEN=1300000  # KVキャッシュメモリ制限に基づく最大長（推奨: 1329104以下）
VLLM_GPU_MEMORY_UTILIZATION=0.75  # GPUメモリ使用率（OOMが発生する場合は0.7-0.75に下げる）
VLLM_MAX_NUM_SEQS=128  # 同時に処理する最大シーケンス数（OOMが発生する場合は64-128に下げる、デフォルト: 256）
VLLM_ENABLE_THINKING=false  # true or false
# VLLM_REASONING_EFFORT="low"  # low, medium, high（gpt-ossなど対応モデル用）
                                # 注: VLLM_REASONING_EFFORTを指定すると自動的にenable_thinking=trueになります
# VLLM_MAX_TOKENS=8192  # 通常モードでの最大出力トークン数（デフォルト: 8192）
# VLLM_THINKING_MAX_TOKENS=32768  # 思考モードでの思考部分の最大出力トークン数（デフォルト: 32768）
# VLLM_ANSWER_MAX_TOKENS=10  # 思考モードでの回答部分の最大出力トークン数（デフォルト: 10）

# === OpenAI設定 ===
OPENAI_REASONING_EFFORT="medium"  # low, medium, high
OPENAI_USE_STRUCTURED_OUTPUTS=true  # true or false

# === 共通設定 ===
NUM=0  # 0で全問
SEED="" # ランダムシード（Noneにする場合は空文字列に設定: SEED=""）

echo "================================"
echo "複数プロンプト×複数モデル実験開始"
echo "================================"
echo "プロンプト数: ${#PROMPT_FILES[@]}"
echo "vLLMモデル数: ${#VLLM_MODELS[@]}"
echo "OpenAIモデル数: ${#OPENAI_MODELS[@]}"
echo "総実験数: $((${#PROMPT_FILES[@]} * (${#VLLM_MODELS[@]} + ${#OPENAI_MODELS[@]})))"
echo "================================"
echo ""

# 実験カウンター
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=$((${#PROMPT_FILES[@]} * (${#VLLM_MODELS[@]} + ${#OPENAI_MODELS[@]})))

# プロンプトごとにループ
for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
    echo ""
    echo "========================================"
    echo "プロンプト: $PROMPT_FILE"
    echo "========================================"
    
    # vLLMモデルの実行
    if [ ${#VLLM_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "=== vLLMモデルを実行 ==="
        for MODEL in "${VLLM_MODELS[@]}"; do
            EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
            echo ""
            echo "--- 実験 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS ---"
            echo "モデル: $MODEL"
            echo "プロンプト: $PROMPT_FILE"
            
            CMD="uv run python src/eval_multi_prompts.py"
            CMD="$CMD --model $MODEL"
            CMD="$CMD --num $NUM"
            CMD="$CMD --prompt_files $PROMPT_FILE"
            CMD="$CMD --tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE"
            # CMD="$CMD --max_model_len $VLLM_MAX_MODEL_LEN"
            CMD="$CMD --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION"
            CMD="$CMD --max_num_seqs $VLLM_MAX_NUM_SEQS"
            
            if [ -n "$VLLM_MAX_TOKENS" ]; then
                CMD="$CMD --max_tokens $VLLM_MAX_TOKENS"
            fi
            
            if [ -n "$VLLM_THINKING_MAX_TOKENS" ]; then
                CMD="$CMD --thinking_max_tokens $VLLM_THINKING_MAX_TOKENS"
            fi
            
            if [ -n "$VLLM_ANSWER_MAX_TOKENS" ]; then
                CMD="$CMD --answer_max_tokens $VLLM_ANSWER_MAX_TOKENS"
            fi
            
            if [ -n "$SEED" ]; then
                CMD="$CMD --seed $SEED"
            fi
            
            if [ "$VLLM_ENABLE_THINKING" = true ]; then
                CMD="$CMD --enable_thinking"
            fi
            
            if [ -n "$VLLM_REASONING_EFFORT" ]; then
                CMD="$CMD --reasoning_effort $VLLM_REASONING_EFFORT"
            fi
            
            echo "実行: $CMD"
            eval $CMD || echo "警告: $MODEL ($PROMPT_FILE) の実行中にエラーが発生しました"
            echo ""
        done
    fi

    # OpenAIモデルの実行
    if [ ${#OPENAI_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "=== OpenAIモデルを実行 ==="
        for MODEL in "${OPENAI_MODELS[@]}"; do
            EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
            echo ""
            echo "--- 実験 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS ---"
            echo "モデル: $MODEL"
            echo "プロンプト: $PROMPT_FILE"
            
            CMD="uv run python src/eval_multi_prompts.py"
            CMD="$CMD --model $MODEL"
            CMD="$CMD --num $NUM"
            CMD="$CMD --prompt_files $PROMPT_FILE"
            CMD="$CMD --reasoning_effort $OPENAI_REASONING_EFFORT"
            
            if [ -n "$SEED" ]; then
                CMD="$CMD --seed $SEED"
            fi
            
            if [ "$OPENAI_USE_STRUCTURED_OUTPUTS" = true ]; then
                CMD="$CMD --use_structured_outputs"
            fi
            
            echo "実行: $CMD"
            eval $CMD || echo "警告: $MODEL ($PROMPT_FILE) の実行中にエラーが発生しました"
            echo ""
        done
    fi
done

echo ""
echo "========================================"
echo "すべての実験が完了しました"
echo "========================================"
echo "実行されたプロンプト:"
for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
    echo "  - $PROMPT_FILE"
done
echo ""
echo "実行されたモデル:"
for MODEL in "${VLLM_MODELS[@]}" "${OPENAI_MODELS[@]}"; do
    echo "  - $MODEL"
done
echo ""
echo "総実験数: $EXPERIMENT_COUNT"
echo "結果は output/ja/four_choice_tsv/${NUM}/ 以下に保存されています"
echo "各モデルディレクトリ内にプロンプト名のフォルダが作成されています"
echo "========================================"
