#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
複数のプロンプトで四択問題を評価し、結果を集計するスクリプト
使用例:
  # eval_promptフォルダ内の全プロンプトで実験
  uv run python src/eval_multi_prompts.py --model gpt-4o --dataset data/ja/gpt5/qa.jsonl
  
  # HuggingFace Datasetから読み込む場合
  uv run python src/eval_multi_prompts.py --model gpt-4o --dataset cl-nagoya/jFrameBench
  
  # 特定のプロンプトファイルと問題数を指定
  uv run python src/eval_multi_prompts.py --model gpt-4o --num 100 --prompt_files eval_prompt/prompt_v1.txt eval_prompt/prompt_v2.txt --dataset cl-nagoya/jFrameBench
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from eval_common_four_choice import (
    load_problems,
    create_messages,
    process_results,
    print_stats,
    save_results,
    load_prompt_template,
)

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))


OPENAI_MODEL_CAPABILITIES = {
    # === GPT-5シリーズ ===
    "gpt-5.1": {"reasoning_effort": ["none", "minimal", "low", "medium", "high"], "structured_outputs": True},
    "gpt-5": {"reasoning_effort": ["minimal", "low", "medium", "high"], "structured_outputs": True},
    "gpt-5-nano": {"reasoning_effort": ["minimal", "low", "medium", "high"], "structured_outputs": True},
    
    # === o-シリーズ（推論モデル） ===
    "o4-mini": {"reasoning_effort": ["low", "medium", "high"], "structured_outputs": True},
    
    # === GPT-4oシリーズ（reasoning_effort非対応） ===
    "gpt-4o": {"reasoning_effort": None, "structured_outputs": True},
    "gpt-4o-mini": {"reasoning_effort": None, "structured_outputs": True},
    "gpt-4-turbo": {"reasoning_effort": None, "structured_outputs": False},
    "gpt-4": {"reasoning_effort": None, "structured_outputs": False},
    "gpt-3.5-turbo": {"reasoning_effort": None, "structured_outputs": False},
}

# vLLMモデルごとの機能サポート情報
VLLM_MODEL_CAPABILITIES = {
    "openai/gpt-oss-20b": {"reasoning_effort": ["low", "medium", "high"]},
    "openai/gpt-oss-120b": {"reasoning_effort": ["low", "medium", "high"]},
}

# Structured Outputs用のJSON schema（四択問題用）
CHOICE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "four_choice_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["1", "2", "3", "4"],
                    "description": "選択した回答（1, 2, 3, または4）"
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
    }
}


def detect_backend(model_name):
    """モデル名からバックエンドを自動判定"""
    openai_patterns = ['gpt-', 'o4-', 'o1-', 'o3-']
    for pattern in openai_patterns:
        if model_name.startswith(pattern):
            return 'openai'
    return 'vllm'


def get_openai_model_capabilities(model_name):
    """モデル名から機能サポート情報を取得（OpenAI用）"""
    if model_name in OPENAI_MODEL_CAPABILITIES:
        return OPENAI_MODEL_CAPABILITIES[model_name]
    
    for base_model, caps in OPENAI_MODEL_CAPABILITIES.items():
        if model_name.startswith(base_model):
            return caps
    
    print(f"警告: モデル '{model_name}' の機能情報が不明です。デフォルト設定を使用します。")
    return {"reasoning_effort": None, "structured_outputs": True}


def get_vllm_model_capabilities(model_name):
    """モデル名から機能サポート情報を取得（vLLM用）"""
    if model_name in VLLM_MODEL_CAPABILITIES:
        return VLLM_MODEL_CAPABILITIES[model_name]
    return {"reasoning_effort": None}


def run_vllm_single_prompt(args, all_problems, all_messages, output_dir):
    """vLLMを使用して単一プロンプトで推論を実行"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    
    caps = get_vllm_model_capabilities(args.model)
    supports_reasoning = caps["reasoning_effort"] is not None
    
    if args.reasoning_effort:
        if supports_reasoning:
            supported_efforts = caps["reasoning_effort"]
            if args.reasoning_effort not in supported_efforts:
                print(f"⚠️  警告: reasoning_effort '{args.reasoning_effort}' はこのモデルでサポートされていない可能性があります。")
                print(f"    サポート値: {supported_efforts}")
            if not args.enable_thinking:
                args.enable_thinking = True
        else:
            print(f"⚠️  警告: モデル '{args.model}' は reasoning_effort をサポートしていません。")
            args.reasoning_effort = None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    texts = []
    for messages in all_messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort if args.reasoning_effort else None,
        )
        texts.append(text)
    
    llm_kwargs = {
        "model": args.model, 
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)

    if args.enable_thinking:
        thinking_params = SamplingParams(
            temperature=0.6, 
            top_p=0.95, 
            top_k=20, 
            max_tokens=args.thinking_max_tokens,
            stop=["</think>"],
            include_stop_str_in_output=True
        )
        thinking_outputs = llm.generate(texts, thinking_params)
        
        answer_prompts = [
            text + output.outputs[0].text + "\n\n"
            for text, output in zip(texts, thinking_outputs)
        ]
        
        structured_outputs = StructuredOutputsParams(choice=['1', '2', '3', '4'])
        answer_params = SamplingParams(
            temperature=0.7, 
            top_p=0.8, 
            top_k=20, 
            max_tokens=args.answer_max_tokens,
            structured_outputs=structured_outputs
        )
        answer_outputs = llm.generate(answer_prompts, answer_params)
        
        generated_texts = [
            thinking_output.outputs[0].text + "\n\n" + answer_output.outputs[0].text
            for thinking_output, answer_output in zip(thinking_outputs, answer_outputs)
        ]
    else:
        structured_outputs = StructuredOutputsParams(choice=['1', '2', '3', '4'])
        sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.8, 
            top_k=20, 
            max_tokens=args.max_tokens,
            structured_outputs=structured_outputs
        )
        vllm_outputs = llm.generate(texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in vllm_outputs]
    
    all_problems, stats = process_results(all_problems, generated_texts)
    return all_problems, stats


def run_openai_single_prompt(args, all_problems, all_messages, output_dir):
    """OpenAI APIを使用して単一プロンプトで推論を実行"""
    import asyncio
    
    caps = get_openai_model_capabilities(args.model)
    supports_reasoning = caps["reasoning_effort"] is not None
    api_params = {"model": args.model}
    
    api_params["response_format"] = CHOICE_SCHEMA

    if supports_reasoning:
        if args.reasoning_effort:
            if args.reasoning_effort not in caps["reasoning_effort"]:
                print(f"警告: '{args.reasoning_effort}' はこのモデルでサポートされていません。"
                      f" 対応値: {caps['reasoning_effort']}")
            api_params["reasoning_effort"] = args.reasoning_effort
        else:
            api_params["reasoning_effort"] = "medium"
    else:
        if args.reasoning_effort:
            print("警告: このモデルはreasoning_effortをサポートしていません。無視されます。")
    
    actual_params = {
        "requested": {k: v for k, v in api_params.items() if k != "response_format"},
        "model_capabilities": caps,
        "first_response_model": None,
    }
    
    # 非同期処理を実行
    generated_texts, first_response = asyncio.run(
        run_openai_async(all_messages, api_params)
    )
    
    # 最初のレスポンス情報を記録
    if first_response:
        actual_params["first_response_model"] = first_response.model
        actual_params["system_fingerprint"] = getattr(first_response, "system_fingerprint", None)
    
    all_problems, stats = process_results(all_problems, generated_texts)
    
    params_file = output_dir / "params.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(actual_params, f, indent=2, ensure_ascii=False)
    
    return all_problems, stats


async def run_openai_async(all_messages, api_params):
    """OpenAI APIを非同期で並列実行"""
    import asyncio
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    
    async def process_single_message(i, messages):
        """単一のメッセージを処理"""
        try:
            request_params = {**api_params, "messages": messages}
            response = await client.chat.completions.create(**request_params)
            generated_text = response.choices[0].message.content
            return i, generated_text, response if i == 0 else None
        except Exception as e:
            print(f"APIエラー (index {i}): {e}")
            return i, None, None
    
    # 全てのリクエストを並列実行
    print(f"OpenAI API推論中（並列処理）: {len(all_messages)}件")
    tasks = [process_single_message(i, messages) for i, messages in enumerate(all_messages)]
    results = await asyncio.gather(*tasks)
    
    # 結果を元の順序でソート
    results_sorted = sorted(results, key=lambda x: x[0])
    generated_texts = [text for _, text, _ in results_sorted]
    first_response = results_sorted[0][2] if results_sorted else None
    
    return generated_texts, first_response


def run_single_prompt_experiment(args, prompt_file, prompt_name, all_problems_original):
    """単一のプロンプトで実験を実行"""
    import copy
    
    print(f"\n{'='*60}")
    print(f"プロンプト: {prompt_name}")
    print(f"{'='*60}")
    
    # プロンプトテンプレートを読み込む
    prompt_template = load_prompt_template(prompt_file)
    print(f"\nプロンプトテンプレート:\n{prompt_template}\n")
    
    # 問題のコピーを作成
    all_problems = copy.deepcopy(all_problems_original)
    
    # メッセージを生成
    all_messages = create_messages(all_problems, prompt_template)
    
    # 出力ディレクトリを作成
    suffix = ""
    backend = detect_backend(args.model)
    
    if backend == 'vllm':
        effective_thinking = args.enable_thinking or (args.reasoning_effort is not None)
        if effective_thinking:
            suffix = "_thinking"
        else:
            suffix = "_no_thinking"
        if args.reasoning_effort:
            suffix += f"_reasoning_{args.reasoning_effort}"
    else:  # openai
        caps = get_openai_model_capabilities(args.model)
        supports_reasoning = caps["reasoning_effort"] is not None
        if supports_reasoning and args.reasoning_effort:
            suffix = f"_reasoning_{args.reasoning_effort}"
        elif supports_reasoning:
            suffix = "_reasoning_medium"
    
    safe_model_name = args.model.replace('/', '_').replace(':', '_')
    num_str = str(args.num) if args.num is not None else "all"
    output_dir = Path(args.output_dir) / args.language / num_str / f"{safe_model_name}{suffix}" / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # バックエンドに応じて推論を実行
    if backend == 'vllm':
        all_problems, stats = run_vllm_single_prompt(args, all_problems, all_messages, output_dir)
    else:  # openai
        all_problems, stats = run_openai_single_prompt(args, all_problems, all_messages, output_dir)
    
    # 統計を表示
    print_stats(stats)
    
    # 結果を保存
    save_results(output_dir, all_problems, stats)
    
    return stats, output_dir


def aggregate_results(all_stats, prompt_names, base_output_dir):
    """複数のプロンプトの結果を集計"""
    print(f"\n{'='*60}")
    print("全プロンプトの結果集計")
    print(f"{'='*60}\n")
    
    # 各プロンプトの正答率を表示
    results_df = []
    for prompt_name, stats in zip(prompt_names, all_stats):
        results_df.append({
            'プロンプト': prompt_name,
            '正答率': f"{stats['accuracy']*100:.1f}%",
            '正解数': sum(stats['scores']),
            '総問題数': stats['total'],
            'エラー数': stats['error_count']
        })
    
    df = pd.DataFrame(results_df)
    print(df.to_string(index=False))
    
    # 平均正答率を計算
    avg_accuracy = sum(s['accuracy'] for s in all_stats) / len(all_stats)
    print(f"\n平均正答率: {avg_accuracy*100:.1f}%")
    
    # 集計結果を保存
    summary_file = base_output_dir / "aggregated_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("複数プロンプトの実験結果集計\n")
        f.write("="*60 + "\n\n")
        f.write(df.to_string(index=False) + "\n\n")
        f.write(f"平均正答率: {avg_accuracy*100:.1f}%\n")
    print(f"\n集計結果を {summary_file} に保存しました")
    
    # TSVファイルも保存
    tsv_file = base_output_dir / "aggregated_summary.tsv"
    df.to_csv(tsv_file, sep="\t", index=False)
    print(f"集計結果を {tsv_file} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description='複数のプロンプトで四択問題を評価し、結果を集計するスクリプト',
    )
    
    # 共通引数
    parser.add_argument('--num', type=int, default=None, help='解く問題数（指定しない場合は全問）')
    parser.add_argument('--model', type=str, required=True, help='評価対象のLLMモデル')
    parser.add_argument('--dataset', type=str, 
                        required=True,
                        help='データセットのパス（ローカルファイルのパスまたはHuggingFace Dataset名）')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='HuggingFace Datasetのsplit（デフォルト: train）')
    parser.add_argument('--output_dir', type=str, default='output', help='出力ディレクトリのベースパス')
    parser.add_argument('--language', type=str, default='ja', help='評価対象言語')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=['low', 'medium', 'high', 'minimal', 'none'],
                        help='推論の深さ')
    
    # プロンプト関連
    parser.add_argument('--prompt_files', type=str, nargs='+', default=None,
                        help='使用するプロンプトファイルのパス（複数指定可）。指定しない場合はeval_promptフォルダ内の全ファイルを使用')
    
    # vLLM固有の引数
    vllm_group = parser.add_argument_group('vLLM固有のオプション')
    vllm_group.add_argument('--enable_thinking', action='store_true', help='思考モードを有効にする（vLLMのみ）')
    vllm_group.add_argument('--max_model_len', type=int, default=None, 
                           help='vLLMのmax_model_len（Noneの場合はモデルのデフォルト値）')
    vllm_group.add_argument('--tensor_parallel_size', type=int, default=1, 
                           help='vLLMのtensor_parallel_size')
    vllm_group.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                           help='GPUメモリ使用率（0.0-1.0、デフォルト: 0.9）')
    vllm_group.add_argument('--max_num_seqs', type=int, default=None,
                           help='vLLMのmax_num_seqs（同時に処理する最大シーケンス数、デフォルト: 256）')
    vllm_group.add_argument('--max_tokens', type=int, default=8192,
                           help='通常モードでの最大出力トークン数（デフォルト: 8192）')
    vllm_group.add_argument('--thinking_max_tokens', type=int, default=32768,
                           help='思考モードでの思考部分の最大出力トークン数（デフォルト: 32768）')
    vllm_group.add_argument('--answer_max_tokens', type=int, default=10,
                           help='思考モードでの回答部分の最大出力トークン数（デフォルト: 10）')
    
    args = parser.parse_args()
    
    print(f"引数: {args}\n")
    
    # プロンプトファイルのリストを取得
    if args.prompt_files is None:
        # eval_promptフォルダ内の全ファイルを使用
        project_root = Path(__file__).parent.parent
        prompt_dir = project_root / "eval_prompt"
        if not prompt_dir.exists():
            print(f"エラー: {prompt_dir} が見つかりません", file=sys.stderr)
            sys.exit(1)
        
        prompt_files = sorted(prompt_dir.glob("*.txt"))
        if not prompt_files:
            print(f"エラー: {prompt_dir} にプロンプトファイルが見つかりません", file=sys.stderr)
            sys.exit(1)
    else:
        prompt_files = [Path(f) for f in args.prompt_files]
    
    print(f"使用するプロンプトファイル数: {len(prompt_files)}")
    for pf in prompt_files:
        print(f"  - {pf.name}")
    print()
    
    # 問題を読み込む（全プロンプトで共通）
    all_problems_original = load_problems(args.dataset, args.num, args.dataset_split)
    
    # 各プロンプトで実験を実行
    all_stats = []
    prompt_names = []
    
    for prompt_file in prompt_files:
        prompt_name = prompt_file.stem  # ファイル名から拡張子を除いたもの
        prompt_names.append(prompt_name)
        
        stats, output_dir = run_single_prompt_experiment(
            args, 
            prompt_file, 
            prompt_name, 
            all_problems_original
        )
        all_stats.append(stats)
    
    # 結果を集計
    suffix = ""
    backend = detect_backend(args.model)
    
    if backend == 'vllm':
        effective_thinking = args.enable_thinking or (args.reasoning_effort is not None)
        if effective_thinking:
            suffix = "_thinking"
        else:
            suffix = "_no_thinking"
        if args.reasoning_effort:
            suffix += f"_reasoning_{args.reasoning_effort}"
    else:  # openai
        caps = get_openai_model_capabilities(args.model)
        supports_reasoning = caps["reasoning_effort"] is not None
        if supports_reasoning and args.reasoning_effort:
            suffix = f"_reasoning_{args.reasoning_effort}"
        elif supports_reasoning:
            suffix = "_reasoning_medium"
    
    safe_model_name = args.model.replace('/', '_').replace(':', '_')
    num_str = str(args.num) if args.num is not None else "all"
    base_output_dir = Path(args.output_dir) / args.language / num_str / f"{safe_model_name}{suffix}"
    
    aggregate_results(all_stats, prompt_names, base_output_dir)


if __name__ == "__main__":
    main()
