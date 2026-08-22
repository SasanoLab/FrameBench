#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
複数のプロンプトで四択問題を評価し、結果を集計するスクリプト
使用例:
  # eval_prompt/jaフォルダ内の全プロンプトで実験
  uv run python src/evaluation/eval_multi_prompts.py --model gpt-4o --dataset data/ja/gpt5/qa.jsonl
  
  # HuggingFace Datasetから読み込む場合
  uv run python src/evaluation/eval_multi_prompts.py --model gpt-4o --language ja
  
  # 特定のプロンプトファイルと問題数を指定
  uv run python src/evaluation/eval_multi_prompts.py --model gpt-4o --num 100 --prompt_files eval_prompt/ja/prompt_v1.txt eval_prompt/ja/prompt_v2.txt --dataset cl-nagoya/jFrameBench

  # 文A/文Bを入れ替えたミラー問題も同一実行で評価（件数約2倍）
  uv run python src/evaluation/eval_multi_prompts.py --model gpt-4o --language ja --swap_statements
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import yaml
from dotenv import load_dotenv

# プロジェクトルートの .env を自動読み込み（存在しない場合は無視）
load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from eval_utils import (  # noqa: E402
    load_problems,
    create_messages,
    process_results,
    print_stats,
    save_results,
    load_prompt_template,
)

# モデルプロファイルをYAMLから読み込む
_DEFAULT_PROFILES_PATH = Path(__file__).parent / "model_profiles.yaml"

def _load_profiles(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

_PROFILES = _load_profiles(_DEFAULT_PROFILES_PATH)
_VLLM_DEFAULTS: dict = _PROFILES.get("vllm_defaults", {})
_MODEL_PROFILES: dict = _PROFILES.get("models", {})

DEFAULT_DATASETS = {
    "ja": "cl-nagoya/jFrameBench",
    "en": "cl-nagoya/FrameBench",
}

# Structured Outputs用のJSON schema（四択問題用）
CHOICE_VLLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["1", "2", "3", "4"],
            "description": "選択した回答（1, 2, 3, または4）",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}

CHOICE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "four_choice_answer",
        "strict": True,
        "schema": CHOICE_VLLM_JSON_SCHEMA,
    }
}


def make_choice_structured_outputs(profile: dict):
    """vLLM 用 structured output パラメータ（Closed Model 同等の JSON または choice 制約）"""
    from vllm.sampling_params import StructuredOutputsParams

    if profile.get("thinking_answer_format") == "json":
        return StructuredOutputsParams(
            json=CHOICE_VLLM_JSON_SCHEMA,
            disable_additional_properties=True,
        )
    return StructuredOutputsParams(choice=["1", "2", "3", "4"])


def get_model_profile(model_name: str) -> dict:
    """model_profiles.yaml からモデルプロファイルを取得する（前方一致フォールバック付き）"""
    if model_name in _MODEL_PROFILES:
        return _MODEL_PROFILES[model_name]
    for key, profile in _MODEL_PROFILES.items():
        if model_name.startswith(key):
            return profile
    # 未登録モデル: モデル名パターンからバックエンドを推定してデフォルト値を返す
    openai_patterns = ['gpt-', 'o4-', 'o1-', 'o3-']
    if any(model_name.startswith(p) for p in openai_patterns):
        print(f"警告: モデル '{model_name}' のプロファイルが未登録です。OpenAIデフォルトを使用します。")
        return {"backend": "openai", "reasoning_effort": None, "structured_outputs": True}
    gemini_patterns = ['gemini-']
    if any(model_name.startswith(p) for p in gemini_patterns):
        print(f"警告: モデル '{model_name}' のプロファイルが未登録です。Geminiデフォルトを使用します。")
        return {
            "backend": "openai",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key_env": "GOOGLE_API_KEY",
            "reasoning_effort": None,
            "structured_outputs": True,
        }
    claude_patterns = ['claude-']
    if any(model_name.startswith(p) for p in claude_patterns):
        print(f"警告: モデル '{model_name}' のプロファイルが未登録です。Anthropicデフォルトを使用します。")
        return {
            "backend": "openai",
            "base_url": "https://api.anthropic.com/v1/",
            "api_key_env": "ANTHROPIC_API_KEY",
            "reasoning_effort": None,
            "structured_outputs": False,
        }
    print(f"警告: モデル '{model_name}' のプロファイルが未登録です。vLLMデフォルトを使用します。")
    return {"backend": "vllm", "reasoning_effort": None}


def detect_backend(model_name: str) -> str:
    """モデルプロファイルからバックエンドを取得する"""
    return get_model_profile(model_name).get("backend", "vllm")


def get_thinking_stop(args) -> str:
    """thinkタグの終了文字列を取得する（CLI引数 > YAMLモデルプロファイル > YAMLデフォルト の優先順位）"""
    if getattr(args, "thinking_stop", None):
        return args.thinking_stop
    profile = get_model_profile(args.model)
    if "thinking_stop" in profile:
        return profile["thinking_stop"]
    return _VLLM_DEFAULTS.get("thinking_stop", "</think>")


def get_json_single_max_tokens(profile: dict, args) -> int:
    """JSON 単発生成の max_tokens（YAMLモデル > vllm_defaults > 8192）"""
    if "json_single_max_tokens" in profile:
        base = profile["json_single_max_tokens"]
    else:
        base = _VLLM_DEFAULTS.get("json_single_max_tokens", 8192)
    return max(base, args.answer_max_tokens, 32)


def _merge_sampling(phase: str, args, profile: dict) -> tuple[float, float, int]:
    """サンプリングパラメータを YAMLデフォルト → YAMLモデルプロファイル → CLI引数 の順で解決する

    phase が "thinking" の場合は thinking_sampling セクションを参照する。
    phase が "answer" の場合は answer_sampling セクションを参照する。
    回答フェーズは選択肢制約をかけるため、top_k/top_p で候補数字を落とさない設定を使う。
    phase が "sampling" の場合は sampling セクション（通常モード）を参照する。

    Returns:
        (temperature, top_p, top_k)
    """
    section_by_phase = {
        "thinking": "thinking_sampling",
        "answer": "answer_sampling",
        "sampling": "sampling",
    }
    section_key = section_by_phase[phase]
    defaults = _VLLM_DEFAULTS.get(section_key, {})
    model_sampling = profile.get(section_key, {})

    def _val(key: str, fallback):
        return model_sampling.get(key, defaults.get(key, fallback))

    temperature = _val("temperature", 0.7)
    top_p = _val("top_p", 0.8)
    top_k = _val("top_k", 20)

    # CLI引数があれば最優先
    if phase == "thinking":
        if getattr(args, "thinking_temperature", None) is not None:
            temperature = args.thinking_temperature
        if getattr(args, "thinking_top_p", None) is not None:
            top_p = args.thinking_top_p
        if getattr(args, "thinking_top_k", None) is not None:
            top_k = args.thinking_top_k
    elif phase == "answer":
        if getattr(args, "answer_temperature", None) is not None:
            temperature = args.answer_temperature
        if getattr(args, "answer_top_p", None) is not None:
            top_p = args.answer_top_p
        if getattr(args, "answer_top_k", None) is not None:
            top_k = args.answer_top_k
    else:
        if getattr(args, "temperature", None) is not None:
            temperature = args.temperature
        if getattr(args, "top_p", None) is not None:
            top_p = args.top_p
        if getattr(args, "top_k", None) is not None:
            top_k = args.top_k

    return temperature, top_p, top_k


def _format_messages_for_vllm(messages: list[dict], profile: dict) -> list[dict]:
    """モデルごとの chat_template が期待する message 形式へ寄せる。"""
    if profile.get("message_content_format") != "multimodal_list":
        return messages

    formatted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message = {**message, "content": [{"type": "text", "text": content}]}
        formatted.append(message)
    return formatted


def _message_content_to_text(content) -> str:
    """chat_template がない vLLM モデル向けに message content からテキストを取り出す。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_plain_prompt(messages: list[dict]) -> str:
    """tokenizer.chat_template がないモデルでは、単一 user prompt をそのまま使う。"""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _message_content_to_text(messages[0].get("content", ""))

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_to_text(message.get("content", ""))
        rendered.append(f"{role.upper()}: {content}")
    return "\n".join(rendered)


def append_generation_prefill(text: str, prefill: str | None) -> str:
    """指定時だけ、生成開始位置の直前に任意の文字列を補う。"""
    return text + prefill if prefill is not None else text


def resolve_generation_prefill(
    cli_prefill: str | None,
    profile: dict,
    effective_thinking: bool,
) -> str | None:
    """CLI指定を優先し、通常生成時だけモデル既定のprefillを補う。"""
    if cli_prefill is not None:
        return cli_prefill
    if effective_thinking:
        return None
    return profile.get("no_thinking_generation_prefill")


def _content_has_multimodal_part(content) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            return True
    return False


def _messages_have_multimodal_content(all_messages: list[list[dict]]) -> bool:
    return any(
        _content_has_multimodal_part(message.get("content"))
        for messages in all_messages
        for message in messages
    )


def validate_multimodal_support(args, all_messages: list[list[dict]]) -> None:
    """入力メッセージとモデルプロファイルのマルチモーダル対応を照合する。"""
    profile = get_model_profile(args.model)
    has_multimodal_input = _messages_have_multimodal_content(all_messages)
    supports_multimodal = profile.get("supports_multimodal") is True

    if has_multimodal_input and not supports_multimodal:
        raise ValueError(
            f"モデル '{args.model}' は model_profiles.yaml で supports_multimodal: true "
            "に設定されていないため、画像などのマルチモーダル入力は評価できません。"
        )

    if supports_multimodal and not has_multimodal_input and not getattr(args, "multimodal_check_reported", False):
        print("ℹ️  このモデルはマルチモーダル対応として登録されていますが、今回の入力はテキストのみです。")
        args.multimodal_check_reported = True


def run_vllm_single_prompt(args, all_problems, all_messages, output_dir):
    """vLLMを使用して単一プロンプトで推論を実行"""
    from vllm import LLM, SamplingParams
    profile = get_model_profile(args.model)
    # supports_thinking: false が明示されている場合は優先
    if profile.get("supports_thinking") is False:
        supports_reasoning = False
    else:
        supports_reasoning = profile.get("reasoning_effort") is not None

    if args.reasoning_effort:
        if supports_reasoning:
            supported_efforts = profile["reasoning_effort"]
            if args.reasoning_effort not in supported_efforts:
                print(f"⚠️  警告: reasoning_effort '{args.reasoning_effort}' はこのモデルでサポートされていない可能性があります。")
                print(f"    サポート値: {supported_efforts}")
            if not args.enable_thinking:
                args.enable_thinking = True
        else:
            print(f"⚠️  警告: モデル '{args.model}' は reasoning_effort をサポートしていません。")
            args.reasoning_effort = None

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    trust_remote_code = args.trust_remote_code or profile.get("trust_remote_code", False)
    if trust_remote_code:
        llm_kwargs["trust_remote_code"] = True

    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()

    json_single_call = (
        args.enable_thinking
        and profile.get("thinking_answer_format") == "json"
        and profile.get("thinking_harmony") is False
    )
    harmony_two_phase = args.enable_thinking and not json_single_call

    texts = []
    profile_chat_template = profile.get("chat_template")
    tokenizer_chat_template = getattr(tokenizer, "chat_template", None)
    for messages in all_messages:
        messages = _format_messages_for_vllm(messages, profile)
        if profile_chat_template or tokenizer_chat_template:
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": harmony_two_phase,
            }
            if profile_chat_template:
                chat_template_kwargs["chat_template"] = profile_chat_template
            if args.reasoning_effort:
                chat_template_kwargs["reasoning_effort"] = args.reasoning_effort
            text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        else:
            text = _format_plain_prompt(messages)
        texts.append(append_generation_prefill(text, args.generation_prefill))

    thinking_stop = get_thinking_stop(args)
    skip_special_tokens = profile.get("skip_special_tokens", True)

    if json_single_call:
        s_temp, s_top_p, s_top_k = _merge_sampling("sampling", args, profile)
        json_max_tokens = get_json_single_max_tokens(profile, args)
        sampling_params = SamplingParams(
            temperature=s_temp,
            top_p=s_top_p,
            top_k=s_top_k,
            max_tokens=json_max_tokens,
            structured_outputs=make_choice_structured_outputs(profile),
            skip_special_tokens=skip_special_tokens,
        )
        vllm_outputs = llm.generate(texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in vllm_outputs]
        thinking_stop = None
    elif harmony_two_phase:
        t_temp, t_top_p, t_top_k = _merge_sampling("thinking", args, profile)
        thinking_params = SamplingParams(
            temperature=t_temp,
            top_p=t_top_p,
            top_k=t_top_k,
            max_tokens=args.thinking_max_tokens,
            stop=[thinking_stop],
            include_stop_str_in_output=True,
            skip_special_tokens=skip_special_tokens,
        )
        thinking_outputs = llm.generate(texts, thinking_params)

        answer_prompts = [
            text + output.outputs[0].text
            for text, output in zip(texts, thinking_outputs)
        ]

        a_temp, a_top_p, a_top_k = _merge_sampling("answer", args, profile)
        answer_max_tokens = args.answer_max_tokens
        if profile.get("thinking_answer_format") == "json":
            answer_max_tokens = max(answer_max_tokens, 32)
        answer_params = SamplingParams(
            temperature=a_temp,
            top_p=a_top_p,
            top_k=a_top_k,
            max_tokens=answer_max_tokens,
            structured_outputs=make_choice_structured_outputs(profile),
            skip_special_tokens=skip_special_tokens,
        )
        answer_outputs = llm.generate(answer_prompts, answer_params)

        generated_texts = [
            thinking_output.outputs[0].text + "\n\n" + answer_output.outputs[0].text
            for thinking_output, answer_output in zip(thinking_outputs, answer_outputs)
        ]
    else:
        s_temp, s_top_p, s_top_k = _merge_sampling("sampling", args, profile)
        structured_outputs = make_choice_structured_outputs(profile)
        sampling_params = SamplingParams(
            temperature=s_temp,
            top_p=s_top_p,
            top_k=s_top_k,
            max_tokens=args.max_tokens,
            structured_outputs=structured_outputs,
            skip_special_tokens=skip_special_tokens,
        )
        vllm_outputs = llm.generate(texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in vllm_outputs]

    all_problems, stats = process_results(
        all_problems,
        generated_texts,
        thinking_stop=thinking_stop if harmony_two_phase else None,
        use_quality_filter=not args.no_quality_filter,
    )
    return all_problems, stats


def run_openai_single_prompt(args, all_problems, all_messages, output_dir):
    """OpenAI互換APIを使用して単一プロンプトで推論を実行"""
    import asyncio
    import os

    profile = get_model_profile(args.model)
    # supports_thinking: false が明示されている場合は優先
    if profile.get("supports_thinking") is False:
        supports_reasoning = False
    else:
        supports_reasoning = profile.get("reasoning_effort") is not None

    # base_url / api_key_env の解決（Gemini, Anthropic 等の OpenAI 互換 API 向け）
    base_url = profile.get("base_url", None)
    api_key_env = profile.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if api_key is None:
        raise EnvironmentError(
            f"環境変数 '{api_key_env}' が設定されていません。"
            f" モデル '{args.model}' の使用には {api_key_env} の設定が必要です。"
        )

    use_structured_outputs = profile.get("structured_outputs", True)

    api_params = {"model": args.model}

    if use_structured_outputs:
        api_params["response_format"] = CHOICE_SCHEMA

    if supports_reasoning:
        if args.reasoning_effort:
            if args.reasoning_effort not in profile["reasoning_effort"]:
                print(f"警告: '{args.reasoning_effort}' はこのモデルでサポートされていません。"
                      f" 対応値: {profile['reasoning_effort']}")
            api_params["reasoning_effort"] = args.reasoning_effort
        else:
            api_params["reasoning_effort"] = "medium"
    else:
        if args.reasoning_effort:
            print("警告: このモデルはreasoning_effortをサポートしていません。無視されます。")

    actual_params = {
        "requested": {k: v for k, v in api_params.items() if k != "response_format"},
        "model_profile": profile,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_concurrency": args.api_concurrency,
        "first_response_model": None,
    }
    
    # 非同期処理を実行
    generated_texts, usage_metadata, first_response = asyncio.run(
        run_openai_async(
            all_messages,
            api_params,
            base_url=base_url,
            api_key=api_key,
            concurrency=args.api_concurrency,
        )
    )
    
    # 最初のレスポンス情報を記録
    if first_response:
        actual_params["first_response_model"] = first_response.model
        actual_params["system_fingerprint"] = getattr(first_response, "system_fingerprint", None)
    
    all_problems, stats = process_results(
        all_problems,
        generated_texts,
        use_quality_filter=not args.no_quality_filter,
    )
    for problem, usage in zip(all_problems, usage_metadata):
        problem.update(usage)
    
    params_file = output_dir / "params.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(actual_params, f, indent=2, ensure_ascii=False)
    
    return all_problems, stats


def _to_plain_dict(value):
    """OpenAI SDKのPydanticモデルや通常dictをプレーンなdictに変換する"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return {}


def extract_usage_metadata(response):
    """OpenAI互換APIレスポンスからTSV保存用のトークン使用量を抽出する"""
    usage = _to_plain_dict(getattr(response, "usage", None))
    completion_details = _to_plain_dict(usage.get("completion_tokens_details"))
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    reasoning_tokens = completion_details.get("reasoning_tokens")

    if reasoning_tokens is None and all(
        isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)
    ):
        inferred_reasoning_tokens = total_tokens - prompt_tokens - completion_tokens
        reasoning_tokens = inferred_reasoning_tokens if inferred_reasoning_tokens > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


async def run_openai_async(all_messages, api_params, base_url=None, api_key=None, concurrency=10):
    """OpenAI互換APIを非同期で並列実行"""
    import asyncio
    import os
    from openai import AsyncOpenAI

    client_kwargs = {}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    elif "OPENAI_API_KEY" in os.environ:
        client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]

    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_single_message(i, messages):
        """単一のメッセージを処理"""
        try:
            async with semaphore:
                request_params = {**api_params, "messages": messages}
                response = await client.chat.completions.create(**request_params)
            generated_text = response.choices[0].message.content
            usage = extract_usage_metadata(response)
            return i, generated_text, usage, response if i == 0 else None
        except Exception as e:
            print(f"APIエラー (index {i}): {e}")
            return i, None, {}, None
    
    provider = base_url or "OpenAI"
    print(f"API推論中（並列処理, provider={provider}, concurrency={concurrency}）: {len(all_messages)}件")
    try:
        tasks = [process_single_message(i, messages) for i, messages in enumerate(all_messages)]
        results = await asyncio.gather(*tasks)
        
        # 結果を元の順序でソート
        results_sorted = sorted(results, key=lambda x: x[0])
        generated_texts = [text for _, text, _, _ in results_sorted]
        usage_metadata = [usage for _, _, usage, _ in results_sorted]
        first_response = results_sorted[0][3] if results_sorted else None
        
        return generated_texts, usage_metadata, first_response
    finally:
        await client.close()


def run_single_prompt_experiment(args, prompt_file, prompt_name, all_problems_original, base_output_dir, backend):
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
    validate_multimodal_support(args, all_messages)
    output_dir = base_output_dir / prompt_name
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
    """複数のプロンプトの結果を集計（品質フィルタ後の正答率を使用）"""
    print(f"\n{'='*60}")
    print("全プロンプトの結果集計")
    print(f"{'='*60}\n")
    
    results_df = []
    for prompt_name, stats in zip(prompt_names, all_stats):
        filtered_scores = stats.get('filtered_scores', stats['scores'])
        filtered_total = stats.get('filtered_total', stats['total'])
        filtered_accuracy = stats.get('filtered_accuracy', stats['accuracy'])
        filtered_error_count = stats.get('filtered_error_count', stats['error_count'])
        results_df.append({
            'プロンプト': prompt_name,
            '正答率': f"{filtered_accuracy*100:.1f}%",
            '正解数': sum(filtered_scores),
            '総問題数': filtered_total,
            'エラー数': filtered_error_count,
        })
    
    df = pd.DataFrame(results_df)
    print(df.to_string(index=False))
    
    accuracies = [s.get('filtered_accuracy', s['accuracy']) for s in all_stats]
    avg_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = pd.Series(accuracies).std(ddof=0)
    print(f"\n平均正答率: {avg_accuracy*100:.1f}%")
    print(f"標準偏差: {std_accuracy*100:.1f}%")
    
    summary_file = base_output_dir / "aggregated_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("複数プロンプトの実験結果集計\n")
        f.write("="*60 + "\n\n")
        f.write(df.to_string(index=False) + "\n\n")
        f.write(f"平均正答率: {avg_accuracy*100:.1f}%\n")
        f.write(f"標準偏差: {std_accuracy*100:.1f}%\n")
    print(f"\n集計結果を {summary_file} に保存しました")
    
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
    parser.add_argument('--dataset', type=str, default=None,
                        help='データセットのパス（ローカルファイルのパスまたはHuggingFace Dataset名）。未指定時はlanguageに応じた既定値を使用')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='HuggingFace Datasetのsplit（デフォルト: train）')
    parser.add_argument('--output_dir', type=str, default='output', help='出力ディレクトリのベースパス')
    parser.add_argument('--output_num_label', type=str, default=None,
                        help='出力ディレクトリの件数ラベル（例: 入力100行だけを使う場合に 100 を指定）')
    parser.add_argument('--language', type=str, default='ja', choices=sorted(DEFAULT_DATASETS),
                        help='評価対象言語（デフォルト: ja）')
    parser.add_argument('--swap_statements', action='store_true',
                        help='各問題の直後に文A/文Bを入れ替えた同一タスクのミラー問題を追加し、評価件数を約2倍にする（正解は1↔2で整合）')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=['low', 'medium', 'high', 'minimal', 'none'],
                        help='推論の深さ')
    parser.add_argument('--api_concurrency', type=int, default=10,
                        help='OpenAI互換APIの同時リクエスト数（デフォルト: 10）')
    parser.add_argument('--no_quality_filter', action='store_true',
                        help='人手アノテーション品質フィルタを使わず、全問題で集計する')
    
    # プロンプト関連
    parser.add_argument('--prompt_files', type=str, nargs='+', default=None,
                        help='使用するプロンプトファイルのパス（複数指定可）。指定しない場合はeval_prompt/<language>フォルダ内の全ファイルを使用')
    
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
    vllm_group.add_argument('--thinking_max_tokens', type=int, default=8192,
                           help='思考モードでの思考部分の最大出力トークン数（デフォルト: 8192）')
    vllm_group.add_argument('--answer_max_tokens', type=int, default=10,
                           help='思考モードでの回答部分の最大出力トークン数（デフォルト: 10）')
    vllm_group.add_argument('--trust_remote_code', action='store_true',
                           help='リモートコードの実行を許可（カスタムモデルコードを使用するモデルで必要）')
    vllm_group.add_argument(
        '--generation_prefill',
        type=str,
        default=None,
        help='チャットテンプレート適用後の入力末尾に付加し、その直後から生成する文字列（vLLMのみ）',
    )

    # サンプリングパラメータ上書き（vLLM）
    # 未指定時は model_profiles.yaml の値が使われる
    sampling_group = parser.add_argument_group('サンプリングパラメータ上書き（vLLM、未指定時はmodel_profiles.yaml準拠）')
    sampling_group.add_argument('--temperature', type=float, default=None,
                                help='通常モード・回答フェーズの temperature')
    sampling_group.add_argument('--top_p', type=float, default=None,
                                help='通常モード・回答フェーズの top_p')
    sampling_group.add_argument('--top_k', type=int, default=None,
                                help='通常モード・回答フェーズの top_k')
    sampling_group.add_argument('--thinking_temperature', type=float, default=None,
                                help='thinkingフェーズの temperature')
    sampling_group.add_argument('--thinking_top_p', type=float, default=None,
                                help='thinkingフェーズの top_p')
    sampling_group.add_argument('--thinking_top_k', type=int, default=None,
                                help='thinkingフェーズの top_k')
    sampling_group.add_argument('--answer_temperature', type=float, default=None,
                                help='thinking有効時の回答フェーズの temperature')
    sampling_group.add_argument('--answer_top_p', type=float, default=None,
                                help='thinking有効時の回答フェーズの top_p')
    sampling_group.add_argument('--answer_top_k', type=int, default=None,
                                help='thinking有効時の回答フェーズの top_k')
    sampling_group.add_argument('--thinking_stop', type=str, default=None,
                                help='thinkタグの終了文字列（デフォルト: model_profiles.yaml の thinking_stop）')

    # プロファイルファイルの上書き
    parser.add_argument('--model_profile_file', type=str, default=None,
                        help='モデルプロファイルYAMLファイルのパス（デフォルト: src/evaluation/model_profiles.yaml）')

    args = parser.parse_args()

    if args.api_concurrency < 1:
        parser.error('--api_concurrency は1以上を指定してください')

    if args.dataset is None:
        args.dataset = DEFAULT_DATASETS[args.language]

    # カスタムプロファイルファイルが指定された場合は再読み込み
    if args.model_profile_file:
        global _PROFILES, _VLLM_DEFAULTS, _MODEL_PROFILES
        _PROFILES = _load_profiles(Path(args.model_profile_file))
        _VLLM_DEFAULTS = _PROFILES.get("vllm_defaults", {})
        _MODEL_PROFILES = _PROFILES.get("models", {})
    
    print(f"引数: {args}\n")
    
    project_root = Path(__file__).parent.parent.parent

    # プロンプトファイルのリストを取得
    if args.prompt_files is None:
        # eval_prompt/<language>フォルダ内の全ファイルを使用
        prompt_dir = project_root / "eval_prompt" / args.language
        if not prompt_dir.exists():
            print(f"エラー: {prompt_dir} が見つかりません", file=sys.stderr)
            sys.exit(1)
        
        prompt_files = sorted(prompt_dir.glob("*.txt"))
        if not prompt_files:
            print(f"エラー: {prompt_dir} にプロンプトファイルが見つかりません", file=sys.stderr)
            sys.exit(1)
    else:
        prompt_files = []
        for f in args.prompt_files:
            prompt_file = Path(f)
            if not prompt_file.is_absolute():
                prompt_file = project_root / prompt_file
            prompt_files.append(prompt_file)
    
    print(f"使用するプロンプトファイル数: {len(prompt_files)}")
    for pf in prompt_files:
        print(f"  - {pf.name}")
    print()
    
    # 問題を読み込む（全プロンプトで共通）
    all_problems_original = load_problems(
        args.dataset,
        args.num,
        args.dataset_split,
        args.language,
        swap_statements=args.swap_statements,
    )

    # 出力ディレクトリを作成
    suffix = ""
    backend = detect_backend(args.model)
    profile = get_model_profile(args.model)
    effective_thinking = None
    generation_prefill_source = None

    if backend == 'vllm':
        vllm_supports_reasoning = (
            profile.get("supports_thinking") is not False
            and profile.get("reasoning_effort") is not None
        )
        # thinking が有効かつ reasoning_effort 対応モデルなら medium をデフォルトに設定
        if args.enable_thinking and vllm_supports_reasoning and not args.reasoning_effort:
            args.reasoning_effort = "medium"
            print("ℹ️  reasoning_effort が未指定のため、デフォルト値 'medium' を使用します。")

        effective_thinking = args.enable_thinking or (args.reasoning_effort is not None)
        cli_generation_prefill = args.generation_prefill
        args.generation_prefill = resolve_generation_prefill(
            cli_generation_prefill,
            profile,
            effective_thinking,
        )
        if cli_generation_prefill is not None:
            generation_prefill_source = "cli"
        elif args.generation_prefill is not None:
            generation_prefill_source = "model_profile"
            print("ℹ️  モデルプロファイルの no_thinking_generation_prefill を使用します。")
        if effective_thinking:
            suffix = "_thinking"
        else:
            suffix = "_no_thinking"
        if args.reasoning_effort:
            suffix += f"_reasoning_{args.reasoning_effort}"
        if generation_prefill_source == "cli":
            suffix += "_generation_prefill"
    else:  # openai
        # supports_thinking: false が明示されている場合は優先
        if profile.get("supports_thinking") is False:
            supports_reasoning = False
        else:
            supports_reasoning = profile.get("reasoning_effort") is not None
        if supports_reasoning and args.reasoning_effort:
            suffix = f"_reasoning_{args.reasoning_effort}"
        elif supports_reasoning:
            suffix = "_reasoning_medium"
    
    safe_model_name = args.model.replace('/', '_').replace(':', '_')
    num_str = args.output_num_label or (str(args.num) if args.num is not None else "all")
    base_output_dir = Path(args.output_dir) / args.language / num_str / f"{safe_model_name}{suffix}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    with open(base_output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "backend": backend,
                "effective_thinking": effective_thinking,
                "generation_prefill": args.generation_prefill,
                "generation_prefill_source": generation_prefill_source,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    
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
            all_problems_original,
            base_output_dir,
            backend
        )
        all_stats.append(stats)
    
    # 結果を集計

    aggregate_results(all_stats, prompt_names, base_output_dir)


if __name__ == "__main__":
    main()
