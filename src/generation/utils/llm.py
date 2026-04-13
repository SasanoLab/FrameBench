# -*- coding: utf-8 -*-
"""
ベンチマーク構築時に使用するLLM APIのラッパー
"""

from openai._client import OpenAI
import os
from typing import Dict, List, Union, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml

# モデル設定（JSON/YAMLで管理可能）
with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), encoding='utf-8') as f:
    _yaml = yaml.safe_load(f)
MODEL_CONFIGS = _yaml['models']

CLIENT_CLASS = {
    # 追加する場合は実装が必要
    "openai": OpenAI,
}

def prompt_to_string(prompt: Union[str, List[Any]]) -> str:
    """メッセージ配列や文字列をResponses APIのinput用テキストに正規化"""
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt:
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        content = getattr(m, "content", "")
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _single_request(args):
    """単一のリクエストを実行（並列処理用）"""
    messages, client, params = args
    try:
        resp = client.responses.create(
            input=messages,
            **params,
        )
        try:
            output_text = resp.output_text
        except Exception as e:
            output_text = str(resp)
        return output_text
    except Exception as e:
        return f"Error: {str(e)}"


def _save_input_logs(model_name: str, prompts: List[List[Dict[str, str]]], output_dir: str):
    """inputプロンプトをログファイルに保存"""
    from datetime import datetime
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    log_path = os.path.join(output_dir, f"input_logs_{safe_model_name}_{timestamp}.md")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            prompt_text = prompt_to_string(prompt)
            f.write(f"## プロンプト #{i+1}\n\n")
            f.write(prompt_text)
            f.write("\n-----\n\n")
    print(f"📝 Inputログを保存しました: {log_path} ({len(prompts)}件)")


def generate_batch(
    model_name: str,
    prompts: List[List[Dict[str, str]]],
    output_dir: Optional[str] = None,
    **model_kwargs
) -> List[str]:
    """
    バッチ処理（すべてのOpenAIモデルをResponses APIで処理）
    
    Args:
        model_name: モデル名
        prompts: プロンプトのリスト（各要素はjson形式: [{"role": "system", "content": "..."}, ...]）
        output_dir: 出力ディレクトリ（inputログ保存用）
        **model_kwargs: モデルパラメータ
    
    Returns:
        List[str]: 生成されたテキストのリスト
    """
    # モデル設定の確認
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # inputログを保存
    if output_dir:
        _save_input_logs(model_name, prompts, output_dir)
    
    params = config.get("default_params", {}).copy()
    params.update({k: v for k, v in model_kwargs.items() if k in params})
    if config.get("reasoning") != "none":
        if params.get('temperature', 1.0) != 1.0:
            print(f"Warning: temperature is not 1.0 for reasoning model {model_name}")
            params['temperature'] = 1.0
    
    print(f"🔄 クライアント初期化中... ({model_name})")
    try:
        client = CLIENT_CLASS[config.get("provider")]()
    except KeyError as e:
        print(f"Error: {e}")
        raise ValueError(f"Unsupported provider: {config.get('provider')}. Available: {list(CLIENT_CLASS.keys())}")

    args_list = [(messages, client, params) for messages in prompts]
    print(f"🚀 並列実行開始: {len(prompts)}リクエスト")
    with tqdm(total=len(prompts), desc=f"処理中 ({model_name})", unit="req",leave=True) as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(_single_request, args): i for i, args in enumerate(args_list)}
            results: List[Optional[str]] = [None] * len(prompts)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    pbar.update(1)
                except Exception as e:
                    results[index] = f"Error: {str(e)}"
                    pbar.update(1)
    return [result if result is not None else "" for result in results]

def list_available_models() -> List[str]:
    """利用可能なモデル一覧を取得"""
    return list(MODEL_CONFIGS.keys())