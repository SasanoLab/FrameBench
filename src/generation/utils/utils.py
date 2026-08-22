import os
import logging
import json
import pandas as pd
from typing import Any, Dict, List, Union
import re

SUPPORTED_LANGUAGES_MAP = {
    "ja": "Japanese",
    "en": "English",
    # "zh": "Chinese",
}

def process_json_response(result: Union[str, Dict[str, Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    LLMからのJSON形式のレスポンスを処理する共通関数
    
    Args:
        result: LLMからの生のレスポンス文字列
        
    Returns:
        パースされたJSONオブジェクト、またはパースに失敗した場合はNone
    """
    if isinstance(result, dict):
        return result
    try:
        # ```jsonで囲まれている場合の処理
        if result.strip().startswith("```json"):
            result = result.strip().split("```json")[1]
        if result.strip().endswith("```"):
            result = result.strip().rsplit("```", 1)[0]
        
        # 文末の全角閉じ引用符を半角に変換（JSONの値の終わりで使われている場合）
        result = re.sub(r'」\n', '"\n', result)
        
        # 末尾カンマを除去する処理
        # オブジェクトや配列内の末尾カンマを除去
        result = re.sub(r',(\s*[}\]])', r'\1', result.strip())
            
        parsed_result = json.loads(result.strip())
        if "results" in parsed_result:
            parsed_result = parsed_result["results"]
        return parsed_result
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"JSON処理エラー: {e}")
        print(f"result: {result}")
        return None

def setup_output_directory(model: str, language: str, base_dir: str = "data") -> str:
    """
    モデル名に基づいて出力ディレクトリを設定する
    
    Args:
        model: モデル名
        base_dir: ベースディレクトリ
        
    Returns:
        出力ディレクトリパス
    """
    output_dir = f"{base_dir}/{language}/{model.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir 

def load_jsonl(data_path: str, description: str = "データ") -> pd.DataFrame:
    """JSONL形式のデータを読み込む共通関数"""
    logging.info(f"{description}を読み込み中...")
    return pd.read_json(data_path, lines=True, orient="records")
