import os
import logging
import json
import pandas as pd
from typing import Any, Dict, List, Union
import re

SUPPORTED_LANGUAGES_MAP = {
    "ja": "Japanese",
    # "en": "English",
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

# def process_reasoning_response(result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     reasoning情報を含むレスポンスを処理する関数
#     o4-miniなどのリーズニングモデル用
    
#     Args:
#         result: レスポンス（文字列またはすでに構造化されたデータ）
        
#     Returns:
#         reasoning, content, JSONデータを含む構造化されたレスポンス
#     """
#     if isinstance(result, dict):
#         # すでに構造化されている場合（OpenAILLMから）
#         response_data = result
#         content = result.get("content", "")
#         reasoning = result.get("reasoning", "")
#     else:
#         # 文字列の場合
#         response_data = {
#             "content": result,
#             "reasoning": None,
#             "model": "unknown"
#         }
#         content = result
#         reasoning = ""
    
#     # contentからJSONを抽出
#     json_data = process_json_response(content)
    
#     return {
#         "reasoning": reasoning,
#         "content": content,
#         "parsed_json": json_data,
#         "model": response_data.get("model", "unknown"),
#         "finish_reason": response_data.get("finish_reason", "unknown")
#     }

# def save_results_with_processing(
#     data: pd.DataFrame, 
#     results: List[str], 
#     output_dir: str, 
#     filename_prefix: str,
#     result_processor=None,
#     additional_data: Dict[str, Any] = None
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     結果を中間ファイルと最終ファイルに保存する共通処理
    
#     Args:
#         data: 元のデータフレーム
#         results: LLMからの結果リスト
#         output_dir: 出力ディレクトリ
#         filename_prefix: ファイル名のプレフィックス
#         result_processor: 結果を処理する関数（オプション）
#         additional_data: 追加で保存するデータ（オプション）
        
#     Returns:
#         (中間結果DataFrame, 最終結果DataFrame)
#     """
#     # 中間結果の作成
#     intermediate_output = []
#     for df_dict, result in zip(data.to_dict(orient="records"), results):
#         output_item = {**df_dict, "raw_result": result}
#         if additional_data:
#             output_item.update(additional_data)
#         intermediate_output.append(output_item)
    
#     intermediate_df = pd.DataFrame(intermediate_output)
    
#     # 中間結果の保存
#     intermediate_path = f"{output_dir}/{filename_prefix}_intermediate.jsonl"
#     intermediate_df.to_json(intermediate_path, orient="records", lines=True, force_ascii=False)
#     print(f"中間結果を {intermediate_path} に保存しました")
    
#     # 結果の処理と最終ファイルの保存
#     if result_processor:
#         final_df = intermediate_df.copy()
#         final_df['processed_result'] = final_df['raw_result'].apply(result_processor)
#         final_df = final_df[
#             (final_df['processed_result'].notna()) & 
#             (final_df['processed_result'].apply(len) > 0)
#         ]
#         final_df = final_df.drop(columns=['raw_result'])
        
#         final_path = f"{output_dir}/{filename_prefix}.jsonl"
#         final_df.to_json(final_path, orient="records", lines=True, force_ascii=False)
#         print(f"最終結果を {final_path} に保存しました")
#         return intermediate_df, final_df
#     else:
#         return intermediate_df, intermediate_df


# def execute_llm_pipeline(
#     prompts: List[str], 
#     model: str, 
#     output_dir: str, 
#     filename_prefix: str,
#     data: pd.DataFrame = None,
#     result_processor=None,
#     sample_n: int = 2,
#     include_reasoning: bool = None
# ) -> tuple[List[str], Dict[str, Any]]:
#     """
#     LLM実行パイプラインの共通処理
    
#     Args:
#         prompts: プロンプトのリスト
#         model: 使用するモデル名
#         output_dir: 出力ディレクトリ
#         filename_prefix: ファイル名プレフィックス
#         data: 元データ（結果保存用）
#         result_processor: 結果処理関数
#         sample_n: コスト予測用サンプル数
#         include_reasoning: reasoning情報を含めるかどうか（o4-mini用）
        
#     Returns:
#         (結果リスト, コスト情報)
        
#     Example:
#         # o4-miniでreasoning情報を含める場合
#         results, cost_info = execute_llm_pipeline(
#             prompts=["質問を作成してください"],
#             model="o4-mini", 
#             output_dir="output/",
#             filename_prefix="qa",
#             include_reasoning=True  # reasoning情報を含める
#         )
        
#         # 結果にはreasoningとcontentが含まれる
#         if isinstance(results[0], dict):
#             print(f"推論過程: {results[0]['reasoning']}")
#             print(f"回答: {results[0]['content']}")
#     """
#     # 出力ディレクトリの作成
#     os.makedirs(output_dir, exist_ok=True)
    
#     # LLMの初期化
#     llm = initialize_llm(model)
    
#     # コスト予測
#     input_cost = llm.prediction_cost(prompts, sample_n=sample_n)
#     print(f"コスト予測: ${input_cost:.6f} USD")
    
#     # Geminiモデルの場合は並列処理を無効化
#     parallel = False if "gemini" in model.lower() else True
    
#     # LLM実行（o4-miniの場合はreasoning対応）
#     if hasattr(llm, 'generate') and 'include_reasoning' in llm.generate.__code__.co_varnames:
#         results, cost_info = llm.generate(prompts, track_cost=True, parallel=parallel, include_reasoning=include_reasoning)
#     else:
#         results, cost_info = llm.generate(prompts, track_cost=True, parallel=parallel)
    
#     print(f"実際のコスト: ${cost_info['total_cost']:.6f} USD")
#     print(f"処理時間: {cost_info['duration']:.1f}秒")
    
#     # 結果の保存
#     if data is not None:
#         intermediate_df, final_df = save_results_with_processing(
#             data, results, output_dir, filename_prefix, result_processor
#         )
#         print(f"中間結果の数: {len(intermediate_df)}")
#         print(f"最終結果の数: {len(final_df)}")
    
#     return results, cost_info


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
