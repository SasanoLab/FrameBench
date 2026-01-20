"""
tools/data/ja/gpt-4.1-mini/step3/eval_round2/evaluations_merged.tsvに存在する問題は、こちらのデータを優先し、それ以外の問題のデータはtools/data/ja/gpt-4.1-mini/step3/eval_round1/evaluations_merged_edited.tsvからもってくる
その後、bad exampleがあれば除外する
tools/data/ja/gpt-4.1-mini/step2/qa.jsonlと同じ形式のjsonlで保存する
Usage:
uv run python postprocess/merge_eval_rounds.py --base_dir data/ja/gpt-4.1-mini/step3
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import argparse


def safe_str(value) -> str:
    """
    pandasの値を安全に文字列に変換する
    
    Args:
        value: pandasの値（NaNの可能性あり）
        
    Returns:
        文字列（NaNの場合は空文字列）
    """
    if value is None:
        return ''
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, bool) and is_na:
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)


def tsv_row_to_qa_jsonl(row: pd.Series) -> Optional[Dict]:
    """
    TSVの1行をqa.jsonl形式の辞書に変換する
    
    Args:
        row: TSVの1行（pandas Series）
        
    Returns:
        qa.jsonl形式の辞書、またはNone（スキップする場合）
    """
    # corrected_questionを確認（存在する場合）
    corrected_question = safe_str(row.get('corrected_question', ''))
    
    # bad exampleを除外
    if corrected_question == 'bad example' or corrected_question == '':
        # corrected_questionが空の場合は、original_questionを使用
        question_text = safe_str(row.get('original_question', ''))
        if not question_text:
            question_text = safe_str(row.get('four-choice-question', ''))
    else:
        question_text = corrected_question
    
    # 必要な情報を取得
    lex_unit_name = safe_str(row.get('lex_unit_name', ''))
    frame_A = safe_str(row.get('frame_A', ''))
    frame_B = safe_str(row.get('frame_B', ''))
    sentence_A = safe_str(row.get('sentence_A', ''))
    sentence_B = safe_str(row.get('sentence_B', ''))
    answer = safe_str(row.get('answer', ''))
    
    # 必須フィールドのチェック
    if not lex_unit_name or not frame_A or not frame_B or not answer:
        return None
    
    # question_textから質問文と文を抽出
    # 形式1: 「質問文」\nSentence 1: ...\nSentence 2: ... (改行あり)
    # 形式2: 「質問文」 Sentence 1: ... Sentence 2: ... (改行なし)
    lines = question_text.strip().split('\n')
    question = lines[0].strip() if lines else ''
    
    # Sentence 1とSentence 2を抽出
    sentences = []
    
    # まず改行で分割された行から抽出を試みる
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('Sentence 1:') or line.startswith('Sentence 2:'):
            colon_pos = line.find(':')
            if colon_pos != -1:
                sentence = line[colon_pos + 1:].strip()
                sentences.append(sentence)
    
    # 改行がない場合（形式2）、最初の行から抽出を試みる
    if len(sentences) < 2 and question:
        # "Sentence 1: ... Sentence 2: ..." のパターンを検索
        pattern = r'Sentence 1:\s*([^S]+?)\s*Sentence 2:\s*(.+)'
        match = re.search(pattern, question_text)
        if match:
            sentences = [match.group(1).strip(), match.group(2).strip()]
            # 質問文から文の部分を除去
            question = re.sub(r'\s*Sentence 1:.*$', '', question).strip()
    
    # 文が2つある場合はそれを使用、ない場合は元のsentence_Aとsentence_Bを使用
    if len(sentences) >= 2:
        sentence_A = sentences[0]
        sentence_B = sentences[1]
    elif not sentence_A or not sentence_B:
        # 文が取得できず、元のデータにもない場合はスキップ
        return None
    
    # 質問文に文を含める（必ず改行を含む形式にする）
    if question:
        full_question = f"{question}\nSentence 1: {sentence_A}\nSentence 2: {sentence_B}"
    else:
        return None
    
    # qa.jsonl形式に変換
    result = {
        'lex_unit_name': lex_unit_name,
        'sentence_pair': {
            frame_A: sentence_A,
            frame_B: sentence_B
        },
        'result': {
            'question': full_question,
            'choices': ['Sentence 1', 'Sentence 2', 'Both sentences', 'Neither sentence'],
            'answer': answer
        },
        'thinking_process': ''  # TSVにはないので空文字
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    # ファイルパスの設定
    round2_file = base_dir / 'eval_round2' / 'evaluations_merged.tsv'
    round1_file = base_dir / 'eval_round1' / 'evaluations_merged_edited.tsv'
    output_file = base_dir / 'qa_annotated.jsonl'
    
    # ファイルの存在確認
    if not round2_file.exists():
        print(f"Error: {round2_file} が見つかりません")
        return
    
    if not round1_file.exists():
        print(f"Error: {round1_file} が見つかりません")
        return
    
    # TSVファイルを読み込む
    print(f"Reading {round2_file}...")
    df_round2 = pd.read_csv(round2_file, sep='\t')
    
    print(f"Reading {round1_file}...")
    df_round1 = pd.read_csv(round1_file, sep='\t')
    
    # eval_round2に存在する問題のoriginal_qa_idを取得
    round2_qa_ids = set(df_round2['original_qa_id'].dropna().astype(str))
    print(f"Found {len(round2_qa_ids)} unique problems in round2")
    
    # eval_round2のデータを処理
    results = []
    seen_qa_ids = set()
    
    print("Processing round2 data...")
    for _, row in df_round2.iterrows():
        qa_id = safe_str(row.get('original_qa_id', ''))
        if not qa_id:
            continue
        
        # 重複チェック
        if qa_id in seen_qa_ids:
            continue
        
        result = tsv_row_to_qa_jsonl(row)
        if result is not None:
            results.append(result)
            seen_qa_ids.add(qa_id)
    
    print(f"Added {len(results)} problems from round2")
    
    # eval_round1のデータを処理（round2に存在しない問題のみ）
    print("Processing round1 data...")
    round1_count = 0
    for _, row in df_round1.iterrows():
        qa_id = safe_str(row.get('original_qa_id', ''))
        if not qa_id:
            continue
        
        # round2に既に存在する場合はスキップ
        if qa_id in round2_qa_ids:
            continue
        
        # 重複チェック
        if qa_id in seen_qa_ids:
            continue
        
        result = tsv_row_to_qa_jsonl(row)
        if result is not None:
            results.append(result)
            seen_qa_ids.add(qa_id)
            round1_count += 1
    
    print(f"Added {round1_count} problems from round1")
    print(f"Total: {len(results)} problems")
    
    # JSONL形式で保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
