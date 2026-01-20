"""
アノテーションした結果を整理する
1. bad exampleは弾く
2. それ以外が必ず、質問文\nSentence 1: \nSentence 2: の形式になっていることを確認
3. Sentenceの最後が. とか、。以外の終端文字で終わってる場合はreplaceする。。に変換する
1で弾いた件数、3で処理した件数、その他の修正をアノテーションによって行っている数を統計する
question_formatにあわせて、元の質問文を確認し、質問文が更新されている場合はすべてのフォーマットの質問文を更新する
結果は/home/yano/work/ghq/github.com/yano0/framebench/tools/data/ja/test_step2/formatted_qa.jsonlと同じ形式で保存する
"""

import json
import sys
from typing import Dict
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    return parser.parse_args()


def normalize_sentence_endings(text: str) -> str:
    """
    Sentence 1とSentence 2の末尾を正規化する
    末尾が.や。以外の終端文字で終わっている場合は。に変換する
    
    Returns:
        修正後のテキスト
    """
    # Sentence 1とSentence 2の行を探して、末尾の終端文字を。に統一
    lines = text.split('\n')
    result_lines = []
    
    for line in lines:
        if line.startswith('Sentence 1:') or line.startswith('Sentence 2:'):
            # 末尾が.で終わっている場合は。に変換
            if line.rstrip().endswith('.') and not line.rstrip().endswith('。'):
                line = line.rstrip()[:-1] + '。'
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def update_question_formats(updated_question: str) -> Dict[str, str]:
    """
    更新された質問文から、すべてのフォーマットの質問文を生成する
    
    Args:
        updated_question: 更新された質問文（corrected_questionの最初の行）
    
    Returns:
        {'single-question': ..., 'four-choice-question': ..., 'two-choice-question': ...}
    """
    question = updated_question.strip()
    
    # 質問文のパターンに基づいて他のフォーマットを生成
    if question.endswith("なのはどちらの文ですか？"):
        base = question.replace("なのはどちらの文ですか？", "")
        single_question = '「' + base + '」と言えますか？'
        two_choice_question = '「' + base + '」のはどちらの文ですか？'
        four_choice_question = '「' + base + '」と言える文をすべて選んでください。'
    elif question.endswith("のはどちらの文ですか？"):
        base = question.replace("のはどちらの文ですか？", "")
        single_question = '「' + base + '」と言えますか？'
        two_choice_question = '「' + base + '」のはどちらの文ですか？'
        four_choice_question = '「' + base + '」と言える文をすべて選んでください。'
    elif question.endswith("と言える文をすべて選んでください。"):
        base = question.replace("と言える文をすべて選んでください。", "")
        single_question = base + 'と言えますか？'
        two_choice_question = base + 'のはどちらの文ですか？'
        four_choice_question = question
    elif question.endswith("と言えますか？"):
        base = question.replace("と言えますか？", "")
        single_question = question
        two_choice_question = base + 'のはどちらの文ですか？'
        four_choice_question = base + 'と言える文をすべて選んでください。'
    else:
        # パターンに一致しない場合は、元の質問文をそのまま使用
        single_question = question
        two_choice_question = question
        four_choice_question = question
    
    return {
        'single-question': single_question,
        'four-choice-question': four_choice_question,
        'two-choice-question': two_choice_question
    }


def validate_format(text: str) -> bool:
    """
    質問文\nSentence 1: \nSentence 2: の形式になっているか確認
    """
    lines = text.split('\n')
    if len(lines) < 3:
        return False
    
    # Sentence 1: とSentence 2: が含まれているか確認
    has_sentence1 = any(line.startswith('Sentence 1:') for line in lines)
    has_sentence2 = any(line.startswith('Sentence 2:') for line in lines)
    
    return has_sentence1 and has_sentence2


def main(args):
    # 入力JSONを読み込み
    input_data = json.load(open(args.input_file, 'r', encoding='utf-8'))
    output_file = Path(args.input_file).parent.parent  / 'text_corrected_qa.jsonl'
    # 統計情報
    bad_example_count = 0
    ending_modified_count = 0
    annotation_correction_count = 0
    question_updated_count = 0
    output_data = []
    
    for key, entry in input_data.items():
        corrected_question = entry.get('corrected_question', '')
        original_question = entry.get('original_question', '')
        question_format = entry.get('question_format', 'four_choices')
        
        # 1. bad exampleは弾く
        if corrected_question == 'bad example':
            bad_example_count += 1
            continue
        
        # 2. フォーマットの確認
        if not validate_format(corrected_question):
            print(f"警告: {key} はフォーマットが不正です: {corrected_question[:100]}", file=sys.stderr)
            continue
        
        # corrected_questionから質問文とSentenceを抽出
        corrected_lines = corrected_question.split('\n')
        updated_question_text = corrected_lines[0].strip()  # 最初の行が質問文
        
        # 3. Sentenceの末尾を正規化
        normalized_corrected = normalize_sentence_endings(corrected_question)
        normalized_original = normalize_sentence_endings(original_question)
        
        # 末尾の終端文字が修正されたか確認
        if normalized_original != original_question:
            ending_modified_count += 1
        
        # アノテーションによる修正をカウント（末尾の修正以外）
        if normalized_corrected != normalized_original:
            annotation_correction_count += 1
        
        # question_formatに合わせて、元の質問文を確認し、更新されている場合はすべてのフォーマットの質問文を更新
        original_single = entry.get('single-question', '')
        original_four_choice = entry.get('four-choice-question', '')
        original_two_choice = entry.get('two-choice-question', '')
        
        # 現在のquestion_formatに対応する元の質問文を取得
        original_question_for_format = ''
        if question_format == 'four_choices':
            original_question_for_format = original_four_choice
        elif question_format == 'two_choices':
            original_question_for_format = original_two_choice
        elif question_format == 'single':
            original_question_for_format = original_single
        
        # 質問文が更新されているか確認
        if updated_question_text != original_question_for_format:
            question_updated_count += 1
            # すべてのフォーマットの質問文を更新
            updated_formats = update_question_formats(updated_question_text)
            single_question = updated_formats['single-question']
            four_choice_question = updated_formats['four-choice-question']
            two_choice_question = updated_formats['two-choice-question']
        else:
            # 更新されていない場合は元の質問文を使用
            single_question = original_single
            four_choice_question = original_four_choice
            two_choice_question = original_two_choice
        
        # Sentence 1とSentence 2のテキストを抽出（"Sentence 1: "や"Sentence 2: "のプレフィックスを除去）
        sentence_1_line = normalized_corrected.split('\n')[1]
        sentence_2_line = normalized_corrected.split('\n')[2]
        sentence_1 = sentence_1_line.replace('Sentence 1: ', '').strip()
        sentence_2 = sentence_2_line.replace('Sentence 2: ', '').strip()
        
        # formatted_qa.jsonlと同じ形式で出力
        output_data.append({
            'original_qa_id': entry.get('original_qa_id', ''),
            'lex_unit_name': entry.get('lex_unit_name', ''),
            'single-question': single_question,
            'four-choice-question': four_choice_question,
            'two-choice-question': two_choice_question,
            'sentence_A': sentence_1,
            'sentence_B': sentence_2,
            'frame_A': entry.get('frame_A', ''),
            'frame_B': entry.get('frame_B', ''),
        })
    
    # 統計情報を標準エラー出力に出力
    print("=== 統計情報 ===", file=sys.stderr)
    print(f"bad exampleで弾いた件数: {bad_example_count}", file=sys.stderr)
    print(f"末尾の終端文字を修正した件数: {ending_modified_count}", file=sys.stderr)
    print(f"アノテーションによる修正件数: {annotation_correction_count}", file=sys.stderr)
    print(f"質問文が更新された件数: {question_updated_count}", file=sys.stderr)
    print(f"有効なエントリ数: {len(output_data)}", file=sys.stderr)
    print(f"総エントリ数: {len(input_data)}", file=sys.stderr)
    
    # 結果をformatted_qa.jsonlと同じ形式で出力
    pd.DataFrame(output_data).to_json(output_file, orient='records', force_ascii=False, lines=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)