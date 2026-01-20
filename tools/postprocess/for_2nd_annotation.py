"""
/home/yano/work/ghq/github.com/yano0/framebench/tools/data/ja/gpt-4.1-mini/step3/annotated/corrected.tsv
などを標準入力から読み込んで、corrected_questionが空でないかつ、bad exampleでないものを抽出して、
アノテーション用に/home/yano/work/ghq/github.com/yano0/framebench/tools/data/ja/gpt-4.1-mini/step3/qa.jsonlとおなじフォーマットにする
SentenceAとSentenceBだけが編集されている場合はSentenceA_primeとSentenceB_primeは空文字にする（アノテーションしたくないので）
同様にSentenceA_primeとSentenceB_primeが編集されている場合はSentenceAとSentenceBは空文字にする
質問文が更新されている場合も含め、その他の場合は全文そのままにしてアノテーションできるようにする
Usage:
uv run python postprocess/for_2nd_annotation.py --input_file data/ja/gpt-4.1-mini/step3/eval_round1/evaluations_merged_edited.tsv
"""

import argparse
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path


def parse_question_text(question_text: str) -> Tuple[Optional[str], list]:
    """
    質問文テキストをパースして、質問文と文のリストを返す
    
    Args:
        question_text: 「質問文」\nSentence 1: ...\nSentence 2: ... の形式
        
    Returns:
        (質問文, [文1, 文2, ...]) のタプル
    """
    if not question_text or pd.isna(question_text):
        return None, []
    
    lines = question_text.strip().split('\n')
    if len(lines) == 0:
        return None, []
    
    question = lines[0].strip()
    sentences = []
    
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('Sentence 1:') or line.startswith('Sentence 2:') or \
           line.startswith('Sentence 3:') or line.startswith('Sentence 4:') or \
           line.startswith('Sentence a:') or line.startswith('Sentence b:') or \
           line.startswith('Sentence a prime:') or line.startswith('Sentence b prime:'):
            # プレフィックスを除去
            colon_pos = line.find(':')
            if colon_pos != -1:
                sentence = line[colon_pos + 1:].strip()
                sentences.append(sentence)
    
    return question, sentences


def detect_edited_parts(original_question: str, corrected_question: str, 
                        original_sentence_a: str, original_sentence_b: str,
                        original_sentence_a_prime: str, original_sentence_b_prime: str) -> Dict[str, bool]:
    """
    original_questionとcorrected_questionを比較して、どの部分が編集されたかを判定
    
    Args:
        original_question: 元の質問文テキスト
        corrected_question: 修正後の質問文テキスト
        original_sentence_a: 元のsentence_A
        original_sentence_b: 元のsentence_B
        original_sentence_a_prime: 元のsentence_A_prime
        original_sentence_b_prime: 元のsentence_B_prime
    
    Returns:
        {
            'question_edited': bool,
            'sentence_a_edited': bool,
            'sentence_b_edited': bool,
            'sentence_a_prime_edited': bool,
            'sentence_b_prime_edited': bool
        }
    """
    orig_q, orig_sentences = parse_question_text(original_question)
    corr_q, corr_sentences = parse_question_text(corrected_question)
    
    result = {
        'question_edited': False,
        'sentence_a_edited': False,
        'sentence_b_edited': False,
        'sentence_a_prime_edited': False,
        'sentence_b_prime_edited': False
    }
    
    if orig_q is None or corr_q is None:
        return result
    
    # 質問文が編集されたか
    if orig_q != corr_q:
        result['question_edited'] = True
    
    # original_questionには通常2つの文（Sentence 1, Sentence 2）が含まれる
    # corrected_questionには2つまたは4つの文が含まれる可能性がある
    
    # Sentence A (Sentence 1) が編集されたか
    # original_questionのSentence 1とcorrected_questionのSentence 1を比較
    if len(orig_sentences) > 0 and len(corr_sentences) > 0:
        if orig_sentences[0] != corr_sentences[0]:
            result['sentence_a_edited'] = True
        # 元のデータのsentence_Aとも比較
        if original_sentence_a and corr_sentences[0] != original_sentence_a:
            result['sentence_a_edited'] = True
    elif len(corr_sentences) > 0:
        # original_questionに文がないが、corrected_questionに文がある場合
        if original_sentence_a and corr_sentences[0] != original_sentence_a:
            result['sentence_a_edited'] = True
    
    # Sentence B (Sentence 2) が編集されたか
    if len(orig_sentences) > 1 and len(corr_sentences) > 1:
        if orig_sentences[1] != corr_sentences[1]:
            result['sentence_b_edited'] = True
        # 元のデータのsentence_Bとも比較
        if original_sentence_b and corr_sentences[1] != original_sentence_b:
            result['sentence_b_edited'] = True
    elif len(corr_sentences) > 1:
        # original_questionに文がないが、corrected_questionに文がある場合
        if original_sentence_b and corr_sentences[1] != original_sentence_b:
            result['sentence_b_edited'] = True
    
    # corrected_questionに4つの文が含まれている場合、Sentence A'とB'が編集されたと判定
    if len(corr_sentences) >= 4:
        # Sentence A' (Sentence 3) が編集されたか
        if len(corr_sentences) > 2:
            if original_sentence_a_prime and corr_sentences[2] != original_sentence_a_prime:
                result['sentence_a_prime_edited'] = True
            elif not original_sentence_a_prime:
                # 元のデータにsentence_A_primeがなかった場合は、新しく追加されたと判定
                result['sentence_a_prime_edited'] = True
        
        # Sentence B' (Sentence 4) が編集されたか
        if len(corr_sentences) > 3:
            if original_sentence_b_prime and corr_sentences[3] != original_sentence_b_prime:
                result['sentence_b_prime_edited'] = True
            elif not original_sentence_b_prime:
                # 元のデータにsentence_B_primeがなかった場合は、新しく追加されたと判定
                result['sentence_b_prime_edited'] = True
    
    return result


def update_question_formats(updated_question: str) -> Dict[str, str]:
    """
    更新された質問文から、すべてのフォーマットの質問文を生成する
    
    Args:
        updated_question: 更新された質問文
        
    Returns:
        {
            'single-question': str,
            'two-choice-question': str,
            'four-choice-question': str
        }
    """
    # 質問文のフォーマットを変換
    # 「...と言えますか？」→ single-question
    # 「...のはどちらの文ですか？」→ two-choice-question
    # 「...と言える文をすべて選んでください。」→ four-choice-question
    
    single_q = updated_question
    two_choice_q = updated_question.replace('と言えますか？', 'のはどちらの文ですか？')
    four_choice_q = updated_question.replace('と言えますか？', 'と言える文をすべて選んでください。')
    
    # 既に他のフォーマットの場合は変換
    if 'のはどちらの文ですか？' in updated_question:
        single_q = updated_question.replace('のはどちらの文ですか？', 'と言えますか？')
        four_choice_q = updated_question.replace('のはどちらの文ですか？', 'と言える文をすべて選んでください。')
    elif 'と言える文をすべて選んでください。' in updated_question:
        single_q = updated_question.replace('と言える文をすべて選んでください。', 'と言えますか？')
        two_choice_q = updated_question.replace('と言える文をすべて選んでください。', 'のはどちらの文ですか？')
    
    return {
        'single-question': single_q,
        'two-choice-question': two_choice_q,
        'four-choice-question': four_choice_q
    }


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


def process_row(row: pd.Series) -> Optional[Dict]:
    """
    1行のデータを処理して、qa.jsonl形式の辞書を返す
    
    Returns:
        qa.jsonl形式の辞書、またはNone（スキップする場合）
    """
    corrected_question = safe_str(row.get('corrected_question', ''))
    
    # corrected_questionが空でないかつ、bad exampleでないものを抽出
    if corrected_question == '' or corrected_question == 'bad example':
        return None
    
    # original_questionとcorrected_questionを比較
    original_question = safe_str(row.get('original_question', ''))
    original_sentence_a = safe_str(row.get('sentence_A', ''))
    original_sentence_b = safe_str(row.get('sentence_B', ''))
    original_sentence_a_prime = safe_str(row.get('sentence_A_prime', ''))
    original_sentence_b_prime = safe_str(row.get('sentence_B_prime', ''))
    
    edited_parts = detect_edited_parts(
        original_question, corrected_question,
        original_sentence_a, original_sentence_b,
        original_sentence_a_prime, original_sentence_b_prime
    )
    
    # corrected_questionから質問文と文を抽出
    corr_q, corr_sentences = parse_question_text(corrected_question)
    
    # 元のデータから情報を取得
    original_qa_id = safe_str(row.get('original_qa_id', ''))
    lex_unit_name = safe_str(row.get('lex_unit_name', ''))
    
    sentence_A = original_sentence_a
    sentence_B = original_sentence_b
    sentence_A_prime = original_sentence_a_prime
    sentence_B_prime = original_sentence_b_prime
    
    frame_A = safe_str(row.get('frame_A', ''))
    frame_B = safe_str(row.get('frame_B', ''))
    
    # 元の質問文フォーマット
    original_single = safe_str(row.get('single-question', ''))
    original_two_choice = safe_str(row.get('two-choice-question', ''))
    original_four_choice = safe_str(row.get('four-choice-question', ''))
    
    # 質問文が更新されているか確認
    if edited_parts['question_edited'] and corr_q:
        # すべてのフォーマットの質問文を更新
        updated_formats = update_question_formats(corr_q)
        single_question = updated_formats['single-question']
        two_choice_question = updated_formats['two-choice-question']
        four_choice_question = updated_formats['four-choice-question']
    else:
        # 更新されていない場合は元の質問文を使用
        single_question = original_single
        two_choice_question = original_two_choice
        four_choice_question = original_four_choice
    
    # 質問文が空の場合は元の質問文を使用
    if not single_question:
        single_question = original_single
    if not two_choice_question:
        two_choice_question = original_two_choice
    if not four_choice_question:
        four_choice_question = original_four_choice
    
    # 編集された部分に応じて、文を設定
    # SentenceAとSentenceBだけが編集されている場合
    # (corrected_questionに2つの文しかなく、それらがSentence AとBに対応する場合)
    if len(corr_sentences) == 2 and \
       (edited_parts['sentence_a_edited'] or edited_parts['sentence_b_edited']) and \
       not edited_parts['sentence_a_prime_edited'] and \
       not edited_parts['sentence_b_prime_edited'] and \
       not edited_parts['question_edited']:
        # SentenceA_primeとSentenceB_primeは空文字にする
        sentence_A = corr_sentences[0]
        sentence_B = corr_sentences[1]
        sentence_A_prime = ''
        sentence_B_prime = ''
    
    # SentenceA_primeとSentenceB_primeが編集されている場合
    # (corrected_questionに4つの文があり、Sentence 3と4が編集された場合)
    elif len(corr_sentences) >= 4 and \
         (edited_parts['sentence_a_prime_edited'] or edited_parts['sentence_b_prime_edited']) and \
         not edited_parts['sentence_a_edited'] and \
         not edited_parts['sentence_b_edited'] and \
         not edited_parts['question_edited']:
        # SentenceAとSentenceBは空文字にする
        sentence_A = ''
        sentence_B = ''
        sentence_A_prime = corr_sentences[2]
        sentence_B_prime = corr_sentences[3]
    
    # その他の場合（質問文が更新されている場合も含む）は全文そのまま
    else:
        # corrected_questionから文を取得
        if len(corr_sentences) >= 2:
            sentence_A = corr_sentences[0]
            sentence_B = corr_sentences[1]
        if len(corr_sentences) >= 4:
            sentence_A_prime = corr_sentences[2]
            sentence_B_prime = corr_sentences[3]
        else:
            # corrected_questionに2つの文しかない場合は、元のデータのsentence_A_primeとsentence_B_primeを使用
            sentence_A_prime = original_sentence_a_prime
            sentence_B_prime = original_sentence_b_prime
    
    # qa.jsonl形式で出力
    result = {
        'original_qa_id': original_qa_id,
        'lex_unit_name': lex_unit_name,
        'question': four_choice_question,  # デフォルトはfour-choice-question
        'single-question': single_question,
        'two-choice-question': two_choice_question,
        'four-choice-question': four_choice_question,
        'sentence_A': sentence_A,
        'sentence_B': sentence_B,
        'sentence_A_prime': sentence_A_prime,
        'sentence_B_prime': sentence_B_prime,
        'frame_A': frame_A,
        'frame_B': frame_B
    }
    
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='入力TSVファイルのパス')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # TSVファイルを読み込む
    df = pd.read_csv(args.input_file, sep='\t')
    output_file = Path(args.input_file).parent.parent/'eval_round2' / 'qa.jsonl'
    # 各行を処理
    results = []
    seen_ids = set()  # 重複チェック用のセット
    for _, row in df.iterrows():
        result = process_row(row)
        if result is not None:
            # 重複チェック: original_qa_idで重複を防ぐ
            original_qa_id = result['original_qa_id']
            if original_qa_id not in seen_ids:
                seen_ids.add(original_qa_id)
                results.append(result)
            # 重複している場合はスキップ
    
    pd.DataFrame(results).to_json(output_file, orient='records', force_ascii=False, lines=True)


if __name__ == '__main__':
    main()
