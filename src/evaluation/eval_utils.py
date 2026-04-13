#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
四択問題評価の共通モジュール

データの読み込み、結果の保存、ユーティリティ関数を提供
"""

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import re
import random

os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

# プロジェクトルートディレクトリのパスを取得
PROJECT_ROOT = Path(__file__).parent.parent.parent

# デフォルトのプロンプトテンプレート（文A/文B形式）
DEFAULT_PROMPT_TEMPLATE = """
{four_choice_question}

文A: {statement_a}

文B: {statement_b}

選択肢: {choices_text}

回答する際は、文の最後の動詞に注目して判断してください。
回答は選択肢の番号「1」、「2」、「3」、「4」のいずれかで答えてください。
""".strip()


def load_prompt_template(prompt_file=None):
    """プロンプトテンプレートを読み込む
    
    Args:
        prompt_file: プロンプトファイルのパス（Noneの場合はデフォルト）
    
    Returns:
        prompt_template: プロンプトテンプレート文字列
    """
    if prompt_file is None:
        return DEFAULT_PROMPT_TEMPLATE
    
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        # 相対パスの場合はプロジェクトルートからの相対パス
        prompt_path = PROJECT_ROOT / prompt_path
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def extract_choice_number(text, thinking_stop=None):
    """テキストから選択肢番号（1, 2, 3, or 4）を抽出する関数

    以下の形式に対応:
    - JSON形式: {"answer": "1"} or {"answer": "2"} etc.
    - thinking形式: <think>...</think>（または任意のthinking_stop）の後の数字
    - プレーンテキスト: 1, 2, 3, or 4

    Args:
        text: 抽出対象のテキスト
        thinking_stop: thinkタグの終了文字列（例: "</think>"）。
                       指定された場合はその後のテキストを対象にする。
    """
    import json

    if text is None:
        return None

    text = str(text).strip()

    # JSON形式の場合（Structured Outputs）
    if text.startswith('{'):
        try:
            parsed = json.loads(text)
            answer = parsed.get("answer")
            if answer in ["1", "2", "3", "4"]:
                return int(answer)
        except json.JSONDecodeError:
            pass  # JSONパースに失敗したら通常の処理へ

    # thinking形式の場合: 指定されたstop文字列 → 既知フォールバックの順で試みる
    stop_candidates = []
    if thinking_stop:
        stop_candidates.append(thinking_stop)
    # 既知のフォールバック（thinking_stopと重複する場合はスキップ）
    for fallback in ["</think>", "assistantfinal"]:
        if fallback not in stop_candidates:
            stop_candidates.append(fallback)

    for stop in stop_candidates:
        if stop in text:
            text = text.split(stop)[-1].strip()
            break

    # 1, 2, 3, 4のいずれかが単独で出現する場合
    # 優先順位: 最初に出現する数字
    for i in range(1, 5):
        pattern = rf'\b{i}\b'
        if re.search(pattern, text):
            # 他の数字が含まれていないかチェック
            other_numbers = [j for j in range(1, 5) if j != i]
            has_other = any(re.search(rf'\b{j}\b', text) for j in other_numbers)
            if not has_other:
                return i
    
    # 複数の数字が含まれる場合は最初に出現するものを返す
    for i in range(1, 5):
        if str(i) in text:
            return i
    
    return None


def parse_original_question_text(question_text):
    """original_questionフィールドからquestion, sentence_a, sentence_b, sentence_a_prime, sentence_b_primeを抽出"""
    if pd.isna(question_text):
        return None, None, None, None, None
    
    text = str(question_text).strip()
    
    # 各センテンスの位置を探す
    sentence_a_pos = text.find('Sentence a:')
    sentence_b_pos = text.find('Sentence b:')
    sentence_a_prime_pos = text.find('Sentence a prime:')
    sentence_b_prime_pos = text.find('Sentence b prime:')
    
    # いずれかが見つからない場合
    if sentence_a_pos == -1 or sentence_b_pos == -1 or sentence_a_prime_pos == -1 or sentence_b_prime_pos == -1:
        return None, None, None, None, None
    
    # 問題文を抽出（Sentence a: の前まで）
    question_line = text[:sentence_a_pos].strip()
    # "Question: "というプレフィックスを除去
    if question_line.startswith('Question:'):
        question_line = question_line[len('Question:'):].strip()
    
    # Sentence aを抽出（Sentence a: の後からSentence b: の前まで）
    sentence_a = text[sentence_a_pos + len('Sentence a:'):sentence_b_pos].strip()
    
    # Sentence bを抽出（Sentence b: の後からSentence a prime: の前まで）
    sentence_b = text[sentence_b_pos + len('Sentence b:'):sentence_a_prime_pos].strip()
    
    # Sentence a primeを抽出（Sentence a prime: の後からSentence b prime: の前まで）
    sentence_a_prime = text[sentence_a_prime_pos + len('Sentence a prime:'):sentence_b_prime_pos].strip()
    
    # Sentence b primeを抽出（Sentence b prime: の後から最後まで）
    sentence_b_prime = text[sentence_b_prime_pos + len('Sentence b prime:'):].strip()
    
    return question_line, sentence_a, sentence_b, sentence_a_prime, sentence_b_prime


def randomize_japanese_choices(correct_choice_type):
    """日本語形式で選択肢の順番をランダムにして、正解の番号を返す
    
    Args:
        correct_choice_type: 正解のタイプ (1: Aのみ, 2: Bのみ, 3: 両方, 4: どちらも)
    
    Returns:
        choice_order: ランダム化された選択肢の情報リスト
        expected_choice: 正解の番号 (1-4)
    """
    # 4つの選択肢を定義 (choice_type, label)
    choices_info = [
        (1, "文A"),
        (2, "文B"), 
        (3, "文Aと文B"),
        (4, "該当なし")
    ]
    
    # ランダムにシャッフル
    shuffled = random.sample(choices_info, 4)
    
    # 選択肢の順番を作成し、正解の番号を見つける
    choice_order = []
    expected_choice = None
    
    for idx, (choice_type, label) in enumerate(shuffled, 1):
        choice_order.append({
            'number': idx,
            'type': choice_type,
            'label': label
        })
        
        # 正解の番号を記録
        if choice_type == correct_choice_type:
            expected_choice = idx
    
    return choice_order, expected_choice


def create_four_choice_problems(df):
    """データフレームから四択問題のリストを生成
    
    Args:
        df: データフレーム
        japanese_format: 日本語形式かどうか
    
    - original_questionフィールドがある場合: Sentence a/bとa'/b'から2つの問題を生成（正解はaとa'）
    - questionフィールドがある場合（qa.jsonl形式）: sentence_a/bとa_prime/b_primeから2つの問題を生成
    - 各問題は4択形式で、選択肢の順番はランダム化される
    """
    problems = []
    
    for idx, row in df.iterrows():
        question_line = row['question']
        sentence_a = row['sentence_a']
        sentence_b = row['sentence_b']
        sentence_a_prime = row.get('sentence_a_prime', None)
        sentence_b_prime = row.get('sentence_b_prime', None)
        matched_answer_score_ab = sum([
            int(row.get('match_answer_annotator1_ab', False)),
            int(row.get('match_answer_annotator2_ab', False)),
            int(row.get('match_answer_annotator3_ab', False))
        ])
        quality_score_ab = sum([
            int(row.get('quality_annotator1_ab', False)),
            int(row.get('quality_annotator2_ab', False)),
            int(row.get('quality_annotator3_ab', False))
        ])
        
        # 問題1: Sentence a vs b（正解はa = choice_type 1）
        choice_order_1, expected_choice_1 = randomize_japanese_choices(correct_choice_type=1)
        
        data_id = row.get('original_qa_id', f"row_{idx}")
        
        problems.append({
            'data_id': f"{data_id}_ab",
            'original_qa_id': row.get('original_qa_id', ''),
            'lex_unit_name': row.get('lex_unit_name', ''),
            'four_choice_question': question_line,
            'statement_a': sentence_a,
            'statement_b': sentence_b,
            'correct_choice_type': 1,  # Statement Aが正解
            'choice_order': choice_order_1,
            'expected_choice': expected_choice_1,
            'original_data': row.to_dict(),
            'matched_answer_score': matched_answer_score_ab,
            'quality_score': quality_score_ab,
            'problem_type': 'ab',
        })
        
        # 問題2: Sentence a' vs b'（正解はa' = choice_type 2）がある場合
        if pd.notna(sentence_a_prime) and pd.notna(sentence_b_prime):
            matched_answer_score_prime = sum([
                int(row.get('match_answer_annotator1_prime', False)),
                int(row.get('match_answer_annotator2_prime', False)),
                int(row.get('match_answer_annotator3_prime', False))
            ])
            quality_score_prime = sum([
                int(row.get('quality_annotator1_prime', False)),
                int(row.get('quality_annotator2_prime', False)),
                int(row.get('quality_annotator3_prime', False))
            ])
            # 注意: a'をStatement Bとして、b'をStatement Aとして配置
            choice_order_2, expected_choice_2 = randomize_japanese_choices(correct_choice_type=2)
            
            problems.append({
                'data_id': f"{data_id}_prime",
                'original_qa_id': row.get('original_qa_id', ''),
                'lex_unit_name': row.get('lex_unit_name', ''),
                'four_choice_question': question_line,
                'statement_a': sentence_b_prime,
                'statement_b': sentence_a_prime,
                'correct_choice_type': 2,  # Statement Bが正解（a'）
                'choice_order': choice_order_2,
                'expected_choice': expected_choice_2,
                'original_data': row.to_dict(),
                'matched_answer_score': matched_answer_score_prime,
                'quality_score': quality_score_prime,
                'problem_type': 'prime',
            })
    
    return problems


def parse_question_from_jsonl(question_text):
    """JSONL形式のquestionフィールドから質問文と2つの文を抽出
    
    Args:
        question_text: "質問文\nSentence 1: ...\nSentence 2: ..." 形式の文字列
    
    Returns:
        question_line: 質問文
        sentence_1: Sentence 1の内容
        sentence_2: Sentence 2の内容
    """
    if not question_text:
        return None, None, None
    
    text = str(question_text).strip()
    
    # Sentence 1とSentence 2の位置を探す
    sentence_1_pos = text.find('Sentence 1:')
    sentence_2_pos = text.find('Sentence 2:')
    
    if sentence_1_pos == -1 or sentence_2_pos == -1:
        return None, None, None
    
    # 質問文を抽出（Sentence 1: の前まで）
    question_line = text[:sentence_1_pos].strip()
    
    # Sentence 1を抽出（Sentence 1: の後からSentence 2: の前まで）
    sentence_1 = text[sentence_1_pos + len('Sentence 1:'):sentence_2_pos].strip()
    
    # Sentence 2を抽出（Sentence 2: の後から最後まで）
    sentence_2 = text[sentence_2_pos + len('Sentence 2:'):].strip()
    
    return question_line, sentence_1, sentence_2


def answer_to_choice_type(answer, choices):
    """JSONL形式のanswerをchoice_typeに変換
    
    Args:
        answer: 正解（例: "Sentence 1", "Sentence 2", "Both sentences", "Neither sentence"）
        choices: 選択肢のリスト
    
    Returns:
        choice_type: 1 (Sentence 1), 2 (Sentence 2), 3 (Both sentences), 4 (Neither sentence)
    """
    if answer == "Sentence 1":
        return 1
    elif answer == "Sentence 2":
        return 2
    elif answer == "Both sentences":
        return 3
    elif answer == "Neither sentence":
        return 4
    else:
        # 選択肢リストからインデックスを探す
        try:
            idx = choices.index(answer)
            return idx + 1
        except (ValueError, AttributeError):
            return None

def load_problems_from_hf_dataset(dataset_name, split='train', num=None):
    """HuggingFace Datasetから問題を読み込む
    
    Args:
        dataset_name: HuggingFace Datasetの名前（例: "username/dataset-name"）
        split: データセットのsplit（デフォルト: 'train'）
        num: 解く問題数（Noneまたは0の場合は全問）
    
    Returns:
        all_problems: 問題のリスト
    """
    from datasets import load_dataset
    
    # HuggingFace Datasetを読み込む
    dataset = load_dataset(dataset_name, split=split)
    
    # データセットをDataFrameに変換
    df = pd.DataFrame(dataset)
    
    # create_four_choice_problemsを使って問題を生成
    all_problems = create_four_choice_problems(df)
    
    print("生成された四択問題数: {}".format(len(all_problems)))
    
    # 解く問題数の設定
    if num is not None and num > 0:
        num_questions = min(num, len(all_problems))
        all_problems = all_problems[:num_questions]
        print("選択された問題数: {}".format(len(all_problems)))
    
    return all_problems


def load_problems(dataset_path, num=None, split='train'):
    """JSONLファイルまたはHuggingFace Datasetから問題を読み込む
    
    Args:
        dataset_path: JSONLファイルのパスまたはHuggingFace Datasetの名前
        num: 解く問題数（Noneまたは0の場合は全問）
        split: HF Datasetの場合のsplit（デフォルト: 'train'）
    
    Returns:
        all_problems: 問題のリスト
    """
    # ローカルファイルかHF Datasetかを判定
    file_path = Path(dataset_path)
    if file_path.exists():
        # ローカルファイルとして読み込む
        df = pd.read_json(dataset_path, lines=True)
    elif '/' in dataset_path:
        # HuggingFace Datasetとして読み込む
        from datasets import load_dataset
        dataset = load_dataset(dataset_path, split=split)
        df = pd.DataFrame(dataset)
    else:
        raise FileNotFoundError("データセットが見つかりません: {}".format(dataset_path))
    all_problems = create_four_choice_problems(df)
    print("生成された四択問題数: {}".format(len(all_problems)))
    if num is not None and num > 0:
        num_questions = min(num, len(all_problems))
        all_problems = all_problems[:num_questions]
        print("選択された問題数: {}".format(len(all_problems)))
    return all_problems


def create_messages(problems, prompt_template=None):
    """問題リストからメッセージリストを生成
    
    Args:
        problems: 問題のリスト（choice_orderを含む）
        prompt_template: プロンプトテンプレート（Noneの場合はデフォルト）
    
    Returns:
        all_messages: メッセージのリスト（各要素は[{"role": "user", "content": ...}]）
    """
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    
    all_messages = []
    for prob in problems:
        # choice_orderから選択肢ラベルを生成
        choices_text = ""
        for choice in prob['choice_order']:
            choices_text += f"{choice['number']}: {choice['label']}     "
        choices_text = choices_text.strip()
        
        messages = [{"role": "user", "content": prompt_template.format(
            four_choice_question=prob['four_choice_question'],
            statement_a=prob['statement_a'],
            statement_b=prob['statement_b'],
            choices_text=choices_text,
        )}]
        
        all_messages.append(messages)
    return all_messages


def process_results(all_problems, generated_texts, thinking_stop=None):
    """生成されたテキストを処理して結果を集計

    Args:
        all_problems: 問題のリスト
        generated_texts: 生成されたテキストのリスト
        thinking_stop: thinkタグの終了文字列（例: "</think>"）。
                       extract_choice_number に渡される。

    Returns:
        all_problems: 結果が追加された問題のリスト
        stats: 統計情報の辞書
    """
    choice_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    choice_type_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 元の選択肢タイプごとのカウント
    error_count = 0
    scores = []

    for i, generated_text in enumerate(generated_texts):
        extracted_choice = extract_choice_number(generated_text, thinking_stop=thinking_stop)
        
        # ランダム化された選択肢から正解を見つける
        expected_choice = all_problems[i]['expected_choice']
        
        # LLM回答の分布（選択肢番号）
        if extracted_choice in [1, 2, 3, 4]:
            choice_counts[extracted_choice] += 1
            
            # 選択肢番号を元の選択肢タイプに変換
            choice_order = all_problems[i]['choice_order']
            for choice in choice_order:
                if choice['number'] == extracted_choice:
                    choice_type = choice['type']
                    choice_type_counts[choice_type] += 1
                    break
        else:
            error_count += 1
        
        score = 1 if extracted_choice == expected_choice else 0
        all_problems[i]['llm_response'] = generated_text
        all_problems[i]['extracted_choice'] = extracted_choice
        all_problems[i]['score'] = score
        scores.append(score)
    
    stats = {
        'total': len(all_problems),
        'choice1_count': choice_counts[1],
        'choice2_count': choice_counts[2],
        'choice3_count': choice_counts[3],
        'choice4_count': choice_counts[4],
        'choice_type1_count': choice_type_counts[1],  # 文A
        'choice_type2_count': choice_type_counts[2],  # 文B
        'choice_type3_count': choice_type_counts[3],  # 文Aと文B
        'choice_type4_count': choice_type_counts[4],  # 該当なし
        'error_count': error_count,
        'scores': scores,
        'accuracy': sum(scores) / len(all_problems) if all_problems else 0
    }
    
    return all_problems, stats


def print_stats(stats):
    """統計情報を表示"""
    total = stats['total']
    print("\n" + "="*50)
    print("LLM回答の分布:")
    
    print(f"  選択肢1: {stats['choice1_count']}/{total} ({stats['choice1_count']/total*100:.1f}%)")
    print(f"  選択肢2: {stats['choice2_count']}/{total} ({stats['choice2_count']/total*100:.1f}%)")
    print(f"  選択肢3: {stats['choice3_count']}/{total} ({stats['choice3_count']/total*100:.1f}%)")
    print(f"  選択肢4: {stats['choice4_count']}/{total} ({stats['choice4_count']/total*100:.1f}%)")
    
    print("\n元の選択肢タイプ別:")
    print(f"  文A: {stats['choice_type1_count']}/{total} ({stats['choice_type1_count']/total*100:.1f}%)")
    print(f"  文B: {stats['choice_type2_count']}/{total} ({stats['choice_type2_count']/total*100:.1f}%)")
    print(f"  文Aと文B: {stats['choice_type3_count']}/{total} ({stats['choice_type3_count']/total*100:.1f}%)")
    print(f"  該当なし: {stats['choice_type4_count']}/{total} ({stats['choice_type4_count']/total*100:.1f}%)")
    
    print(f"\n  抽出失敗: {stats['error_count']}/{total} ({stats['error_count']/total*100:.1f}%)")
    print(f"  正答率: {stats['accuracy']*100:.1f}%")
    print("="*50)


def save_results(output_dir, all_problems, stats):
    """結果を保存
    
    Args:
        output_dir: 出力ディレクトリ（Path）
        all_problems: 結果が追加された問題のリスト
        stats: 統計情報の辞書
        japanese_format: 日本語形式かどうか
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = stats['total']
    
    # 結果のサマリーを保存
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"選択肢1: {stats['choice1_count']}/{total} ({stats['choice1_count']/total*100:.1f}%)\n")
        f.write(f"選択肢2: {stats['choice2_count']}/{total} ({stats['choice2_count']/total*100:.1f}%)\n")
        f.write(f"選択肢3: {stats['choice3_count']}/{total} ({stats['choice3_count']/total*100:.1f}%)\n")
        f.write(f"選択肢4: {stats['choice4_count']}/{total} ({stats['choice4_count']/total*100:.1f}%)\n")
        f.write("\n")
        f.write("元の選択肢タイプ別:\n")
        f.write(f"文A: {stats['choice_type1_count']}/{total} ({stats['choice_type1_count']/total*100:.1f}%)\n")
        f.write(f"文B: {stats['choice_type2_count']}/{total} ({stats['choice_type2_count']/total*100:.1f}%)\n")
        f.write(f"文Aと文B: {stats['choice_type3_count']}/{total} ({stats['choice_type3_count']/total*100:.1f}%)\n")
        f.write(f"該当なし: {stats['choice_type4_count']}/{total} ({stats['choice_type4_count']/total*100:.1f}%)\n")
        f.write("\n")
        f.write(f"抽出失敗: {stats['error_count']}/{total} ({stats['error_count']/total*100:.1f}%)\n")
        f.write(f"正答率: {stats['accuracy']*100:.1f}%\n")
        f.write(f"正解数: {sum(stats['scores'])}\n")
        f.write(f"不正解数: {total - stats['error_count'] - sum(stats['scores'])}\n")
        f.write(f"エラー数: {stats['error_count']}\n")
        f.write(f"総問題数: {total}\n")
    print(f"サマリーを {summary_file} に保存しました")
    
    # TSVファイルも生成（アノテーション情報を含む）
    tsv_file = output_dir / "result.tsv"
    pd.DataFrame(all_problems).to_csv(tsv_file, sep="\t", index=False)
    print(f"結果（アノテーション情報含む）を {tsv_file} に保存しました")


def create_output_dir(base_output_dir, language, num, model_name, suffix=""):
    """出力ディレクトリを作成
    
    Args:
        base_output_dir: 出力ディレクトリのベースパス
        language: 言語
        num: 問題数
        model_name: モデル名
        suffix: サフィックス（例: "_thinking"）
        japanese_format: 日本語形式かどうか
    
    Returns:
        output_dir: 出力ディレクトリ（Path）
    """
    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    
    num_str = str(num) if num is not None else "all"
    output_dir = Path(base_output_dir) / language / num_str / f"{safe_model_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
