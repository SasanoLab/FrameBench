#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4文構造のQAデータを生成するスクリプト

既存の2文QAデータから、以下の4文構造を持つデータを生成します：
- Sentence A: 元の答えの文（Yesになる文）
- Sentence B: 元の答えでない文（Noになる文）
- Sentence A': 追加で生成されたYesになる文
- Sentence B': 追加で生成されたNoになる文

使い方：
    python src/generation/3_gen_additional_sentences.py --qa_file <入力ファイル> --output_dir <出力ディレクトリ>

特徴：
- Both sentencesとNeither sentenceを同時に生成してマージ
- バッチ処理（デフォルト50ペア = 100プロンプトずつ）
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import tomllib
from jinja2 import Template
from tqdm import tqdm
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# 既存モジュールのインポート
from utils.llm import generate_batch
from utils.utils import process_json_response

# フレームデータのパス
DATA_PATHS = {
    'ja': {
        'frames': "data/ja-framenet/frames.jsonl",
        'exemplars': "data/ja-framenet/exemplars.jsonl"
    },
    'en': {
        'frames': "data/en-framenet/frames.jsonl",
        'exemplars': "data/en-framenet/exemplars.jsonl"
    }
}


class AdditionalChoiceGenerator:
    """追加の選択肢を持つ問題を生成するクラス"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", language: str = 'ja', 
                 prompt_file: str = "prompts/gen_other_choice_ja.toml", seed: int = 42):
        self.model_name = model_name
        self.language = language
        self.seed = seed
        self.frame_data = {}

        
        self.prompt_template = self._load_prompt(prompt_file)
        
        # ランダムシードの設定
        random.seed(seed)
    
    def _load_prompt(self, prompt_file: str) -> str:
        """TOMLファイルからプロンプトテンプレートを読み込む"""
        # 相対パスの場合は、スクリプトのディレクトリから解決
        if not os.path.isabs(prompt_file):
            script_dir = Path(__file__).parent
            prompt_file = script_dir / prompt_file
        
        with open(prompt_file, 'rb') as f:
            prompt_data = tomllib.load(f)
        
        # messagesからcontentを抽出
        if 'messages' in prompt_data and prompt_data['messages']:
            return prompt_data['messages'][0]['content']
        else:
            raise ValueError(f"プロンプトファイル {prompt_file} にメッセージが見つかりません")
    
    def load_frame_definitions(self, frames_path: str = None, exemplars_path: str = None):
        """フレーム定義を読み込む"""
        if frames_path is None:
            frames_path = DATA_PATHS[self.language]['frames']
        if exemplars_path is None:
            exemplars_path = DATA_PATHS[self.language]['exemplars']
        
        if not Path(frames_path).exists():
            raise FileNotFoundError(f"Frame definitions file not found: {frames_path}")
        
        if not Path(exemplars_path).exists():
            raise FileNotFoundError(f"Exemplars file not found: {exemplars_path}")
        
        self.exemplars = pd.read_json(exemplars_path, lines=True, orient='records')
        
        with open(frames_path, 'r', encoding='utf-8') as f:
            for line in f:
                frame_data = json.loads(line)
                self.frame_data[frame_data['frame_name']] = frame_data
        
        print(f"フレーム定義を{len(self.frame_data)}件読み込みました")
        print(f"例文を{len(self.exemplars)}件読み込みました")
    
    def load_qa_data(self, qa_file_path: str) -> List[Dict[str, Any]]:
        """QAデータを読み込む"""
        if not Path(qa_file_path).exists():
            raise FileNotFoundError(f"QA data file not found: {qa_file_path}")
        
        qa_data = []
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 必要なキーが存在するか確認
                    if 'lex_unit_name' in item:
                        qa_data.append(item)
                except json.JSONDecodeError:
                    continue
        
        print(f"QAデータを{len(qa_data)}件読み込みました")
        return qa_data
    
    def _format_frame_definition(self, frame_name: str, verb: str) -> str:
        """フレーム定義をフォーマット"""
        if frame_name not in self.frame_data:
            return f"フレーム名: {frame_name}\n定義: （定義が見つかりません）"
        
        frame = self.frame_data[frame_name]
        definition = frame.get('definition', '定義なし')
        examples = self.exemplars[(self.exemplars['frame'] == frame_name) & (self.exemplars['verb'] == verb)]['sentence'].tolist()[:3]
        
        # コアフレーム要素を追加
        core_fes = frame.get('core_frame_elements', [])
        if core_fes:
            fes_str = "\n".join([f"  - {fe['name']}: {fe['definition']}" for fe in core_fes[:3]])  # 最大3つまで
            return f"フレーム名: {frame_name}\n定義: {definition}\nコアフレーム要素:\n{fes_str}\n例文:\n{examples}"
        else:
            return f"フレーム名: {frame_name}\n定義: {definition}\n例文:\n{examples}"
    
    def _extract_question_without_sentences(self, full_question: str) -> str:
        """質問から文を除いた質問部分のみを抽出"""
        # "Sentence 1:"で分割して最初の部分を取得
        if "Sentence 1:" in full_question:
            return full_question.split("Sentence 1:")[0].strip()
        return full_question
    
    def create_prompt_for_item(self, qa_item: Dict[str, Any], target_answer: str) -> Optional[tuple]:
        """
        QAアイテムから新しい文を生成するためのプロンプトを作成
        
        Args:
            qa_item: QAデータ項目
            target_answer: 目標とする答え（"Both sentences" または "Neither sentence"）
        
        Returns:
            (プロンプト, フレーム名, 文, 動詞)のタプル、またはNone
        """
        print(qa_item)
        try:
            lex_unit_name = qa_item['lex_unit_name']
            
            # 動詞を抽出
            verb = lex_unit_name.split('.')[0]
            
            # 質問を抽出
            single_question = qa_item['single-question']
            
            # 生成戦略を決定
            if target_answer == "Both sentences":
                target_frame_name = qa_item['frame_A']
                target_sentence = qa_item['sentence_A']
                answer_instruction = "肯定"
                target_answer = "Yes"
            elif target_answer == "Neither sentence":
                target_frame_name = qa_item['frame_B']
                target_sentence = qa_item['sentence_B']
                answer_instruction = "否定"
                target_answer = "No"
            print(single_question)
            
            # フレーム定義を取得
            frame_definition = self._format_frame_definition(target_frame_name, verb)
            
            # プロンプトをレンダリング
            template = Template(self.prompt_template)
            rendered_prompt = template.render(
                single_question=single_question,
                verb=verb,
                definitions_a=frame_definition,
                sentence=target_sentence,
                frame_a=target_frame_name,
                answer_ja=answer_instruction,
                Answer=target_answer,
                answer_choice=target_answer
            )
            
            messages = [{"role": "user", "content": rendered_prompt}]
            
            return messages, target_frame_name, target_sentence, verb
        
        except Exception as e:
            print(f"プロンプト作成エラー: {e}")
            return None
    

    def generate_four_sentence_qa(self, qa_data: List[Dict[str, Any]],
                                 max_items: Optional[int] = None,
                                 output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        4文構造のQAデータを生成（最初からA, B, A', B'形式で）
        
        Args:
            qa_data: QAデータのリスト
            max_items: 処理する最大アイテム数（Noneの場合は全て）
            output_dir: 出力ディレクトリ
        
        Returns:
            4文構造のDataFrame（A, B, A', B'）
        """
      
        items_to_process = qa_data[:max_items] if max_items else qa_data
        
       
        print(f"\n{len(items_to_process)}件のアイテムについて、BothとNeitherの両方を生成します...")
        
        # 各アイテムについて、BothとNeither両方のプロンプトを作成
        prompt_pairs = []  # [(both_prompt, neither_prompt, item, both_meta, neither_meta), ...]
        valid_items = []
        
        for item in tqdm(items_to_process, desc="プロンプト作成中"):
            both_result = self.create_prompt_for_item(item, "Both sentences")
            neither_result = self.create_prompt_for_item(item, "Neither sentence")
            
            if both_result and neither_result:
                both_prompt, both_frame, both_sentence, _ = both_result
                neither_prompt, neither_frame, neither_sentence, _ = neither_result
                
                prompt_pairs.append({
                    'item': item,
                    'both_prompt': both_prompt,
                    'neither_prompt': neither_prompt,
                    'both_meta': {'frame': both_frame, 'sentence': both_sentence},
                    'neither_meta': {'frame': neither_frame, 'sentence': neither_sentence}
                })
                valid_items.append(item)
        
        print(f"\n{len(prompt_pairs)}件のペアを作成しました")
        
        if not prompt_pairs:
            print("生成するプロンプトがありません")
            return pd.DataFrame()
        
        # プロンプトをファイルに保存（デバッグ用）
        if output_dir:
            prompts_file = output_dir / "prompts_pairs.txt"
            with open(prompts_file, 'w', encoding='utf-8') as f:
                for i, pair in enumerate(prompt_pairs):
                    f.write("=" * 60 + "\n")
                    f.write(f"Pair {i+1}/{len(prompt_pairs)}\n")
                    f.write("=" * 60 + "\n")
                    f.write("\n--- Both sentences prompt ---\n")
                    f.write(pair['both_prompt'])
                    f.write("\n\n--- Neither sentence prompt ---\n")
                    f.write(pair['neither_prompt'])
                    f.write("\n\n")
            print(f"プロンプトペアを保存: {prompts_file}")
        
        # バッチサイズ（各バッチで処理するペア数）
        batch_size = 50  # 50ペア = 100プロンプト
        
        print(f"\nバッチサイズ: {batch_size}ペア（{batch_size * 2}プロンプト）ずつ処理します")
        
        # バッチごとに処理
        final_data = []
        for batch_start in range(0, len(prompt_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(prompt_pairs))
            batch_pairs = prompt_pairs[batch_start:batch_end]
            
            print(f"\nバッチ {batch_start // batch_size + 1}/{(len(prompt_pairs) + batch_size - 1) // batch_size}: {len(batch_pairs)}ペアを処理中...")
            
            # バッチ内のプロンプトをフラット化
            batch_prompts = []
            for pair in batch_pairs:
                batch_prompts.append(pair['both_prompt'])
                batch_prompts.append(pair['neither_prompt'])
            
            try:
                # バッチ生成
                results = generate_batch(
                    prompts=batch_prompts,
                    model_name=self.model_name,
                    seed=self.seed
                )
                
                # 結果をペアに戻してパース
                batch_data = []
                for i, pair in enumerate(batch_pairs):
                    both_idx = i * 2
                    neither_idx = i * 2 + 1
                    both_result = results[both_idx]
                    neither_result = results[neither_idx]
                    # パース
                    both_parsed = process_json_response(both_result)
                    neither_parsed = process_json_response(neither_result)
                    
                    if both_parsed and neither_parsed:
                        item = pair['item']
                        # 生成された文を取得
                        both_generated = both_parsed.get('generated_sentence', '')
                        neither_generated = neither_parsed.get('generated_sentence', '')
                        sentence_A_prime = both_generated
                        sentence_B_prime = neither_generated
                        batch_data.append({
                            'original_qa_id': item['original_qa_id'],
                            'lex_unit_name': item['lex_unit_name'],
                            'question': item['single-question'],
                            'single-question': item['single-question'],
                            'two-choice-question': item['two-choice-question'],
                            'four-choice-question': item['four-choice-question'],
                            'sentence_A': item['sentence_A'],
                            'sentence_B': item['sentence_B'],
                            'sentence_A_prime': sentence_A_prime,
                            'sentence_B_prime': sentence_B_prime,
                            'frame_A': item['frame_A'],
                            'frame_B': item['frame_B'],
                        })
                final_data.extend(batch_data)
            except Exception as e:
                print(f"バッチ処理中にエラーが発生: {e}")
                break
        
        return pd.DataFrame(final_data)
    
    def generate_additional_choices(self, qa_data: List[Dict[str, Any]], 
                                   target_answer: str = "Both sentences",
                                   max_items: Optional[int] = None,
                                   output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        追加の選択肢を持つ問題を生成
        
        Args:
            qa_data: QAデータのリスト
            target_answer: 目標とする答え（"Both sentences" または "Neither sentence"）
            max_items: 処理する最大アイテム数（Noneの場合は全て）
        
        Returns:
            生成結果のDataFrame
        """
        # プロンプトを作成
        prompts = []
        valid_items = []
        used_for_generations = []
        
        items_to_process = qa_data[:max_items] if max_items else qa_data
        
        # 処理済みのアイテムをフィルタリング（動的にIDを生成して比較）
        data_flag = "yes" if target_answer == "Both sentences" else "no"
        for item in tqdm(items_to_process):
            qa_id = item['original_qa_id'] + "_" + data_flag
            print(f"\n{target_answer}を生成するプロンプトを作成中...")
            result = self.create_prompt_for_item(item, target_answer)
            if result:
                prompt, target_frame_name, target_sentence, verb = result
                prompts.append(prompt)
                valid_items.append(item)
                used_for_generations.append({"frame_name": target_frame_name, "sentence": target_sentence, "lex_unit_name": item['lex_unit_name']})
        
        print(f"\n{len(prompts)}件のプロンプトを作成しました")
        
        if not prompts:
            print("生成するプロンプトがありません")
            return pd.DataFrame()
        
        # プロンプトをファイルに保存
        if output_dir:
            prompts_file = output_dir / f"prompts_{target_answer.replace(' ', '_').lower()}.txt"
            with open(prompts_file, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(prompts):
                    f.write("=" * 60 + "\n")
                    f.write(f"Prompt {i+1}/{len(prompts)}\n")
                    f.write("=" * 60 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
            print(f"プロンプトを保存: {prompts_file}")
        
        # サンプルプロンプトを表示
        print("\n=== サンプルプロンプト ===")
        print(prompts[0])
        print("=" * 50)
        
        # LLMで生成
        print(f"\n{self.model_name}を使用して文を生成中...")
        results = generate_batch(
            self.model_name,
            prompts,
        )
        
        # 結果を処理
        output_data = []
        for i, (item, prompt, result, used_for_generation) in enumerate(zip(valid_items, prompts, results,used_for_generations)):
            parsed_result = process_json_response(result)
            # original_qaから一意なIDを生成
            data_flag = "yes" if target_answer == "Both sentences" else "no"
            qa_id = item['original_qa_id'] + "_" + data_flag
            output_data.append({
                'original_qa_id': item['original_qa_id'],
                "id": qa_id,
                'lex_unit_name': item['lex_unit_name'],
                'sentence_pair': item['sentence_pair'],
                'original_qa': item['result'],
                'generated_response': result,
                'parsed_result': parsed_result,
                'target_answer': target_answer,
                "used_frame_name": used_for_generation['frame_name'],
                "used_sentence": used_for_generation['sentence'],
                "single_question": item['single_question'],
                "two-choice-question": item['two-choice-question'],
                "four-choice-question": item['four-choice-question'],
            })
        
        # 新しく生成したデータのみを返す（既存データとの結合は後で行う）
        return pd.DataFrame(output_data)
    
    def merge_and_label_sentences(self, both_df: pd.DataFrame, neither_df: pd.DataFrame) -> pd.DataFrame:
        """
        Both sentencesとNeither sentenceの生成結果をマージし、A, B, A', B'にラベル付け
        
        Args:
            both_df: Both sentences用の中間結果（生成されたYesになる文を含む）
            neither_df: Neither sentence用の中間結果（生成されたNoになる文を含む）
        
        Returns:
            マージされた最終データのDataFrame
            カラム: original_qa_id, question, sentence_A, sentence_B, sentence_A_prime, sentence_B_prime, 
                   frame_A, frame_B, lex_unit_name
        """
        merged_data = []
        
        # original_qa_idでグループ化してマージ
        both_dict = {row['original_qa_id']: row for _, row in both_df.iterrows()}
        neither_dict = {row['original_qa_id']: row for _, row in neither_df.iterrows()}
        
        # 共通のoriginal_qa_idを取得
        common_ids = set(both_dict.keys()) & set(neither_dict.keys())
        
        for qa_id in common_ids:
            both_row = both_dict[qa_id]
            neither_row = neither_dict[qa_id]
            
            # 元のQAから答えを取得
            original_qa = both_row['original_qa']
            answer = original_qa.get('answer', '')
            
            # sentence_pairから2つの文とフレーム名を取得
            sentence_pair = both_row['sentence_pair']
            frame_names = list(sentence_pair.keys())
            sentences = list(sentence_pair.values())
            
            # Sentence 1: / Sentence 2: のプレフィックスを除去
            sentence_1 = sentences[0].replace("Sentence 1: ", "") if sentences[0].startswith("Sentence 1: ") else sentences[0]
            sentence_2 = sentences[1].replace("Sentence 2: ", "") if sentences[1].startswith("Sentence 2: ") else sentences[1]
            
            # 生成された文を取得
            both_generated = both_row['parsed_result'].get('generated_sentence', '') if both_row['parsed_result'] else ''
            neither_generated = neither_row['parsed_result'].get('generated_sentence', '') if neither_row['parsed_result'] else ''
            
            # 答えに応じてA, B, A', B'にラベル付け
            if answer == "Sentence 1":
                # Sentence 1が答え（Yesになる文）
                sentence_A = sentence_1
                sentence_B = sentence_2
                frame_A = frame_names[0]
                frame_B = frame_names[1]
                # Both sentencesではSentence 2を書き換えてYesにする → A'
                sentence_A_prime = both_generated
                # Neither sentenceではSentence 1を書き換えてNoにする → B'
                sentence_B_prime = neither_generated
            elif answer == "Sentence 2":
                # Sentence 2が答え（Yesになる文）
                sentence_A = sentence_2
                sentence_B = sentence_1
                frame_A = frame_names[1]
                frame_B = frame_names[0]
                # Both sentencesではSentence 1を書き換えてYesにする → A'
                sentence_A_prime = both_generated
                # Neither sentenceではSentence 2を書き換えてNoにする → B'
                sentence_B_prime = neither_generated
            else:
                # 答えが"Both sentences"や"Neither sentence"の場合はスキップ
                continue
            
            merged_data.append({
                'original_qa_id': qa_id,
                'lex_unit_name': both_row['lex_unit_name'],
                'single_question': both_row['single_question'],
                'two-choice-question': both_row['two-choice-question'],
                'four-choice-question': both_row['four-choice-question'],
                'sentence_A': sentence_A,
                'sentence_B': sentence_B,
                'sentence_A_prime': sentence_A_prime,
                'sentence_B_prime': sentence_B_prime,
                'frame_A': frame_A,
                'frame_B': frame_B,
            })
        
        return pd.DataFrame(merged_data)
    
    def process_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成結果を処理して、最終的なQAフォーマットに変換
        
        Args:
            results_df: 生成結果のDataFrame
        
        Returns:
            最終的なQAデータのDataFrame
        """
        final_data = []
        
        for _, row in results_df.iterrows():
            if row['parsed_result'] is None:
                continue
            
            try:
                parsed = row['parsed_result']
                question = row['two-choice-question']
                generated_sentence = parsed['generated_sentence']
                original_sentence = row['used_sentence']
                frame_name = row['used_frame_name']
                answer = row['target_answer']
                
                # sentence_pairから2つの文を取得
                sentences = list(row['sentence_pair'].values())
                
                # 答えに応じて文を配置
                if answer == "Both sentences":
                    # 両方が答えになるように調整
                    sentence_1 = generated_sentence if generated_sentence else sentences[0]
                    sentence_2 = original_sentence if original_sentence else sentences[1]
                elif answer == "Neither sentence":
                    # どちらも答えでなくなるように調整
                    sentence_1 = generated_sentence if generated_sentence else sentences[0]
                    sentence_2 = original_sentence if original_sentence else sentences[1]
                else:
                    continue
                
                # 完全な質問を構築
                full_question = f"{question}\nSentence 1: {sentence_1}\nSentence 2: {sentence_2}"
                
                final_qa = {
                    'question': full_question,
                    'choices': ["Sentence 1", "Sentence 2", "Both sentences", "Neither sentence"],
                    'answer': answer
                }
                
                final_data.append({
                    'lex_unit_name': row['lex_unit_name'],
                    "frame_name": frame_name,
                    'sentence_pair': {frame_name: sentence_1, frame_name: sentence_2},
                    'result': final_qa,
                    'generation_type': row['target_answer'],
                    'original_qa_id': row['original_qa_id'],
                    "single_question": row['single_question'],
                    "two-choice-question": row['two-choice-question'],
                    "four-choice-question": row['four-choice-question'],
                })
            
            except Exception as e:
                print(f"結果処理エラー: {e}")
                continue
        
        return pd.DataFrame(final_data)


def main():
    parser = argparse.ArgumentParser(description='追加の選択肢を持つ問題を生成')
    parser.add_argument('--qa_file', type=str, required=True,
                       help='入力QAファイルのパス')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='出力ディレクトリ')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='使用するLLMモデル')
    parser.add_argument('--language', type=str, default='ja',
                       choices=['ja', 'en'], help='言語')
    parser.add_argument('--target_answer', type=str, default='all',
                       choices=['both', 'neither', 'all'],
                       help='（非推奨：このパラメータは無視されます。常に両方を生成します）')
    parser.add_argument('--max_items', type=int, default=None,
                       help='処理する最大アイテム数')
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ジェネレータの初期化
    generator = AdditionalChoiceGenerator(
        model_name=args.model,
        language=args.language,
        seed=args.seed
    )
    
    # フレーム定義を読み込み
    generator.load_frame_definitions()
    
    # QAデータを読み込み
    qa_data = generator.load_qa_data(args.qa_file)
    
    results_df = generator.generate_four_sentence_qa(
        qa_data,
        max_items=args.max_items,
        output_dir=output_dir
    )
    
    if not results_df.empty:
        final_output_file = output_dir / "qa.jsonl"
        results_df.to_json(final_output_file, orient='records', lines=True, force_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"  生成完了")
        print(f"{'='*60}")
        print(f"最終結果: {len(results_df)}件")
        print(f"保存先: {final_output_file}")
        
        # データ構造の確認
        print("\n=== データ構造 ===")
        print(f"カラム: {list(results_df.columns)}")
        
        # サンプルを表示
        if len(results_df) > 0:
            print("\n=== サンプルデータ（1件目）===")
            sample = results_df.iloc[0]
            print(f"Question: {sample['question']}")
            print(f"Sentence A (Yes): {sample['sentence_A']}")
            print(f"Sentence B (No): {sample['sentence_B']}")
            print(f"Sentence A' (Yes): {sample['sentence_A_prime']}")
            print(f"Sentence B' (No): {sample['sentence_B_prime']}")
        
    else:
        print("\n生成された結果がありません")


if __name__ == "__main__":
    main()

