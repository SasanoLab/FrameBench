#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
フレームペアを生成して、問題文を生成する。

Usage:
python src/generation/2_generate_frame_qa.py
    --data_root <data_root>
    --language <language>
    --model <model>
    --num_pairs <num_pairs>
    --no_quality_filter <no_quality_filter>
    --num <num>
    --include_definition <include_definition>
    --include_thinking <include_thinking>
    --num_examples <num_examples>
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import tomllib
import itertools
from jinja2 import Template
import json
import hashlib
from dotenv import load_dotenv
import os

from utils.llm import generate_batch, list_available_models
from utils.utils import process_json_response, setup_output_directory, SUPPORTED_LANGUAGES_MAP, load_jsonl
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFINITION = """
## {frame_name} 
Definition: {definition}
Core Frame Elements: {core_FEs}
Examples: {examples}
""".strip()


class FrameQAPipeline:
    """FrameNet QA生成パイプライン"""
    
    def __init__(self, model: str = "gpt-4o-mini", language: str = "ja", max_tokens: int = 5000, seed: int = 42, prompt_file: str = None, 
                 include_definition: bool = True, include_thinking: bool = True, num_examples: int = 3):
        """
        Args:
            model: 使用するLLMモデル
            language: 生成言語
            max_tokens: 最大トークン数
            seed: ランダムシード
            prompt_file: プロンプトファイルのパス
            *以下は実験用のパラメータ
            include_definition: フレーム定義を含めるかどうか
            include_thinking: thinking_processを出力させるかどうか
            num_examples: フレームごとの例文の数
        """
        self.model = model
        self.language = language
        self.max_tokens = max_tokens
        self.seed = seed
        self.include_definition = include_definition
        self.include_thinking = include_thinking
        self.num_examples = num_examples
        
        # シードの設定
        np.random.seed(seed)

        self.prompt_file = prompt_file
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"プロンプトファイルが見つかりません: {self.prompt_file}")
        self.prompt = self._load_prompt(self.prompt_file)

        # 実験条件に応じた出力ディレクトリを生成
        exp_suffix = self._generate_experiment_suffix()
        base_dir = setup_output_directory(model, language)
        self.output_dir = f"{base_dir}{exp_suffix}/step2" if exp_suffix else f"{base_dir}/step2"
        
        if language not in SUPPORTED_LANGUAGES_MAP:
            raise ValueError(f"無効な言語: {language}")
        
        logger.info(f"利用するプロンプトファイル: {self.prompt_file}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
    
    def _generate_experiment_suffix(self) -> str:
        """実験条件に応じたディレクトリサフィックスを生成"""
        suffix_parts = []
        
        # デフォルト設定（定義あり、thinkingあり、例文3つ）からの変更のみを記録
        if not self.include_definition:
            suffix_parts.append("nodef")
        if not self.include_thinking:
            suffix_parts.append("nothink")
        if self.num_examples != 3:
            suffix_parts.append(f"ex{self.num_examples}")
        
        return "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    def _load_prompt(self, prompt_file: str) -> List[Dict[str, str]]:
        """TOMLファイルからプロンプトを読み込む（メッセージのリストとして返す）"""
        try:
            with open(self.prompt_file, 'rb') as f:
                prompt_data = tomllib.load(f)
            
            # messagesを全て取得
            if 'messages' in prompt_data and prompt_data['messages']:
                messages = prompt_data['messages']
                return messages
            else:
                logger.warning(f"プロンプトファイル {prompt_file} にメッセージが見つかりません")
                return []
        except Exception as e:
            logger.error(f"プロンプトファイルの読み込み中にエラーが発生: {e}")
            raise e

    def load_and_filter_lexical_units(self, data_path: str) -> pd.DataFrame:
        """lexical unitsデータを読み込み、多義動詞のみをフィルタリング"""
        data = load_jsonl(data_path, "lexical units")
        data = data[data['lex_unit_pos'] == 'V']
        data = data[data['frames'].map(lambda x: len(x) > 1)]
        logger.info(f"多義動詞のLU数: {len(data)}")
        return data
    
    def choice_all_frame_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """全てのフレームペアを選択してDataFrameで返す"""
        output_df = []
        for idx, row in data.iterrows():
            frame_pairs = list(itertools.combinations(row["frames"], 2))
            output_df.extend([{
                'lex_unit_name': row['lex_unit_name'],
                'frame_pair': frame_pair,
            } for frame_pair in frame_pairs])
        return pd.DataFrame(output_df)

    def concat_info(self, frame_pairs: pd.DataFrame, frame_data: pd.DataFrame,frame_examples: pd.DataFrame) -> pd.DataFrame:
        """frame_pairs,frame_data,frame_examplesを結合"""
        frame_pairs['frame_element_pair'] = None
        frame_pairs['frame_example_pair'] = None
        rows_to_remove = []  # 削除する行のインデックスを記録
        
        for i, row in frame_pairs.iterrows():
            frames = row['frame_pair']
            frame_element_pair = []
            frame_example_pair = []
            for frame in frames:
                try:
                    frame_element_pair.append([f"{fe['name']}: {fe['definition']}" for fe in frame_data[frame_data['frame_name'] == frame["frame_name"]]['core_frame_elements'].values[0]])
                    
                    # 各フレームから最大3文をランダムに選択
                    verb = row['lex_unit_name'].split(".")[0]
                    
                    frame_examples_filtered = frame_examples[
                        (frame_examples['frame'] == frame["frame_name"]) & 
                        (frame_examples['verb'] == verb)
                    ]
                    # ひらがなが常用されるので変換
                    if "為る" in verb:
                        verb = verb.replace("為る", "する")
                    if len(frame_examples_filtered) > 0 and self.num_examples > 0:
                        # 設定された数の例文をランダムに選択
                        n_examples = min(self.num_examples, len(frame_examples_filtered))
                        selected_examples = frame_examples_filtered.sample(n_examples, random_state=self.seed + i)
                        
                        # 各例文で喚起語を**で囲む
                        highlighted_examples = []
                        for _, example_row in selected_examples.iterrows():
                            if self.language == "en":
                                sentence = example_row['sentence']
                                words = sentence.split(" ")
                                lu = words[example_row['lu_idx']]
                                highlighted_sentence = sentence.replace(lu, "**" + lu + "**")
                                highlighted_examples.append(highlighted_sentence)
                            elif self.language == "ja":
                                sentence = example_row['sentence']
                                start, end = example_row['target_span']
                                highlighted_sentence = sentence[:start] + "**" + sentence[start:end] + "**" + sentence[end:]
                                highlighted_examples.append(highlighted_sentence)
                        frame_example_pair.append(highlighted_examples)
                    else:
                        # 例文が見つからない場合は空のリスト
                        frame_example_pair.append([])
                        
                except Exception as e:
                    import pdb; pdb.set_trace()
                    logger.warning(f"フレーム {frame['frame_name']} の例文処理でエラー: {e}")
                    # 例外が発生した場合、この行を削除対象に追加
                    rows_to_remove.append(i)
                    break  # この行の処理を中断
            
            # 削除対象でない場合のみ値を設定
            if i not in rows_to_remove:
                frame_pairs.at[i, 'frame_element_pair'] = frame_element_pair
                frame_pairs.at[i, 'frame_example_pair'] = frame_example_pair
        
        # 削除対象の行をまとめて削除
        if rows_to_remove:
            frame_pairs = frame_pairs.drop(rows_to_remove).reset_index(drop=True)
            logger.info(f"フレームの情報統合時に例外が発生したため、{len(rows_to_remove)}行を削除しました")
        
        return frame_pairs
    
    def make_prompt(self, lu_name: str, frames: List[Dict[str, Any]], frame_element_pair: List[List[str]], frame_example_pair: List[List[str]], n: int = 2):
        """プロンプトを作成"""
        frame_names = [frame['frame_name'] for frame in frames]
        definitions = [frame['frame_definition'] for frame in frames]
        assert len(frame_names) == len(definitions) == len(frame_element_pair) == len(frame_example_pair)
        
        definitions_str_list = []
        verb = lu_name.split(".")[0]
        if "為る" in verb:
            verb = verb.replace("為る", "する")
        
        for frame, definition, frame_element, frame_examples in zip(frame_names, definitions, frame_element_pair, frame_example_pair):
            # 複数の例文を適切にフォーマット
            if isinstance(frame_examples, list) and len(frame_examples) > 0:
                examples_str = '\n'.join([f"- {example}" for example in frame_examples])
            else:
                examples_str = ""
            
            # フレーム定義を含めるかどうかで出力を変更
            if self.include_definition:
                definitions_str_list.append(DEFINITION.format(frame_name=frame, definition=definition, core_FEs=', '.join(frame_element), examples=examples_str))
            else:
                # 定義なしの場合はフレーム名と例文のみ
                frame_info = f"## {frame}"
                if examples_str:
                    frame_info += f"\nExamples: {examples_str}"
                definitions_str_list.append(frame_info)
        
        # 各メッセージをJinja2テンプレートとしてレンダリング
        messages = []
        try:
            for msg in self.prompt:
                template = Template(msg['content'])
                rendered_content = template.render(
                    verb=verb,
                    definitions_a=definitions_str_list[0],
                    definitions_b=definitions_str_list[1],
                    n=n,
                    frame_a=frame_names[0],
                    frame_b=frame_names[1]
                )
                
                messages.append({"role": msg['role'], "content": rendered_content})
            
            return messages
        except Exception as e:
            logger.error(f"プロンプトのレンダリングに失敗しました: {e}")
            raise e
    
    def generate_sentence_pairs_and_qa(self, data: pd.DataFrame, num_pairs: int = 2) -> pd.DataFrame:
        """LLMから結果を取得し、パースして平坦化したDataFrameを返す"""
        # プロンプト生成
        prompts = data.apply(
            lambda x: self.make_prompt(x['lex_unit_name'], x['frame_pair'], x['frame_element_pair'], x['frame_example_pair'], num_pairs), 
            axis=1
        ).tolist()
        logger.info(f"総プロンプト数: {len(prompts)}")
        
        # generate_batchを使用
        results = generate_batch(self.model, prompts, output_dir=self.output_dir, temperature=1.0, max_tokens=self.max_tokens)
        data['generated_response'] = results
        data.to_json(f"{self.output_dir}/intermediate_result.jsonl", orient="records", lines=True, force_ascii=False)
        logger.info(f"{self.output_dir}/intermediate_result.jsonl に経過を保存しました")
        # 結果をパースして平坦化
        output_data = []
        removed_duplicate_count = 0
        json_parse_error_count = 0
        no_result_count = 0
        for df_dict in data.to_dict(orient="records"):
            result = df_dict['generated_response']
            # 結果を処理
            if result is None:
                no_result_count += 1
                continue
            result = process_json_response(result) if isinstance(result, str) else result
            if result is None:
                json_parse_error_count += 1
                continue
            if not isinstance(result, list):
                result = [result]
            for result_item in result:
                frame_a, frame_b = list(result_item.get('sentence_pair', {}).keys())
                sentence_a = result_item.get('sentence_pair', {}).get(frame_a, '').replace("Sentence 1: ", "")
                sentence_b = result_item.get('sentence_pair', {}).get(frame_b, '').replace("Sentence 2: ", "")
                if sentence_a.strip() == sentence_b.strip():
                    removed_duplicate_count += 1
                    continue
                result_item.update({'lex_unit_name': df_dict.get('lex_unit_name', '')})
                result_item.update({'frame_pair': df_dict.get('frame_pair', [])})
                result_item.update({'frame_A': frame_a})
                result_item.update({'frame_B': frame_b})
                result_item.update({'sentence_A': sentence_a})
                result_item.update({'sentence_B': sentence_b})
                output_data.append(result_item)
        logger.info(f"JSONパースエラー: {json_parse_error_count}件")
        logger.info(f"結果がない: {no_result_count}件")
        logger.info(f"全くおなじ文をペアにしてしまっている{removed_duplicate_count}件")
        return pd.DataFrame(output_data)
    
    def _generate_qa_id(self, sentence_pair: Dict[str, Any]) -> str:
        """
        sentence_pairのハッシュ値を生成
        Args:
            sentence_pair: 文ペア
        
        Returns:
            一意なID（ハッシュ値の先頭16文字）
        """
        qa_str = json.dumps(sentence_pair, ensure_ascii=False, sort_keys=True)
        hash_obj = hashlib.sha256(qa_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def _add_original_qa_id_if_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameにoriginal_qa_idカラムがない場合は追加する
        
        Args:
            df: 中間結果のDataFrame
        
        Returns:
            original_qa_idが追加されたDataFrame
        """
        if 'original_qa_id' not in df.columns:
            logger.info("original_qa_idが見つかりません。生成します...")
            df['original_qa_id'] = df['sentence_pair'].apply(lambda x: self._generate_qa_id(x))
            logger.info(f"{len(df)}件のoriginal_qa_idを生成しました")
        return df
    
    def _add_question_format_for_ja(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        questionから、question、single-question、"two-choice-question、"four-choice-question"に幅出し
        """
        single_questions = []
        two_choice_questions = []
        four_choice_questions = []
        for qa in df['qa'].to_list():
            question = qa['question']
            if question.startswith("より"):
                question = question[2:]
            if question.endswith("なのはどちらの文ですか？"):
                single_question = '「' + question.replace("なのはどちらの文ですか？", "」と言えますか？")
                two_choice_question = '「' + question.replace("なのはどちらの文ですか？", "」のはどちらの文ですか？")
                four_choice_question = '「' + question.replace("なのはどちらの文ですか？", "」と言える文をすべて選んでください。")
            elif question.endswith("のはどちらの文ですか？"):
                single_question = '「' + question.replace("のはどちらの文ですか？", "」と言えますか？")
                two_choice_question = '「' + question.replace("のはどちらの文ですか？", "」のはどちらの文ですか？")
                four_choice_question = '「' + question.replace("のはどちらの文ですか？", "」と言える文をすべて選んでください。")
            else:
                logger.warning(f"{question}は対応していない問題形式です")
                single_question = None
                two_choice_question = None
                four_choice_question = None
            single_questions.append(single_question)
            two_choice_questions.append(two_choice_question)
            four_choice_questions.append(four_choice_question)
        df["single-question"] = single_questions
        df['four-choice-question'] = four_choice_questions
        df['two-choice-question'] = two_choice_questions
        return df

    def _format_qa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        original_qa_idの付与とresultの整形、問題形式の幅出し（ルールベース）
        """
        logger.info("追加の文形式の準備を開始...")
        
        # original_qa_idの追加
        df = self._add_original_qa_id_if_missing(df)
        
        # 日本語の場合のみ、問題形式の幅出し
        if self.language == "ja":
            df = self._add_question_format_for_ja(df)
            # single-questionがNoneでないものだけをフィルタリング
            df = pd.DataFrame(df[df['single-question'].notna()])
            logger.info(f"問題形式の幅出し後: {len(df)}件のデータ")
        elif self.language == "en":
            # TODO: Englishの問題形式の幅出し
            raise ValueError("English is not supported yet")
        
        # 必要な列のみを選択
        columns_to_select = ['original_qa_id', 'lex_unit_name', 'single-question', 'four-choice-question', 'two-choice-question', 'sentence_A', 'sentence_B', 'frame_A', 'frame_B','sentence_pair']
        # 存在する列のみを選択
        available_columns = [col for col in columns_to_select if col in df.columns]
        return pd.DataFrame(df[available_columns])
    
    def run_pipeline(self, data_root: str = "data", num_pairs: int = 2,
                    num: int | None = None):
        """パイプライン全体を実行（新ライブラリ使用）"""
        logger.info("FrameNet QA生成パイプラインを開始...")
        
        # データパスの設定（既存コードに合わせる）
        lu_data_path = f"{data_root}/{self.language}-framenet/lexical_units.jsonl"
        frame_data_path = f"{data_root}/{self.language}-framenet/frames.jsonl"
        frame_examples_data_path = f"{data_root}/{self.language}-framenet/exemplars.jsonl"
        # 1. lexical unitsデータの読み込み
        lexical_units = self.load_and_filter_lexical_units(lu_data_path)
        frames = load_jsonl(frame_data_path, "frames")
        frame_examples = load_jsonl(frame_examples_data_path, "frame_examples")

        # 2. フレームペアの生成
        frame_pairs = self.choice_all_frame_pairs(lexical_units)
        logger.info(f"フレームペア数: {len(frame_pairs)}")
        frame_pairs = self.concat_info(frame_pairs, frames,frame_examples)
        if num is not None and num > 0 and num < frame_pairs.shape[0]:
            frame_pairs = frame_pairs.sample(num, random_state=self.seed)

        # 3. 文ペアとQAの生成 中間状態を保存
        qa_data = self.generate_sentence_pairs_and_qa(frame_pairs, num_pairs)
        logger.info(f"生成されたQA数: {len(qa_data)}")
        
        # 5. 整形（original_qa_idの付与、問題形式の幅出し）
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        final_qa_data = self._format_qa(qa_data)
        
        # 最終的なformatted_qa.jsonlを保存
        final_qa_data.to_json(f"{self.output_dir}/qa.jsonl", 
                            orient="records", lines=True, force_ascii=False)
        logger.info(f"パイプライン完了: 最終的に{len(final_qa_data)}件のQAを生成しました")
        logger.info(f"結果を {self.output_dir}/qa.jsonl に保存しました")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='FrameNet QA生成パイプライン')
    parser.add_argument('--data_root', type=str, default='data', help='データのルートディレクトリ')
    parser.add_argument('--prompt_file', type=str, default=None, help='利用するプロンプトファイルへのパス')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='使用するLLM')
    parser.add_argument('--language', type=str, default='en', help='利用対象言語')
    parser.add_argument('--max_tokens', type=int, default=7000, help='生成する最大トークン数')
    parser.add_argument('--num_pairs', type=int, default=2, help='単一の入力から生成する問題数')
    parser.add_argument('--num', type=int, default=None, help='利用するフレームペアの数')
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード')
    
    # 実験用パラメータ
    parser.add_argument('--no_definition', action='store_true', help='フレーム定義を含めない')
    parser.add_argument('--no_thinking', action='store_true', help='thinking_processを出力させない')
    parser.add_argument('--num_examples', type=int, default=3, help='フレームごとの最大例文数')
    
    args = parser.parse_args()
    
    # パイプラインの実行
    pipeline = FrameQAPipeline(
        model=args.model,
        language=args.language,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_file=args.prompt_file,
        include_definition=not args.no_definition,
        include_thinking=not args.no_thinking,
        num_examples=args.num_examples
    )
    
    pipeline.run_pipeline(
        data_root=args.data_root,
        num_pairs=args.num_pairs,
        num=args.num
    )


if __name__ == "__main__":
    main() 