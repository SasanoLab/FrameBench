# -*- coding: utf-8 -*-
import random
import json
import yaml
import os
from typing import Dict, Optional
from datetime import datetime

class BaseFrameQAApp:
    """Frame QA アプリケーションの共通処理を提供するベースクラス"""
    
    def __init__(self, data_file: str, criteria_file: str, auth_config_file: str = None, user_assignment_file: str = None):
        self.data_file = data_file
        self.criteria_file = criteria_file
        self.auth_config_file = auth_config_file or "src/auth_config.json"
        self.user_assignment_file = user_assignment_file or "src/user_assignment.json"
        self.current_index = 0
        self.original_data = []  # 元の全データを保持
        self.data = []  # 現在のユーザー用のフィルタリングされたデータ
        self.criteria = {}
        self.annotations = {}
        self.current_user = None
        self.auth_config = {}
        self.user_assignments = {}
        self.user_data_range = None
        self.annotation_file = None  # ユーザー別アノテーションファイル名を保存
        self.question_format = "four_choices"  # デフォルトは4択問題
        
        # データと評価基準を読み込み
        self.load_data()
        self.load_criteria()
        self.load_auth_config()
        self.load_user_assignments()
        
        # セッション開始時に既存のアノテーションデータをクリア
        self.clear_session_data()
    
    def change_dataset(self, new_data_file: str):
        """データセットを動的に変更する"""
        if not os.path.exists(new_data_file):
            return False, f"ファイルが存在しません: {new_data_file}"
        
        # 現在のアノテーションを保存
        self.save_current_annotations()
        
        # 新しいデータファイルを設定
        self.data_file = new_data_file
        
        # データを再読み込み
        try:
            self.load_data()
            # インデックスをリセット
            self.current_index = 0
            # ユーザー割り当てを再適用
            if self.current_user:
                self.apply_user_assignment(self.current_user)
                # 一時保存されたアノテーションを読み込み
                self.load_temp_annotations()
            return True, f"データセットを変更しました: {os.path.basename(new_data_file)}"
        except Exception as e:
            return False, f"データセットの変更に失敗しました: {str(e)}"
    
    def save_current_annotations(self):
        """現在のアノテーションを一時保存"""
        if self.current_user and self.annotations:
            # ユーザー別のアノテーションファイルに保存
            filename = f"temp_annotations_{self.current_user}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.annotations, f, ensure_ascii=False, indent=2)
                print(f"アノテーションを一時保存しました: {filename}")
            except Exception as e:
                print(f"アノテーションの一時保存に失敗: {e}")
    
    def load_temp_annotations(self):
        """一時保存されたアノテーションを読み込み"""
        if self.current_user:
            filename = f"temp_annotations_{self.current_user}.json"
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        self.annotations = json.load(f)
                    print(f"一時保存されたアノテーションを読み込みました: {filename}")
                    return True
            except Exception as e:
                print(f"一時保存されたアノテーションの読み込みに失敗: {e}")
        return False
        
    def load_data(self):
        """JSONLファイルから質問データを読み込む"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.original_data = [json.loads(line.strip()) for line in f if line.strip()]
                # 各データアイテムに一意のIDを追加
                for i, item in enumerate(self.original_data):
                    item['data_id'] = f"data_{i:04d}"  # data_0000, data_0001, ... の形式
                self.data = self.original_data.copy()  # 初期状態では全データを使用
            print(f"データを読み込みました: {len(self.original_data)}件")
            # データを展開（問題形式に応じて複数の問題に分割）
            self._expand_data()
        except Exception as e:
            print(f"データの読み込みエラー: {e}")
            self.original_data = []
            self.data = []
    
    def _get_shuffle_seed(self) -> int:
        """シャッフル用のシード値を生成（再現性のため）"""
        # データファイル名、問題形式、ユーザーIDからシードを生成
        import hashlib
        
        # シード生成に使用する要素
        seed_components = [
            os.path.basename(self.data_file),  # データファイル名
        ]
        
        # 文字列を結合してハッシュ化
        seed_string = "_".join(seed_components)
        hash_value = hashlib.md5(seed_string.encode()).hexdigest()
        
        # ハッシュ値を整数に変換（最初の8文字を16進数として解釈）
        seed = int(hash_value[:8], 16)
        
        return seed
    
    def _expand_data(self):
        """問題形式に応じてデータを展開"""
        # 展開前のデータを一時保存（フィルタ済みの元データ）
        unexpanded_data = self.data.copy()
        expanded = []
        
        for original_index, item in enumerate(unexpanded_data):
            # 問題形式に応じた展開情報を取得
            expanded_items = self._expand_single_item(item, original_index)
            expanded.extend(expanded_items)
        
        # シード値を設定してシャッフル（再現性のため）
        seed = self._get_shuffle_seed()
        random.seed(seed)
        random.shuffle(expanded)
        
        self.data = expanded
        print(f"データを展開しました: {len(unexpanded_data)}件 → {len(self.data)}件の問題（シャッフル済み、シード: {seed}）")
    
    def _expand_single_item(self, item, original_index):
        """1つのアイテムを複数の問題に展開"""
        # 問題形式に応じた表示情報を取得
        if self.question_format == "four_choices":
            display_info = self._get_four_choices_display_info(item, original_index)
        elif self.question_format == "two_choices":
            display_info = self._get_two_choices_display_info(item)
        elif self.question_format == "single":
            display_info = self._get_single_display_info(item)
        elif self.question_format == "all_sentence":
            display_info = self._get_all_sentence_display_info(item)
        else:
            display_info = self._get_four_choices_display_info(item, original_index)
        
        lex_units, questions, sentences_pairs, frames_pairs, answers = display_info
        
        # 有効な問題がない場合は空のリストを返す
        if not sentences_pairs:
            return []
        
        # 各問題を個別のアイテムとして生成
        expanded_items = []
        num_problems = len(sentences_pairs)
        for i in range(num_problems):
            expanded_item = {
                'data_id': item['data_id'],
                'original_index': original_index,
                'sub_index': i,
                'lex_unit_name': lex_units[i],
                'question': questions[i],
                'sentences': sentences_pairs[i],
                'frames': frames_pairs[i],
                'answer': answers[i],
                'question_format': self.question_format,
                # 元のアイテムの情報も保持
                'original_item': item
            }
            # original_qa_idがあれば直接含める（後で参照しやすくするため）
            if 'original_qa_id' in item:
                expanded_item['original_qa_id'] = item['original_qa_id']
            expanded_items.append(expanded_item)
        
        return expanded_items
    
    def _get_four_choices_display_info(self, item, original_index):
        """4択問題の表示情報を取得
        
        基本的には異なるフレームのペアのみを使用し、
        極稀に（10%の確率で）同じフレームのペア[A, A']と[B, B']も含める
        空文字列の文を含むペアは除外される
        """
        sentenceA = item.get("sentence_A", "")
        sentenceB = item.get("sentence_B", "")
        sentenceA_prime = item.get("sentence_A_prime", "")
        sentenceB_prime = item.get("sentence_B_prime", "")
        frameA = item["frame_A"]
        frameB = item["frame_B"]
        question = item["four-choice-question"]
        
        # データIDとoriginal_indexからシードを生成（再現性のため）
        import hashlib
        seed_string = f"{item['data_id']}_{original_index}_four_choices"
        hash_value = hashlib.md5(seed_string.encode()).hexdigest()
        item_seed = int(hash_value[:8], 16)
        
        # このデータ項目専用の乱数生成器を作成
        rng = random.Random(item_seed)
        
        # 5%の確率で同じフレームのペアを含める
        include_both_sentence_pairs = rng.random() < 0.1
        include_neither_sentence_pairs = rng.random() < 0.1
        
        if include_both_sentence_pairs:
            # 全てのペアを含める（従来通り）
            all_pairs = [
                ([sentenceA, sentenceB], [frameA, frameB], 'Sentence 1'),
                ([sentenceA, sentenceA_prime], [frameA, frameA], 'Both sentences'),
                ([sentenceB_prime, sentenceA_prime], [frameB, frameA], 'Sentence 2')
            ]
        elif include_neither_sentence_pairs:
            all_pairs = [
                ([sentenceA, sentenceB], [frameA, frameB], 'Sentence 1'),
                ([sentenceB_prime, sentenceA_prime], [frameB, frameA], 'Sentence 2'),
                ([sentenceB, sentenceB_prime], [frameB, frameB], 'Neither sentence'),
            ]
        else:
            # 異なるフレームのペアのみを使用
            all_pairs = [
                ([sentenceA, sentenceB], [frameA, frameB], 'Sentence 1'),
                ([sentenceB_prime, sentenceA_prime], [frameB, frameA], 'Sentence 2')
            ]
        
        # 空文字列を含まないペアだけをフィルタリング
        valid_pairs = [
            (sents, frames, ans) for sents, frames, ans in all_pairs
            if all(s.strip() for s in sents)  # 全ての文が空でないことを確認
        ]
        
        # 有効なペアがない場合は空のリストを返す
        if not valid_pairs:
            return ([], [], [], [], [])
        
        # シャッフル
        rng.shuffle(valid_pairs)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*valid_pairs)
        
        num_problems = len(valid_pairs)
        return ([item['lex_unit_name']] * num_problems, [question] * num_problems, 
                list(sentences_pairs_shuffled), list(frames_pairs_shuffled), list(answers_shuffled))
    
    def _get_two_choices_display_info(self, item):
        """2択問題の表示情報を取得
        空文字列の文を含むペアは除外される
        """
        sentenceA = item.get("sentence_A", "")
        sentenceB = item.get("sentence_B", "")
        sentenceA_prime = item.get("sentence_A_prime", "")
        sentenceB_prime = item.get("sentence_B_prime", "")
        frameA = item["frame_A"]
        frameB = item["frame_B"]
        question = item["two-choice-question"]
        
        all_pairs = [
            ([sentenceA, sentenceB], [frameA, frameB], 'Sentence 1'),
            ([sentenceB_prime, sentenceA_prime], [frameB, frameA], 'Sentence 2')
        ]
        
        # 空文字列を含まないペアだけをフィルタリング
        valid_pairs = [
            (sents, frames, ans) for sents, frames, ans in all_pairs
            if all(s.strip() for s in sents)  # 全ての文が空でないことを確認
        ]
        
        # 有効なペアがない場合は空のリストを返す
        if not valid_pairs:
            return ([], [], [], [], [])
        
        # シャッフル
        random.shuffle(valid_pairs)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*valid_pairs)
        
        num_problems = len(valid_pairs)
        return ([item['lex_unit_name']] * num_problems, [question] * num_problems,
                list(sentences_pairs_shuffled), list(frames_pairs_shuffled), list(answers_shuffled))
    
    def _get_single_display_info(self, item):
        """単発問題の表示情報を取得
        空文字列の文は除外される
        """
        sentenceA = item.get("sentence_A", "")
        sentenceB = item.get("sentence_B", "")
        sentenceA_prime = item.get("sentence_A_prime", "")
        sentenceB_prime = item.get("sentence_B_prime", "")
        frameA = item["frame_A"]
        frameB = item["frame_B"]
        question = item["single-question"]
        
        all_pairs = [
            ([sentenceA], [frameA], 'Yes'),
            ([sentenceB], [frameB], 'No'),
            ([sentenceA_prime], [frameA], 'Yes'),
            ([sentenceB_prime], [frameB], 'No')
        ]
        
        # 空文字列を含まないペアだけをフィルタリング
        valid_pairs = [
            (sents, frames, ans) for sents, frames, ans in all_pairs
            if all(s.strip() for s in sents)  # 全ての文が空でないことを確認
        ]
        
        # 有効なペアがない場合は空のリストを返す
        if not valid_pairs:
            return ([], [], [], [], [])
        
        # シャッフル
        random.shuffle(valid_pairs)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*valid_pairs)
        
        num_problems = len(valid_pairs)
        return ([item['lex_unit_name']] * num_problems, [question] * num_problems,
                list(sentences_pairs_shuffled), list(frames_pairs_shuffled), list(answers_shuffled))
    
    def _get_all_sentence_display_info(self, item):
        """すべての文を表示する問題の表示情報を取得
        少なくとも1つの文が存在すれば問題を生成する
        """
        sentenceA = item.get("sentence_A", "")
        sentenceB = item.get("sentence_B", "")
        sentenceA_prime = item.get("sentence_A_prime", "")
        sentenceB_prime = item.get("sentence_B_prime", "")
        frameA = item["frame_A"]
        frameB = item["frame_B"]
        question = item["single-question"]
        
        # 空でない文だけを集める
        all_sentences = [sentenceA, sentenceA_prime, sentenceB, sentenceB_prime]
        all_frames = [frameA, frameA, frameB, frameB]
        
        valid_sentences = []
        valid_frames = []
        for sent, frame in zip(all_sentences, all_frames):
            if sent.strip():  # 空でない文のみ
                valid_sentences.append(sent)
                valid_frames.append(frame)
        
        # 有効な文がない場合は空のリストを返す
        if not valid_sentences:
            return ([], [], [], [], [])
        
        sentences_pairs = [valid_sentences]
        frames_pairs = [valid_frames]
        answers = [frameA]
        
        return ([item['lex_unit_name']], [question], sentences_pairs, frames_pairs, answers)
    
    def load_criteria(self):
        """YAMLファイルから評価基準を読み込む"""
        try:
            with open(self.criteria_file, 'r', encoding='utf-8') as f:
                self.criteria = yaml.safe_load(f)
            print("評価基準を読み込みました")
        except Exception as e:
            print(f"評価基準の読み込みエラー: {e}")
            self.criteria = {}
    
    
    def load_auth_config(self):
        """認証設定を読み込む（環境変数優先、ファイル次点、デフォルト最後）"""
        # 環境変数から認証情報を読み込み
        auth_from_env = os.getenv('FRAME_QA_AUTH_USERS')
        if auth_from_env:
            try:
                self.auth_config = json.loads(auth_from_env)
                print(f"環境変数から認証設定を読み込みました: {len(self.auth_config)}件のユーザー")
                return
            except Exception as e:
                print(f"環境変数の認証設定の解析エラー: {e}")
        
        # ファイルから認証情報を読み込み（ローカル開発用）
        try:
            if os.path.exists(self.auth_config_file):
                with open(self.auth_config_file, 'r', encoding='utf-8') as f:
                    auth_data = json.load(f)
                    
                    self.auth_config = auth_data
                    
                print(f"認証設定ファイルを読み込みました: {len(self.auth_config)}件のユーザー")
                return
        except Exception as e:
            print(f"認証設定ファイルの読み込みエラー: {e}")
        
        # デフォルトの認証情報を設定
        self.auth_config = {}
        print("デフォルトの認証情報を使用します")
    
    def load_user_assignments(self):
        """ユーザー割り当て設定を読み込む（環境変数優先、ファイル次点、デフォルト最後）"""
        # 環境変数からユーザー割り当て情報を読み込み
        assignment_from_env = os.getenv('FRAME_QA_USER_ASSIGNMENTS')
        if assignment_from_env:
            try:
                assignment_data = json.loads(assignment_from_env)
                self.user_assignments = assignment_data
                print(f"環境変数からユーザー割り当て設定を読み込みました: {len(self.user_assignments)}件のユーザー")
                return
            except Exception as e:
                print(f"環境変数のユーザー割り当て設定の解析エラー: {e}")
        
        # ファイルからユーザー割り当て情報を読み込み（ローカル開発用）
        try:
            if os.path.exists(self.user_assignment_file):
                with open(self.user_assignment_file, 'r', encoding='utf-8') as f:
                    assignment_data = json.load(f)
                    
                    # user_assignmentsキーがある場合はその中身を使用
                    if "user_assignments" in assignment_data and isinstance(assignment_data["user_assignments"], dict):
                        self.user_assignments = assignment_data["user_assignments"]
                    else:
                        # user_assignmentsキーがない場合は、ルートレベルのキーを直接使用
                        self.user_assignments = assignment_data
                    
                print(f"ユーザー割り当て設定ファイルを読み込みました: {len(self.user_assignments)}件のユーザー")
                return
        except Exception as e:
            print(f"ユーザー割り当て設定ファイルの読み込みエラー: {e}")
            raise e
    
    def apply_user_assignment(self, username: str):
        """ユーザー割り当てを適用してデータをフィルタリング"""
        if username in self.user_assignments:
            user_config = self.user_assignments[username]
            data_range = user_config.get("data_range", {})
            start = data_range.get("start", 0)
            end = data_range.get("end", len(self.original_data))
            
            # データ範囲を適用
            self.data = self.original_data[start:end]
            self.user_data_range = {"start": start, "end": end}
            self.current_index = 0  # インデックスをリセット
            
            print(f"ユーザー {username} のデータ範囲を適用: {start}-{end} ({len(self.data)}件)")
        else:
            # ユーザー割り当てがない場合は全データを使用
            self.data = self.original_data.copy()
            self.user_data_range = None
            self.current_index = 0
            print(f"ユーザー {username} の割り当てが見つからないため、全データを使用: {len(self.data)}件")
        
        # データを展開
        self._expand_data()
    
    def get_current_item(self) -> Optional[Dict]:
        """現在のアイテムを取得"""
        if 0 <= self.current_index < len(self.data):
            return self.data[self.current_index]
        return None
    
    def next_item(self):
        """次のアイテムに移動"""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            return self.update_display()
        return "最後の項目です"
    
    def prev_item(self):
        """前のアイテムに移動"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.update_display()
        return "最初の項目です"
    
    
    def get_four_choices_question_display_info(self):
        """基本的な表示情報を取得"""
        item = self.get_current_item()
        if item is None:
            return "データが見つかりません", "", "", "", "", ""
        
        # 文のペアを取得
        sentenceA = item["sentence_A"]
        sentenceB = item["sentence_B"]
        sentenceA_prime = item["sentence_A_prime"]
        sentenceB_prime = item["sentence_B_prime"]

        frameA = item["frame_A"]
        frameB = item["frame_B"]

        question = item["four-choice-question"]

        sentences_pairs = [[sentenceA, sentenceB],
                            [sentenceA, sentenceA_prime],
                            [sentenceB, sentenceB_prime],
                            [sentenceB_prime, sentenceA_prime]]
        frames_pairs = [[frameA, frameB],
                        [frameA, frameA],
                        [frameB, frameB],
                        [frameB, frameA]]
        answers = ['Sentence 1','Both sentences', 'Neither sentence', 'Sentence 2']
        paired_list = list(zip(sentences_pairs, frames_pairs, answers))
        random.shuffle(paired_list)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*paired_list)
        sentences_pairs_shuffled = list(sentences_pairs_shuffled)
        frames_pairs_shuffled = list(frames_pairs_shuffled)
        answers_shuffled = list(answers_shuffled)
        return ([item['lex_unit_name']] * 4, [question] * 4, sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled)
    
    def get_two_choices_question_display_info(self):
        """基本的な表示情報を取得"""
        item = self.get_current_item()
        if item is None:
            return "データが見つかりません", "", "", "", "", ""
        
        # 文のペアを取得
        sentenceA = item["sentence_A"]
        sentenceB = item["sentence_B"]
        sentenceA_prime = item["sentence_A_prime"]
        sentenceB_prime = item["sentence_B_prime"]

        frameA = item["frame_A"]
        frameB = item["frame_B"]

        question = item["two-choice-question"]

        sentences_pairs = [[sentenceA, sentenceB],
                            [sentenceB_prime, sentenceA_prime]]
        frames_pairs = [[frameA, frameB],
                        [frameB, frameA]]
        answers = ['Sentence 1','Sentence 2']
        paired_list = list(zip(sentences_pairs, frames_pairs, answers))
        random.shuffle(paired_list)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*paired_list)
        sentences_pairs_shuffled = list(sentences_pairs_shuffled)
        frames_pairs_shuffled = list(frames_pairs_shuffled)
        answers_shuffled = list(answers_shuffled)
        return ([item['lex_unit_name']] * 2, [question] * 2, sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled)

    def get_single_question_display_info(self):
        """基本的な表示情報を取得"""
        item = self.get_current_item()
        if item is None:
            return "データが見つかりません", "", "", "", "", ""
        
        # 文のペアを取得
        sentenceA = item["sentence_A"]
        sentenceB = item["sentence_B"]
        sentenceA_prime = item["sentence_A_prime"]
        sentenceB_prime = item["sentence_B_prime"]

        frameA = item["frame_A"]
        frameB = item["frame_B"]

        question = item["single-question"]

        sentences_pairs = [[sentenceA],
                            [sentenceB],
                            [sentenceA_prime],
                            [sentenceB_prime]]
        frames_pairs = [[frameA],
                        [frameB],
                        [frameA],
                        [frameB]]
        answers = ['Yes','No','Yes','No']
        paired_list = list(zip(sentences_pairs, frames_pairs, answers))
        random.shuffle(paired_list)
        sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled = zip(*paired_list)
        sentences_pairs_shuffled = list(sentences_pairs_shuffled)
        frames_pairs_shuffled = list(frames_pairs_shuffled)
        answers_shuffled = list(answers_shuffled)
        return ([item['lex_unit_name']] * 4, [question] * 4, sentences_pairs_shuffled, frames_pairs_shuffled, answers_shuffled)

    def get_all_sentence_question_display_info(self):
        """すべての文を表示する問題の表示情報を取得"""
        item = self.get_current_item()
        if item is None:
            return "データが見つかりません", "", "", "", "", ""
        
        # 文のペアを取得
        sentenceA = item["sentence_A"]
        sentenceB = item["sentence_B"]
        sentenceA_prime = item["sentence_A_prime"]
        sentenceB_prime = item["sentence_B_prime"]

        frameA = item["frame_A"]
        frameB = item["frame_B"]

        question = item["single-question"]

        sentences_pairs = [[sentenceA, sentenceB, sentenceA_prime, sentenceB_prime]]
        frames_pairs = [[frameA, frameB, frameA, frameB]]
        answers = ['-']
        
        return ([item['lex_unit_name']], [question], sentences_pairs, frames_pairs, answers)

    def get_item_info(self):
        """現在の項目情報を取得"""
        item = self.get_current_item()
        if item is None:
            return "データが見つかりません"
        
        # 展開後の通し番号のみ表示
        item_info = f"問題 {self.current_index + 1}/{len(self.data)}"
        
        return item_info
    
    def set_question_format(self, format_type: str):
        """問題形式を設定"""
        if format_type in ["four_choices", "two_choices", "single", "all_sentence"]:
            self.question_format = format_type
            # 問題形式が変わったらデータを再展開
            if self.is_authenticated():
                # ユーザー割り当てを再適用してから展開
                self.apply_user_assignment(self.current_user)
            else:
                # ログイン前は元データから展開
                self.data = self.original_data.copy()
                self._expand_data()
            self.current_index = 0  # インデックスをリセット
            return True, f"問題形式を{self.get_format_name()}に変更しました"
        return False, "無効な問題形式です"
    
    def get_question_format(self) -> str:
        """現在の問題形式を取得"""
        return self.question_format
    
    def get_format_name(self) -> str:
        """問題形式の表示名を取得"""
        format_names = {
            "four_choices": "4択問題",
            "two_choices": "2択問題",
            "single": "単発問題",
            "all_sentence": "全文表示"
        }
        return format_names.get(self.question_format, "不明")
    
    def get_question_display_info(self):
        """現在の問題形式に応じた表示情報を取得"""
        if self.question_format == "four_choices":
            return self.get_four_choices_question_display_info()
        elif self.question_format == "two_choices":
            return self.get_two_choices_question_display_info()
        elif self.question_format == "single":
            return self.get_single_question_display_info()
        elif self.question_format == "all_sentence":
            return self.get_all_sentence_question_display_info()
        else:
            return self.get_four_choices_question_display_info()  # デフォルト
    
    def get_basic_display_info(self):
        """基本的な表示情報を取得"""
        # 問題形式に応じた表示情報を取得
        return self.get_question_display_info()

    
    def update_display(self):
        """表示を更新（サブクラスでオーバーライド可能）"""
        return self.get_basic_display_info()

    
    def get_annotation_key(self):
        """アノテーション保存用のキーを生成（データID + サブインデックス + ユーザーID）"""
        item = self.get_current_item()
        if item and 'data_id' in item:
            # 展開されたアイテムの場合、サブインデックスも含める
            if 'sub_index' in item:
                return f"{item['data_id']}_sub{item['sub_index']}_{self.current_user}"
            else:
                return f"{item['data_id']}_{self.current_user}"
        # フォールバック: データIDがない場合の従来の方式
        return f"{self.current_user}_{self.current_index}"
    
    def save_annotations_to_file(self, output_file: str = "annotations.json"):
        """アノテーションをユーザー別ファイルに保存"""
        try:
            # ファイル名にユーザーIDを含める
            base_name = output_file.replace('.json', '')
            user_file = f"{base_name}_{self.current_user}.json"
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
            return f"アノテーションを保存しました (項目 {self.current_index + 1}/{len(self.data)})"
        except Exception as e:
            return f"保存エラー: {e}"
    
    def create_annotation_with_metadata(self, annotation_data: dict) -> dict:
        """アノテーションデータにメタデータを追加"""
        item = self.get_current_item()
        if item is None:
            return annotation_data
        
        # 元のアイテムを取得（展開されたアイテムの場合はoriginal_itemから、そうでなければ直接）
        original_item = item.get('original_item', item)
        
        # メタデータを追加
        metadata = {
            "data_id": item.get('data_id', f"unknown_{self.current_index}"),
            "user_id": self.get_current_user(),
            "timestamp": datetime.now().isoformat(),
            "current_index": self.current_index,
            "lex_unit_name": item.get("lex_unit_name", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "question_format": item.get("question_format", self.question_format)
        }
        
        # 展開されたアイテムの場合、追加情報
        if 'sub_index' in item:
            metadata["original_index"] = item.get('original_index', 0)
            metadata["sub_index"] = item.get('sub_index', 0)
        
        # 元のアイテムの全てのキーをメタデータに追加（展開時に追加されたキーを除く）
        exclude_keys = {'data_id', 'original_index', 'sub_index', 'lex_unit_name', 
                        'question', 'sentences', 'frames', 'answer', 'question_format', 
                        'original_item'}  # 展開時に追加されたキーは除外
        for key, value in original_item.items():
            if key not in exclude_keys:
                metadata[key] = value
        
        # sentence_pairが元のアイテムに含まれていない場合、sentencesとframesから構築
        if 'sentence_pair' not in metadata:
            sentences = item.get('sentences', [])
            frames = item.get('frames', [])
            if sentences and frames and len(sentences) >= 2 and len(frames) >= 2:
                # 最初の2つの文とフレームからsentence_pairを構築
                sentence_pair = {
                    frames[0]: sentences[0],
                    frames[1]: sentences[1]
                }
                metadata["sentence_pair"] = sentence_pair
        
        # アノテーションデータとメタデータを結合
        return {**metadata, **annotation_data}
    
    def load_annotations_from_file(self, input_file: str = "annotations.json"):
        """既存のアノテーションファイルをユーザー別に読み込み"""
        # アノテーションファイル名を保存（ログイン時の再読み込みに使用）
        self.annotation_file = input_file
        
        try:
            # ユーザーが認証されている場合はユーザー別ファイルを読み込む
            if self.current_user:
                base_name = input_file.replace('.json', '')
                user_file = f"{base_name}_{self.current_user}.json"
                
                if os.path.exists(user_file):
                    with open(user_file, 'r', encoding='utf-8') as f:
                        self.annotations = json.load(f)
                    print(f"既存のアノテーションを読み込みました ({user_file}): {len(self.annotations)}件")
                else:
                    print(f"既存のアノテーションファイルが見つかりません: {user_file}")
                    self.annotations = {}
            else:
                # ログイン前は空の状態
                print("ログイン前のため、アノテーションは読み込まれていません")
                self.annotations = {}
        except Exception as e:
            print(f"アノテーションの読み込みエラー: {e}")
            self.annotations = {}
    
    def authenticate(self, username: str, password: str) -> bool:
        """ユーザー認証"""
        if username in self.auth_config and self.auth_config[username] == password:
            self.current_user = username
            # ユーザーのデータ範囲を適用
            self.apply_user_assignment(username)
            # ユーザー別のアノテーションファイルを読み込み
            if self.annotation_file:
                self.load_annotations_from_file(self.annotation_file)
            print(f"ユーザー {username} でログインしました")
            return True
        return False
    
    
    def logout(self):
        """ログアウト"""
        if self.current_user:
            print(f"ユーザー {self.current_user} がログアウトしました")
            self.current_user = None
    
    def is_authenticated(self) -> bool:
        """認証状態を確認"""
        return self.current_user is not None
    
    def get_current_user(self) -> Optional[str]:
        """現在のユーザーIDを取得"""
        return self.current_user
    
    def get_dataset_name(self) -> str:
        """現在のデータセット名を取得"""
        if not self.data_file:
            return "unknown"
        
        # ファイルパスからデータセット名を生成
        import os
        filename = os.path.basename(self.data_file)
        # 拡張子を除去
        dataset_name = os.path.splitext(filename)[0]
        
        # パスからより詳細な情報を取得
        path_parts = self.data_file.split(os.sep)
        if len(path_parts) >= 3:
            # 例: data/ja/gpt-5/high_quality_qa.jsonl -> gpt-5_high_quality_qa
            dataset_name = f"{path_parts[-2]}_{dataset_name}"
        
        return dataset_name
    
    def get_data_count(self) -> int:
        """現在のデータ数を取得"""
        return len(self.data)
    
    def get_annotations_for_download(self) -> str:
        """ダウンロード用のアノテーションデータを取得"""
        if not self.annotations:
            return "アノテーションデータがありません"
        
        # 現在のユーザーのアノテーションのみをフィルタリング
        user_annotations = {}
        for key, annotation in self.annotations.items():
            if annotation.get("user_id") == self.current_user:
                user_annotations[key] = annotation
        
        if not user_annotations:
            return "現在のユーザーのアノテーションデータがありません"
        
        # データIDでソートして整理
        sorted_annotations = {}
        for key in sorted(user_annotations.keys()):
            sorted_annotations[key] = user_annotations[key]
        
        return json.dumps(sorted_annotations, ensure_ascii=False, indent=2)
    
    def get_annotations_summary(self) -> str:
        """アノテーションのサマリー情報を取得"""
        if not self.annotations:
            return "アノテーションデータがありません"
        
        # 現在のユーザーのアノテーションのみをフィルタリング
        user_annotations = {}
        for key, annotation in self.annotations.items():
            if annotation.get("user_id") == self.current_user:
                user_annotations[key] = annotation
        
        if not user_annotations:
            return "現在のユーザーのアノテーションデータがありません"
        
        # サマリー情報を作成
        summary = {
            "user_id": self.current_user,
            "total_annotations": len(user_annotations),
            "data_range": self.user_data_range,
            "annotations": []
        }
        
        # 各アノテーションの基本情報を追加
        for key, annotation in user_annotations.items():
            summary["annotations"].append({
                "annotation_key": key,
                "data_id": annotation.get("data_id", "unknown"),
                "timestamp": annotation.get("timestamp", ""),
                "lex_unit_name": annotation.get("lex_unit_name", ""),
                "has_correction": "corrected_question" in annotation,
                "has_evaluation": "evaluations" in annotation,
                "original_qa_id": annotation.get("original_qa_id", "")
            })
        
        return json.dumps(summary, ensure_ascii=False, indent=2)
    
    def clear_session_data(self):
        """セッションデータをクリア"""
        self.annotations = {}
        self.current_index = 0
        print("セッションデータをクリアしました")
    
    def clear_user_data(self, user_id: str = None):
        """特定ユーザーのデータをクリア"""
        if user_id is None:
            user_id = self.current_user
        
        if user_id:
            # 特定ユーザーのアノテーションを削除
            keys_to_remove = []
            for key, annotation in self.annotations.items():
                if annotation.get("user_id") == user_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.annotations[key]
            
            print(f"ユーザー {user_id} のデータをクリアしました")
    
    def get_session_info(self) -> str:
        """セッション情報を取得"""
        if not self.current_user:
            return "未ログイン"
        
        user_annotations = {}
        for key, annotation in self.annotations.items():
            if annotation.get("user_id") == self.current_user:
                user_annotations[key] = annotation
        
        # ユーザー割り当て情報を追加
        if self.user_data_range:
            start = self.user_data_range["start"]
            end = self.user_data_range["end"]
            return f"ユーザー: {self.current_user} (データ範囲: {start}-{end}), アノテーション数: {len(user_annotations)}/{len(self.data)}"
        else:
            return f"ユーザー: {self.current_user}, アノテーション数: {len(user_annotations)}/{len(self.data)}"
