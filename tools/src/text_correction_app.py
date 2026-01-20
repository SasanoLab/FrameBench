import gradio as gr
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from .common import BaseFrameQAApp

class TextCorrectionApp(BaseFrameQAApp):
    """テキスト修正専用のFrame QAアプリケーション"""
    
    def __init__(self, data_file: str, criteria_file: str):
        super().__init__(data_file, criteria_file, user_assignment_file="src/user_assignment.json")
        # 既存のアノテーションを読み込み
        self.load_annotations_from_file("text_corrections.json")
    
    def save_text_correction(self, corrected_question: str, comments: str):
        """テキスト修正結果を保存"""
        if not self.is_authenticated():
            return "エラー: ログインが必要です"
        
        item = self.get_current_item()
        if item is None:
            return "エラー: データが見つかりません"
        
        # 修正データを作成
        correction_data = {
            "corrected_question": corrected_question,
            "comments": comments,
            "annotation_type": "text_correction"
        }
        
        # メタデータを追加してアノテーションを作成
        annotation = self.create_annotation_with_metadata(correction_data)
        
        self.annotations[self.get_annotation_key()] = annotation
        
        # ファイルに保存
        return self.save_annotations_to_file("text_corrections.json")
    
    def update_display(self):
        """表示を更新（テキスト修正専用）- 展開されたデータを使用"""
        item = self.get_current_item()
        if item is None:
            return ("データが見つかりません", "", "", "", "", "", False, "", "")
        
        # 展開されたアイテムから情報を取得
        current_info = self.get_item_info()
        verb_info = f"フレーム喚起語: {item.get('lex_unit_name', '')}"
        
        # 質問
        question = item.get('question', '')
        
        # 文とフレーム情報
        sentences = item.get('sentences', [])
        sentences = [f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)]
        frames = item.get('frames', [])
        full_qa = question + '\n' + ('\n').join(sentences)

        # 文情報を構築
        if len(sentences) == 4 and len(frames) == 4:
            # 全文表示の場合
            sentence1_info = f"Sentence 1 フレーム: {frames[0]}\nSentence 2 フレーム: {frames[1]}"
            sentence2_info = f"Sentence 3 フレーム: {frames[2]}\nSentence 4 フレーム: {frames[3]}"
        elif len(sentences) >= 2 and len(frames) >= 2:
            sentence1_info = f"フレーム: {frames[0]}"
            sentence2_info = f"フレーム: {frames[1]}"
        elif len(sentences) == 1 and len(frames) == 1:
            # 単発問題の場合
            sentence1_info = f"フレーム: {frames[0]}"
            sentence2_info = ""
        else:
            sentence1_info = ""
            sentence2_info = ""
        
        # 正解情報
        answer = item.get('answer', '')
        answer_info = f"正解: {answer}"
        
        # 現在の修正があれば表示
        annotation_key = self.get_annotation_key()
        current_correction = self.annotations.get(annotation_key, {})
        corrected_question = current_correction.get("corrected_question", full_qa)
        comments = current_correction.get("comments", "")
        is_bad_example = (corrected_question == "bad example")
        
        return (
            current_info,
            verb_info,
            sentence1_info,
            sentence2_info,
            full_qa,
            corrected_question,
            is_bad_example,
            comments,
            answer_info,
        )

def create_text_correction_interface(data_file: str = None):
    """テキスト修正専用のGradioインターフェースを作成"""
    # デフォルトのデータファイルを設定
    if data_file is None:
        data_file = "data/ja/gpt-5/high_quality_qa.jsonl"
    
    # アプリケーションの初期化
    app = TextCorrectionApp(
        data_file,
        "src/evaluation_criteria.yaml"
    )
    
    with gr.Blocks(title="Frame QA テキスト修正ツール", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Frame QA テキスト修正ツール")
        gr.Markdown("質問の文法的なミスを修正し、より自然な日本語にしてください。")
        
        # データセット選択セクション
        with gr.Row():
            with gr.Column(scale=3):
                dataset_dropdown = gr.Dropdown(
                    label="QAデータセット",
                    choices=[(data_file, data_file)],  # 初期値を選択肢に含める
                    value=data_file,
                    interactive=True,
                    info="アノテーション対象のデータセットを選択してください"
                )
            with gr.Column(scale=1):
                refresh_datasets_btn = gr.Button("データセット一覧を更新", variant="secondary")
        
        # ログインセクション
        with gr.Accordion("ログイン"):
            with gr.Row():
                username_input = gr.Textbox(label="ユーザーID")
                password_input = gr.Textbox(label="パスワード", type="password")
            with gr.Row():
                login_btn = gr.Button("ログイン", variant="primary")
                logout_btn = gr.Button("ログアウト", variant="secondary")
        
        # ログイン後の情報を横並びで表示
        with gr.Row():
            # data/以降のパスを取得
            display_path = data_file.replace('data/', '', 1) if data_file.startswith('data/') else data_file
            dataset_info = gr.Textbox(
                label="選択されたデータセット",
                value=f"データセット: {display_path}",
                interactive=False,
                scale=2
            )
            login_status = gr.Textbox(
                label="ログイン状態",
                value="未ログイン",
                interactive=False,
                scale=1
            )
            format_info = gr.Textbox(
                label="問題形式",
                value="4択問題",
                interactive=False,
                scale=1
            )
        
        # 問題形式選択ボタン
        with gr.Row():
            format_four_btn = gr.Button("4択問題", variant="primary", scale=1)
            format_two_btn = gr.Button("2択問題", variant="secondary", scale=1)
            format_single_btn = gr.Button("単発問題", variant="secondary", scale=1)
            format_all_sentence_btn = gr.Button("全文表示", variant="secondary", scale=1)
        
        # メインコンテンツ（ログイン後に表示）
        main_content = gr.Column()
        
        with main_content:
            with gr.Row():
                with gr.Column(scale=1):
                    # ナビゲーション
                    gr.Markdown("## ナビゲーション")
                    with gr.Row():
                        prev_btn = gr.Button("← 前へ", variant="secondary")
                        next_btn = gr.Button("次へ →", variant="primary")
                    
                    # 現在の項目情報とジャンプ機能
                    with gr.Row():
                        current_info = gr.Textbox(label="現在の項目", interactive=False, scale=3)
                        jump_input = gr.Number(label="項目番号", minimum=1, maximum=1000, step=1, scale=1)
                        jump_btn = gr.Button("ジャンプ", variant="secondary", scale=1)
                    verb_info = gr.Textbox(label="フレーム喚起語", interactive=False)
                    sentence1_info = gr.Textbox(label="文1のフレーム", interactive=False, lines=2)
                    sentence2_info = gr.Textbox(label="文2のフレーム", interactive=False, lines=2)
                    answer_info = gr.Textbox(label="正解", interactive=False)
                    
                with gr.Column(scale=2):
                    # 質問表示と修正
                    gr.Markdown("## 質問の修正")
                    original_question = gr.Textbox(label="元の質問", interactive=False, lines=4)
                    corrected_question = gr.Textbox(
                        label="修正した質問", 
                        lines=4, 
                        placeholder="文法的なミスがあれば修正してください。より自然な日本語にしてください。"
                    )
                    bad_example_chk = gr.Checkbox(label="簡単な修正では解決できない悪問")
                    
                    gr.Markdown("""
                        ### 修正してほしいポイント
                        - 文法エラーの修正、許容不可能な日本語の修正
                        - より自然にするための修正はありがたいですが、Sentence1とSentence2の共通する内容語を減らしてしまうような修正や、以下のポイントに当てはまるような修正は避けてください。
                        
                        ### 修正してはいけないポイント
                        - ❌ 喚起されるフレームが変わってしまうような修正
                        - ❌ 喚起語を他の動詞にするような修正
                        """)
                    
                    # コメント
                    comments = gr.Textbox(
                        label="コメント", 
                        lines=1, 
                        placeholder="なにかあれば記入してください（不要）"
                    )
                    
                    # 保存ボタン
                    save_btn = gr.Button("修正を保存（次の問題へ）", variant="primary")
                    save_status = gr.Textbox(label="保存状況", interactive=False)
                    
                    # セッション管理
                    gr.Markdown("## セッション管理")
                    session_info = gr.Textbox(label="セッション情報", interactive=False)
                    with gr.Row():
                        clear_session_btn = gr.Button("セッションをクリア", variant="secondary")
                        download_btn = gr.Button("アノテーション結果をダウンロード", variant="secondary")
                    download_file = gr.File(label="ダウンロードファイル")
        
        # イベントハンドラー
        def load_datasets():
            """利用可能なデータセットを読み込み"""
            try:
                from .dataset_manager import dataset_manager
                datasets = dataset_manager.find_qa_datasets()
                choices = [(f"{d['name']} ({d['line_count']}行)", d['path']) for d in datasets]
                # 現在のdata_fileが選択肢に含まれているかチェック
                current_value = data_file if any(d['path'] == data_file for d in datasets) else (choices[0][1] if choices else None)
                return gr.update(choices=choices, value=current_value)
            except Exception as e:
                print(f"データセット読み込みエラー: {e}")
                # フォールバック: デフォルトのデータファイルのみ
                return gr.update(choices=[(data_file, data_file)], value=data_file)
        
        def on_dataset_selected(selected_path):
            """データセットが選択された時の処理"""
            if not selected_path or selected_path == data_file:
                # 変更なしの場合も10個の値を返す（dataset_infoのみ更新、他は変更なし）
                display_path = data_file.replace('data/', '', 1) if data_file.startswith('data/') else data_file
                return (
                    gr.update(value=f"データセット: {display_path}"),
                    gr.update(),  # current_info
                    gr.update(),  # verb_info
                    gr.update(),  # sentence1_info
                    gr.update(),  # sentence2_info
                    gr.update(),  # original_question
                    gr.update(),  # corrected_question
                    gr.update(),  # bad_example_chk
                    gr.update(),  # comments
                    gr.update()   # answer_info
                )
            
            # データセットを変更
            success, message = app.change_dataset(selected_path)
            if success:
                # 表示を更新
                display_data = app.update_display()
                (
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_question_v, is_bad_v, comments_v,
                    answer_info_v
                ) = display_data
                corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
                display_path = selected_path.replace('data/', '', 1) if selected_path.startswith('data/') else selected_path
                
                return (
                    gr.update(value=f"データセット: {display_path}"),
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v
                )
            else:
                return (
                    gr.update(value=f"エラー: {message}"),
                    "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
                )
        
        def on_login(username, password):
            if app.authenticate(username, password):
                # ログイン成功時にデータを自動表示
                display_data = app.update_display()
                (
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_question_v, is_bad_v, comments_v,
                    answer_info_v
                ) = display_data
                corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
                session_info = app.get_session_info()
                return (
                    f"ログイン成功: {username}",
                    gr.update(visible=True),
                    session_info,
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v,
                )
            else:
                return "ログイン失敗: ユーザーIDまたはパスワードが正しくありません", gr.update(visible=False), "未ログイン", "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
        
        def on_logout():
            app.logout()
            return "ログアウトしました", gr.update(visible=False)
        
        def on_prev():
            if not app.is_authenticated():
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            result = app.prev_item()
            if isinstance(result, str):
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_question_v, is_bad_v, comments_v,
                answer_info_v
            ) = result
            corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
            return (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_update, is_bad_v, comments_v,
                answer_info_v
            )
        
        def on_next():
            if not app.is_authenticated():
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            result = app.next_item()
            if isinstance(result, str):
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_question_v, is_bad_v, comments_v,
                answer_info_v
            ) = result
            corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
            return (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_update, is_bad_v, comments_v,
                answer_info_v
            )
        
        def on_jump(jump_number):
            if not app.is_authenticated():
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            
            # 項目番号を0ベースのインデックスに変換
            target_index = int(jump_number) - 1
            
            if target_index < 0 or target_index >= len(app.data):
                return "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            
            # 指定された項目にジャンプ
            app.current_index = target_index
            result = app.update_display()
            (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_question_v, is_bad_v, comments_v,
                answer_info_v
            ) = result
            corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
            return (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_update, is_bad_v, comments_v,
                answer_info_v
            )
        
        def on_save(corrected_text, comments_text, is_bad):
            if is_bad:
                corrected_text = "bad example"
            
            # 修正を保存
            save_result = app.save_text_correction(corrected_text, comments_text)
            
            # 保存が成功した場合、次の問題に進む
            if "保存しました" in save_result:
                next_result = app.next_item()
                if isinstance(next_result, str):
                    # エラーの場合は保存結果のみ返す
                    return save_result, "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
                else:
                    # 次の問題のデータを返す
                    (
                        current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                        original_question_v, corrected_question_v, is_bad_v, comments_v,
                        answer_info_v
                    ) = next_result
                    corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
                    return (
                        save_result, current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                        original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v
                    )
            else:
                # 保存に失敗した場合は現在のデータを返す
                display_data = app.update_display()
                (
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_question_v, is_bad_v, comments_v,
                    answer_info_v
                ) = display_data
                corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
                return (
                    save_result, current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v
                )
        
        def on_download():
            if not app.is_authenticated():
                return None
            annotations_data = app.get_annotations_for_download()
            if annotations_data.startswith("アノテーション"):
                return None
            
            # 一時ファイルを作成
            import tempfile
            import os
            user_id = app.get_current_user()
            dataset_name = app.get_dataset_name()
            filename = f"text_corrections_{user_id}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(annotations_data)
                temp_path = f.name
            
            # ファイル名を変更
            final_path = os.path.join(os.path.dirname(temp_path), filename)
            os.rename(temp_path, final_path)
            
            return final_path
        
        def on_clear_session():
            if not app.is_authenticated():
                return "ログインが必要です", "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
            
            app.clear_user_data()
            # クリア後のアノテーションを保存（空のファイルとして保存）
            app.save_annotations_to_file("text_corrections.json")
            session_info = app.get_session_info()
            
            # 現在の問題表示を維持
            display_data = app.update_display()
            (
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_question_v, is_bad_v, comments_v,
                answer_info_v
            ) = display_data
            corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
            
            return (
                session_info,
                current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v
            )
        
        def on_update_session_info():
            if not app.is_authenticated():
                return "未ログイン"
            return app.get_session_info()
        
        def on_load():
            # 初期表示時は認証状態に関係なく空の状態を返す
            return "未ログイン", "未ログイン", "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""

        def on_toggle_bad(is_bad, current_corrected):
            if is_bad:
                return gr.update(value="bad example", interactive=False)
            return gr.update(interactive=True)
        
        def on_change_format(format_type):
            """問題形式を変更"""
            success, message = app.set_question_format(format_type)
            format_name = app.get_format_name()
            
            # ボタンのvariantを更新
            four_variant = "primary" if format_type == "four_choices" else "secondary"
            two_variant = "primary" if format_type == "two_choices" else "secondary"
            single_variant = "primary" if format_type == "single" else "secondary"
            all_sentence_variant = "primary" if format_type == "all_sentence" else "secondary"
            
            # 現在の表示を更新
            if app.is_authenticated():
                display_data = app.update_display()
                (
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_question_v, is_bad_v, comments_v,
                    answer_info_v
                ) = display_data
                corrected_update = gr.update(value=corrected_question_v, interactive=(not is_bad_v))
                
                return (
                    format_name,
                    gr.update(variant=four_variant),
                    gr.update(variant=two_variant),
                    gr.update(variant=single_variant),
                    gr.update(variant=all_sentence_variant),
                    current_info_v, verb_info_v, sentence1_info_v, sentence2_info_v,
                    original_question_v, corrected_update, is_bad_v, comments_v, answer_info_v
                )
            else:
                return (
                    format_name,
                    gr.update(variant=four_variant),
                    gr.update(variant=two_variant),
                    gr.update(variant=single_variant),
                    gr.update(variant=all_sentence_variant),
                    "", "", "", "", "", gr.update(value="", interactive=True), False, "", ""
                )
        
        # イベントの設定
        refresh_datasets_btn.click(load_datasets, outputs=[dataset_dropdown])
        dataset_dropdown.change(
            on_dataset_selected,
            inputs=[dataset_dropdown],
            outputs=[dataset_info, current_info, verb_info, sentence1_info, sentence2_info,
                    original_question, corrected_question, bad_example_chk, comments, answer_info]
        )
        
        # 問題形式変更のイベント
        format_four_btn.click(
            lambda: on_change_format("four_choices"),
            outputs=[format_info, format_four_btn, format_two_btn, format_single_btn, format_all_sentence_btn,
                    current_info, verb_info, sentence1_info, sentence2_info,
                    original_question, corrected_question, bad_example_chk, comments, answer_info]
        )
        format_two_btn.click(
            lambda: on_change_format("two_choices"),
            outputs=[format_info, format_four_btn, format_two_btn, format_single_btn, format_all_sentence_btn,
                    current_info, verb_info, sentence1_info, sentence2_info,
                    original_question, corrected_question, bad_example_chk, comments, answer_info]
        )
        format_single_btn.click(
            lambda: on_change_format("single"),
            outputs=[format_info, format_four_btn, format_two_btn, format_single_btn, format_all_sentence_btn,
                    current_info, verb_info, sentence1_info, sentence2_info,
                    original_question, corrected_question, bad_example_chk, comments, answer_info]
        )
        format_all_sentence_btn.click(
            lambda: on_change_format("all_sentence"),
            outputs=[format_info, format_four_btn, format_two_btn, format_single_btn, format_all_sentence_btn,
                    current_info, verb_info, sentence1_info, sentence2_info,
                    original_question, corrected_question, bad_example_chk, comments, answer_info]
        )
        
        login_btn.click(on_login, inputs=[username_input, password_input], outputs=[
            login_status, main_content, session_info, current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        logout_btn.click(on_logout, outputs=[login_status, main_content])
        
        prev_btn.click(on_prev, outputs=[
            current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        
        next_btn.click(on_next, outputs=[
            current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        
        jump_btn.click(on_jump, inputs=[jump_input], outputs=[
            current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        
        save_btn.click(on_save, inputs=[corrected_question, comments, bad_example_chk], outputs=[
            save_status, current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info
        ])
        download_btn.click(on_download, outputs=[download_file])
        clear_session_btn.click(on_clear_session, outputs=[
            session_info, current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        bad_example_chk.change(on_toggle_bad, inputs=[bad_example_chk, corrected_question], outputs=[corrected_question])
        
        # 初期表示
        interface.load(on_load, outputs=[
            login_status, session_info, current_info, verb_info, sentence1_info, sentence2_info,
            original_question, corrected_question, bad_example_chk, comments, answer_info,
        ])
        
        # データセット一覧の初期読み込み
        interface.load(load_datasets, outputs=[dataset_dropdown])
    
    return interface

if __name__ == "__main__":
    interface = create_text_correction_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861, share=False)
