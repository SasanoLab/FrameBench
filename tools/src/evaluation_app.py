import gradio as gr
import json
import os
from typing import Dict
from datetime import datetime
from .common import BaseFrameQAApp

class EvaluationApp(BaseFrameQAApp):
    """評価専用のFrame QAアプリケーション"""
    
    def __init__(self, data_file: str, criteria_file: str):
        super().__init__(data_file, criteria_file, user_assignment_file="src/user_assignment.json")
        # 既存のアノテーションを読み込み
        self.load_annotations_from_file("evaluations.json")
    
    def save_evaluation(self, evaluations: Dict[str, int], comments: str):
        """評価結果を保存"""
        if not self.is_authenticated():
            return "エラー: ログインが必要です"
        
        item = self.get_current_item()
        if item is None:
            return "エラー: データが見つかりません"
        
        # 展開後のitemから質問を取得
        question = item.get('question', '')
        
        # 文とフレーム情報
        sentences = item.get('sentences', [])
        sentences_formatted = [f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)]
        full_qa = question + '\n' + ('\n').join(sentences_formatted)
        
        # テキスト修正ファイルから修正された質問を取得（あれば）
        corrected_question = full_qa  # デフォルトは元の質問
        try:
            annotation_key = self.get_annotation_key()
            # ユーザー別のテキスト修正ファイルを読み込む
            user_id = self.get_current_user()
            correction_file = f"text_corrections_{user_id}.json"
            
            if os.path.exists(correction_file):
                with open(correction_file, 'r', encoding='utf-8') as f:
                    corrections = json.load(f)
                    if annotation_key in corrections:
                        corrected_question = corrections[annotation_key].get("corrected_question", corrected_question)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # テキスト修正ファイルがない場合は元の質問を使用
        
        # 評価データを作成
        evaluation_data = {
            "corrected_question": corrected_question,
            "evaluations": evaluations,
            "comments": comments,
            "annotation_type": "evaluation"
        }
        
        # メタデータを追加してアノテーションを作成
        annotation = self.create_annotation_with_metadata(evaluation_data)
        
        self.annotations[self.get_annotation_key()] = annotation
        
        # ファイルに保存
        return self.save_annotations_to_file("evaluations.json")
    
    def update_display(self):
        """表示を更新（評価専用）- 展開されたデータを使用"""
        item = self.get_current_item()
        if item is None:
            return ("データが見つかりません", "", "", {})
        
        # 展開されたアイテムから情報を取得
        current_info = self.get_item_info()
        
        # 質問
        question = item.get('question', '')
        
        # 文とフレーム情報
        sentences = item.get('sentences', [])
        sentences = [f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)]
        full_qa = question + '\n' + ('\n').join(sentences)
        
        # 現在の評価があれば表示
        annotation_key = self.get_annotation_key()
        current_evaluation = self.annotations.get(annotation_key, {})
        comments = current_evaluation.get("comments", "")
        current_evaluations = current_evaluation.get("evaluations", {})
        
        return (
            current_info,
            full_qa,
            comments,
            current_evaluations
        )

def create_evaluation_interface(data_file: str = None):
    """評価専用のGradioインターフェースを作成"""
    # デフォルトのデータファイルを設定
    if data_file is None:
        data_file = "data/ja/gpt-5/high_quality_qa.jsonl"
    
    # アプリケーションの初期化
    app = EvaluationApp(
        data_file,
        "src/evaluation_criteria.yaml"
    )
    
    with gr.Blocks(
        title="Frame QA 評価ツール", 
        theme=gr.themes.Soft(),
        css="""
        .evaluation-criterion {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        .evaluation-criterion h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .scale-labels {
            font-size: 12px;
            color: #666;
            margin-top: 6px;
        }
        """
    ) as gradio_interface:
        gr.Markdown("# Frame QA 評価ツール")
        gr.Markdown("修正された質問の品質を複数の観点から5段階で評価してください。")
        
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
        
        # メインコンテンツ（ログイン後に表示）
        main_content = gr.Column()
        
        with main_content:
            with gr.Row():
                with gr.Column(scale=1):
                    # ナビゲーション
                    with gr.Row():
                        prev_btn = gr.Button("← 前の問題", variant="secondary")
                        next_btn = gr.Button("次の問題 →", variant="primary")
                    
                    # 現在の項目情報とジャンプ機能
                    with gr.Row():
                        current_info = gr.Textbox(label="現在の項目", interactive=False, scale=3)
                        jump_input = gr.Number(label="項目番号", minimum=1, maximum=app.get_data_count(), step=1, scale=1)
                        jump_btn = gr.Button("ジャンプ", variant="secondary", scale=1)
                    
                with gr.Column(scale=2):
                    # 質問表示
                    original_question = gr.Textbox(label="質問", interactive=False, lines=4)
                    
                    # 評価項目（問題形式ごとに動的に生成）
                    evaluation_inputs = {}
                    evaluation_containers = []
                    if app.criteria and "evaluation_criteria" in app.criteria:
                        for criterion in app.criteria["evaluation_criteria"]:
                            name = criterion["name"]
                            description = criterion["description"]
                            
                            container = gr.Column(elem_classes=["evaluation-criterion"])
                            evaluation_containers.append((name, container))
                            
                            with container:
                                gr.Markdown(f"### {name}")
                                gr.Markdown(f"{description}")
                                
                                # 問題形式に応じた評価UIを生成するためのプレースホルダー
                                # 初期値として4択問題の選択肢を表示
                                scale_info = criterion.get("scale", {})
                                
                                # 問題形式ごとのスケールを取得
                                format_type = app.get_question_format() if hasattr(app, 'get_question_format') else "four_choices"
                                format_scale = scale_info.get(format_type, scale_info.get("four_choices", {}))
                                
                                # スケールがネストされた形式の場合、適切に処理
                                if not format_scale or not isinstance(format_scale, dict):
                                    # 旧形式（ネストなし）の場合
                                    if isinstance(scale_info, dict) and all(isinstance(k, (int, str)) and not isinstance(v, dict) for k, v in scale_info.items()):
                                        format_scale = scale_info
                                    else:
                                        format_scale = {1: "はい", 2: "いいえ"}
                                
                                # 選択肢とラベルを作成
                                choices = [(f"{k}: {v}", int(k)) for k, v in sorted(format_scale.items(), key=lambda x: int(x[0]))]
                                
                                # 「日本語の品質」の初期値を「自然」（1）に設定
                                default_value = 1 if name == "日本語の品質" else None
                                
                                evaluation_inputs[name] = gr.Radio(
                                    choices=choices,
                                    label="評価",
                                    value=default_value
                                )

                    
                    # コメント
                    comments = gr.Textbox(label="コメント", lines=1, placeholder="なにかあれば記入してください（不要）")
                    
                    # 保存ボタン
                    save_btn = gr.Button("評価を保存（次の問題へ）", variant="primary")
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
            except (ImportError, AttributeError, KeyError) as e:
                print(f"データセット読み込みエラー: {e}")
                # フォールバック: デフォルトのデータファイルのみ
                return gr.update(choices=[(data_file, data_file)], value=data_file)
        
        def on_dataset_selected(selected_path):
            """データセットが選択された時の処理"""
            if not selected_path or selected_path == data_file:
                # 変更なしの場合
                display_path = data_file.replace('data/', '', 1) if data_file.startswith('data/') else data_file
                return (
                    gr.update(value=f"データセット: {display_path}"),
                    gr.update(),  # current_info
                    gr.update(),  # original_question
                    gr.update(),  # comments
                    gr.update(),  # gr.State()
                    gr.update()   # jump_input
                )
            
            # データセットを変更
            success, message = app.change_dataset(selected_path)
            if success:
                # 表示を更新
                display_data = app.update_display()
                display_path = selected_path.replace('data/', '', 1) if selected_path.startswith('data/') else selected_path
                return (
                    gr.update(value=f"データセット: {display_path}"),
                    *display_data,
                    gr.update(maximum=app.get_data_count())
                )
            else:
                return (
                    gr.update(value=f"エラー: {message}"),
                    "", "", "", {},  # 空文字列と空辞書
                    gr.update()  # jump_input
                )
        
        def get_evaluation_choices_for_format(format_type):
            """問題形式に応じた評価項目の選択肢を取得"""
            evaluation_updates = {}
            if app.criteria and "evaluation_criteria" in app.criteria:
                for criterion in app.criteria["evaluation_criteria"]:
                    name = criterion["name"]
                    scale_info = criterion.get("scale", {})
                    
                    # 問題形式ごとのスケールを取得
                    format_scale = scale_info.get(format_type, scale_info.get("four_choices", {}))
                    
                    # スケールがネストされた形式の場合、適切に処理
                    if not format_scale or not isinstance(format_scale, dict):
                        # 旧形式（ネストなし）の場合
                        if isinstance(scale_info, dict) and all(isinstance(k, (int, str)) and not isinstance(v, dict) for k, v in scale_info.items()):
                            format_scale = scale_info
                        else:
                            format_scale = {1: "はい", 2: "いいえ"}
                    
                    # 選択肢とラベルを作成
                    choices = [(f"{k}: {v}", int(k)) for k, v in sorted(format_scale.items(), key=lambda x: int(x[0]))]
                    # 「日本語の品質」の初期値を「自然」（1）に設定
                    default_value = 1 if name == "日本語の品質" else None
                    evaluation_updates[name] = gr.update(choices=choices, value=default_value)
            
            return evaluation_updates
        
        def on_change_format(format_type):
            """問題形式を変更"""
            app.set_question_format(format_type)
            format_name = app.get_format_name()
            
            # ボタンのvariantを更新
            four_variant = "primary" if format_type == "four_choices" else "secondary"
            two_variant = "primary" if format_type == "two_choices" else "secondary"
            single_variant = "primary" if format_type == "single" else "secondary"
            
            # 評価項目の選択肢を更新
            eval_updates = get_evaluation_choices_for_format(format_type)
            
            # 現在の表示を更新
            if app.is_authenticated():
                display_data = app.update_display()
                if len(display_data) >= 4:
                    (
                        current_info_v, question_v, comments_v, current_evaluations
                    ) = display_data
                    
                    result = [
                        format_name,
                        gr.update(variant=four_variant),
                        gr.update(variant=two_variant),
                        gr.update(variant=single_variant),
                        current_info_v, question_v, comments_v, current_evaluations
                    ]
                    # 評価項目の更新を追加
                    for name in evaluation_inputs.keys():
                        result.append(eval_updates.get(name, gr.update()))
                    return tuple(result)
            
            result = [
                format_name,
                gr.update(variant=four_variant),
                gr.update(variant=two_variant),
                gr.update(variant=single_variant),
                "", "", "", {}
            ]
            # 評価項目の更新を追加
            for name in evaluation_inputs.keys():
                result.append(eval_updates.get(name, gr.update()))
            return tuple(result)
        
        def on_login(username, password):
            if app.authenticate(username, password):
                # ログイン成功時にデータを自動表示
                display_data = app.update_display()
                session_info_text = app.get_session_info()
                return f"ログイン成功: {username}", gr.update(visible=True), session_info_text, *display_data
            else:
                return "ログイン失敗: ユーザーIDまたはパスワードが正しくありません", gr.update(visible=False), "未ログイン", "", "", "", {}
        
        def on_logout():
            app.logout()
            return "ログアウトしました", gr.update(visible=False)
        
        def on_prev():
            if not app.is_authenticated():
                return "ログインが必要です", "", "", {}
            result = app.prev_item()
            if isinstance(result, str):
                return result, "", "", {}
            return result
        
        def on_next():
            if not app.is_authenticated():
                return "ログインが必要です", "", "", {}
            result = app.next_item()
            if isinstance(result, str):
                return result, "", "", {}
            return result
        
        def on_jump(jump_number):
            if not app.is_authenticated():
                return "ログインが必要です", "", "", {}
            
            # 項目番号を0ベースのインデックスに変換
            target_index = int(jump_number) - 1
            
            if target_index < 0 or target_index >= len(app.data):
                return f"項目番号が範囲外です (1-{len(app.data)})", "", "", {}
            
            # 指定された項目にジャンプ
            app.current_index = target_index
            result = app.update_display()
            return result
        
        def on_save(*args):
            # 評価値を取得
            evaluations = {}
            for i, (name, _) in enumerate(evaluation_inputs.items()):
                value = args[i]
                if value is None:
                    # 評価項目の値をNoneでリセット
                    eval_resets = tuple([None] * len(evaluation_inputs))
                    return (f"エラー: 「{name}」の評価が選択されていません", "", "", "", {}) + eval_resets
                evaluations[name] = int(value)
            
            # コメントを取得
            comments_text = args[len(evaluation_inputs)]
            
            # 評価を保存
            save_result = app.save_evaluation(evaluations, comments_text)
            
            # 保存が成功した場合、次の問題に進む
            if "保存しました" in save_result:
                next_result = app.next_item()
                if isinstance(next_result, str):
                    # エラーの場合（最後の問題など）
                    eval_resets = tuple([None] * len(evaluation_inputs))
                    return (save_result, "", "", "", {}) + eval_resets
                else:
                    # 次の問題のデータを返す
                    (
                        current_info_v, question_v, comments_v, current_evaluations
                    ) = next_result
                    
                    # 次の問題の評価項目の値を設定（保存済みの評価があればそれを使用、なければデフォルト値）
                    eval_values = tuple([
                        current_evaluations.get(name, 1 if name == "日本語の品質" else None) 
                        for name in evaluation_inputs.keys()
                    ])
                    
                    return (
                        save_result, current_info_v, question_v, comments_v, current_evaluations
                    ) + eval_values
            else:
                # 保存に失敗した場合は現在のデータを返す
                display_data = app.update_display()
                (
                    current_info_v, question_v, comments_v, current_evaluations
                ) = display_data
                eval_values = tuple([
                    current_evaluations.get(name, 1 if name == "日本語の品質" else None) 
                    for name in evaluation_inputs.keys()
                ])
                return (
                    save_result, current_info_v, question_v, comments_v, current_evaluations
                ) + eval_values
        
        def on_download():
            if not app.is_authenticated():
                return None
            annotations_data = app.get_annotations_for_download()
            if annotations_data.startswith("アノテーション"):
                return None
            
            # 一時ファイルを作成
            import tempfile
            user_id = app.get_current_user()
            dataset_name = app.get_dataset_name()
            filename = f"evaluations_{user_id}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(annotations_data)
                temp_path = f.name
            
            # ファイル名を変更
            final_path = os.path.join(os.path.dirname(temp_path), filename)
            os.rename(temp_path, final_path)
            
            return final_path
        
        def on_clear_session():
            if not app.is_authenticated():
                return "ログインが必要です", "エラー: ログインが必要です", "", "", "", {}
            
            app.clear_user_data()
            # クリア後のアノテーションを保存（空のファイルとして保存）
            app.save_annotations_to_file("evaluations.json")
            session_info_text = app.get_session_info()
            
            # 現在の問題表示を維持
            display_data = app.update_display()
            return session_info_text, "セッションをクリアしました（アノテーション数: 0）", *display_data
        
        def on_load():
            # 初期表示時は認証状態に関係なく空の状態を返す
            return "未ログイン", "未ログイン", "", "", "", {}
        
        # イベントの設定
        refresh_datasets_btn.click(load_datasets, outputs=[dataset_dropdown])
        dataset_dropdown.change(
            on_dataset_selected,
            inputs=[dataset_dropdown],
            outputs=[dataset_info, current_info, original_question, comments, gr.State(), jump_input]
        )
        
        # 問題形式変更のイベント
        format_outputs = [format_info, format_four_btn, format_two_btn, format_single_btn,
                         current_info, original_question, comments, gr.State()] + list(evaluation_inputs.values())
        
        format_four_btn.click(
            lambda: on_change_format("four_choices"),
            outputs=format_outputs
        )
        format_two_btn.click(
            lambda: on_change_format("two_choices"),
            outputs=format_outputs
        )
        format_single_btn.click(
            lambda: on_change_format("single"),
            outputs=format_outputs
        )
        
        login_btn.click(on_login, inputs=[username_input, password_input], outputs=[
            login_status, main_content, session_info, current_info,
            original_question, comments, gr.State()
        ])
        logout_btn.click(on_logout, outputs=[login_status, main_content])
        
        prev_btn.click(on_prev, outputs=[
            current_info, original_question, comments, gr.State()
        ])
        
        next_btn.click(on_next, outputs=[
            current_info, original_question, comments, gr.State()
        ])
        
        jump_btn.click(on_jump, inputs=[jump_input], outputs=[
            current_info, original_question, comments, gr.State()
        ])
        
        # 保存ボタンの入力として、評価スライダーとコメントを指定
        save_inputs = list(evaluation_inputs.values()) + [comments]
        save_btn.click(on_save, inputs=save_inputs, outputs=[
            save_status, current_info, original_question, comments, gr.State()
        ] + list(evaluation_inputs.values()))
        download_btn.click(on_download, outputs=[download_file])
        clear_session_btn.click(on_clear_session, outputs=[
            session_info, save_status, current_info, original_question, comments, gr.State()
        ])
        
        # 初期表示
        gradio_interface.load(on_load, outputs=[
            login_status, session_info, current_info, original_question, comments, gr.State()
        ])
        
        # データセット一覧の初期読み込み
        gradio_interface.load(load_datasets, outputs=[dataset_dropdown])
    
    return gradio_interface

if __name__ == "__main__":
    interface = create_evaluation_interface()
    interface.launch(server_name="0.0.0.0", server_port=7862, share=False)
