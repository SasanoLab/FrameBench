import gradio as gr
import os
from typing import Tuple  # 追加
from src.text_correction_app import create_text_correction_interface
from src.evaluation_app import create_evaluation_interface

def create_main_interface(data_file: str):
    """メインのインターフェースを作成"""
    
    with gr.Blocks(title="Frame QA アノテーションツール", theme=gr.themes.Soft()) as main_interface:
        gr.Markdown("# Frame QA アノテーションツール")
        gr.Markdown("""
        Frame QAデータセットの質問品質を評価・修正するためのツール群です。
        
        ## 結果の共有方法
        項目ごとに修正を行い、修正を保存をクリックしてください。
        すべての項目についてアノテーションが完了したら、右下の「アノテーション結果をダウンロード」ボタンでJSONファイルをダウンロードし、ファイルを共有してください。
        
        もし、途中で中断した場合は、毎度アノテーション結果をダウンロードした後、セッションをクリアボタンを推してから中断してください。(再ログインが必要です)
        その場合はすべてのアノテーション結果ファイルを共有してください。

        ## アプリケーション選択
        以下の2つのアプリケーションから選択してください。
        各アプリケーション内でデータセットを選択・変更できます:
        """)
        
        with gr.Row():
            with gr.Column():
                text_correction_btn = gr.Button("テキスト修正アプリを起動", variant="primary")
            
            with gr.Column():
                evaluation_btn = gr.Button("品質評価アプリを起動", variant="primary")
            
        
        # 各アプリケーションのインターフェースを隠し状態で作成
        text_correction_interface = gr.Column(visible=False)  # visible=False を追加
        evaluation_interface = gr.Column(visible=False)  # visible=False を追加
        annotation_interface = gr.Column(visible=False)  # visible=False を追加
        
        # イベントハンドラー - 型アノテーション追加
        def show_text_correction() -> Tuple[dict, dict, dict]:  # 型アノテーション追加
            return (
                gr.update(visible=True), 
                gr.update(visible=False), 
                gr.update(visible=False)
            )
        
        def show_evaluation() -> Tuple[dict, dict, dict]:  # 型アノテーション追加
            return (
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(visible=False)
            )
        
        # イベントの設定
        text_correction_btn.click(
            show_text_correction, 
            outputs=[text_correction_interface, evaluation_interface, annotation_interface]
        )
        evaluation_btn.click(
            show_evaluation, 
            outputs=[text_correction_interface, evaluation_interface, annotation_interface]
        )
        
        # 各アプリケーションのインターフェースを埋め込み
        with text_correction_interface:
            create_text_correction_interface(data_file)
        
        with evaluation_interface:
            create_evaluation_interface(data_file)
        
    return main_interface

if __name__ == "__main__":
    # データファイルのパスを確認
    data_file = "data/ja/gpt-5/qa.jsonl"
    if not os.path.exists(data_file):
        print(f"警告: データファイルが見つかりません: {data_file}")
    interface = create_main_interface(data_file)
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)