"""
フォルダ内の複数の評価アノテーションJSONファイルを処理して、統合TSVを出力するスクリプト
Usage:
    uv run python postprocess/for_human_eval.py data/ja/gpt-4.1-mini/step3/annotated
"""

import json
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OPTION_LIST = ["Sentence 1", "Sentence 2", "Both sentences", "Neither sentence"]


def process_single_json(json_file: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    単一のJSONファイルを処理してDataFrameに変換
    
    Args:
        json_file: 処理するJSONファイルのパス
        output_dir: 中間TSVファイルの出力先（Noneの場合は出力しない）
    
    Returns:
        処理されたDataFrame
    """
    print(f"Processing: {json_file}")
    
    # JSONファイルを読み込む
    annotated_data = json.load(open(json_file, 'r', encoding='utf-8'))
    
    # DataFrameに変換
    df = pd.DataFrame(annotated_data.values())
    
    # 評価結果を追加
    df['predicted_answer'] = df.apply(
        lambda x: OPTION_LIST[x["evaluations"]["回答"] - 1] if "evaluations" in x and "回答" in x["evaluations"] else None,
        axis=1
    )
    df['match_answer'] = df['answer'] == df['predicted_answer']
    df['quality'] = df.apply(
        lambda x: True if x.get("evaluations", {}).get("日本語の品質") == 1 else False,
        axis=1
    )
    df['has_comment'] = df.apply(
        lambda x: True if x.get("comments", "") != "" else False,
        axis=1
    )
    df['is_test'] = df.apply(
        lambda x: True if x.get("answer") == "Both sentences" or x.get("answer") == "Neither sentence" else False,
        axis=1
    )
    
    # primeかabかのフラグを追加
    # sub_indexが0の場合はab、1の場合はprime
    def determine_problem_type(row):
        # pandasのSeriesから値を取得
        if "sub_index" in row.index:
            sub_idx = row["sub_index"]
            # 数値として比較
            if pd.notna(sub_idx):
                sub_idx = int(sub_idx) if isinstance(sub_idx, (int, float)) else sub_idx
                if sub_idx == 0:
                    return "ab"
                elif sub_idx == 1:
                    return "prime"
        
        # デフォルト: sub_indexが0ならab、1ならprime
        return "ab"
    
    df['problem_type'] = df.apply(determine_problem_type, axis=1)
    
    # 統計情報を出力
    print(f"  correct_count: {df['match_answer'].sum()} / {len(df)}")
    print(f"  quality_1_count: {len(df[df['quality']])} / {len(df)}")
    print(f"  correct_quality_1_count: {df[df['quality']]['match_answer'].sum()} / {len(df)}")
    print(f"  test_score: {df[df['is_test']]['match_answer'].sum()} / {len(df[df['is_test']])}")
    
    # 中間TSVファイルを出力（オプション）
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = output_dir / json_file.stem.replace(".json", ".tsv")
        df.to_csv(tsv_path, sep="\t", index=False, encoding='utf-8')
        print(f"  Saved intermediate TSV: {tsv_path}")
    
    return df


def extract_annotator_name(json_file: Path, df: pd.DataFrame) -> str:
    """
    アノテータ名を抽出（user_idから取得、なければファイル名から推測）
    
    Args:
        json_file: JSONファイルのパス
        df: データフレーム（user_idを含む）
    
    Returns:
        アノテータ名
    """
    # まずuser_idから取得を試みる
    if 'user_id' in df.columns and len(df['user_id'].unique()) > 0:
        user_id = df['user_id'].iloc[0]
        # ファイル名のパターンから推測（例: evaluations_admin_step3_qa_...）
        # admin, miyahara, tsujimoto, kurauchi など
        if user_id and user_id != "":
            return user_id
    
    # ファイル名から推測
    filename = json_file.stem
    # evaluations_{user_id}_{dataset}_{timestamp} の形式を想定
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]  # user_id部分
    
    # デフォルト: ファイル名のベース名
    return json_file.stem


def merge_dataframes(dataframes: list, annotator_names: list) -> pd.DataFrame:
    """
    複数のDataFrameをマージ
    original_qa_idとproblem_typeが同じデータを横に並べる
    
    Args:
        dataframes: マージするDataFrameのリスト
        annotator_names: 各DataFrameに対応するアノテータ名のリスト
    
    Returns:
        マージされたDataFrame
    """
    # 共通カラム（original_qa_idとproblem_typeでマージする際に共通として保持するカラム）
    # ただし、original_index、sub_index、sentence_pair、annotation_typeは出力から除外
    # corrected_questionはoriginal_questionにリネームして保持
    common_cols = [
        "data_id", "lex_unit_name", "question", "answer",
        "question_format", "original_index", "sub_index",
        "single-question", "two-choice-question", "four-choice-question",
        "sentence_A", "sentence_B", "sentence_A_prime", "sentence_B_prime",
        "frame_A", "frame_B", "sentence_pair", "corrected_question",
        "annotation_type", "is_test"
    ]
    
    # 出力から除外するカラム
    exclude_cols = [
        "original_index", "sub_index", "sentence_pair", "annotation_type"
    ]
    
    # 評価カラム（各アノテータごとに横に並べるカラム）
    # user_idとtimestampも評価カラムとして扱うが、最終的に出力から除外
    evaluation_cols = [
        "user_id", "timestamp", "current_index", "evaluations",
        "comments", "predicted_answer", "match_answer", "quality", "has_comment"
    ]
    
    # マージキーはoriginal_qa_id、problem_type、sub_index
    # sub_indexも含めることで、同じoriginal_qa_idとproblem_typeでもsub_indexが異なる場合は別の行として扱う
    merge_keys = ["original_qa_id", "problem_type", "sub_index"]
    
    all_data = []
    
    for df, annotator_name in zip(dataframes, annotator_names):
        # 評価カラムにアノテータ名のサフィックスを追加
        rename_dict = {}
        for col in evaluation_cols:
            if col in df.columns:
                rename_dict[col] = f"{col}_{annotator_name}"
        
        df_renamed = df.rename(columns=rename_dict)
        all_data.append(df_renamed)
    
    # 最初のDataFrameをベースにする
    base_df = all_data[0].copy()
    
    # 残りのDataFrameを順次マージ
    for df in all_data[1:]:
        # マージ時に共通カラムが重複しないように、右側の共通カラムを削除
        # ただし、マージキーは保持
        cols_to_keep = [col for col in df.columns if col not in common_cols or col in merge_keys]
        df_to_merge = df.loc[:, cols_to_keep].copy()
        
        # マージ
        base_df = pd.merge(
            base_df,
            df_to_merge,
            on=merge_keys,
            how="outer"
        )
    
    # カラムの順序を整理
    # まず共通カラムとマージキー
    ordered_cols = []
    for col in common_cols + merge_keys:
        if col in base_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    
    # 次に各アノテータの評価カラム
    for annotator_name in annotator_names:
        for col in evaluation_cols:
            col_name = f"{col}_{annotator_name}"
            if col_name in base_df.columns and col_name not in ordered_cols:
                ordered_cols.append(col_name)
    
    # その他のカラム
    for col in base_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    result_df = base_df[ordered_cols]
    assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
    
    # 出力から除外するカラムを削除
    # マージキーのsub_indexも削除
    cols_to_drop = exclude_cols + ["sub_index"]
    # 各アノテータのuser_idとtimestampも削除（サフィックス付き）
    for annotator_name in annotator_names:
        cols_to_drop.extend([f"user_id_{annotator_name}", f"timestamp_{annotator_name}"])
    
    # マージ時に自動生成されたuser_id_x、timestamp_x、user_id_y、timestamp_yなども削除
    for col in result_df.columns:
        if (col.startswith("user_id_") or col.startswith("timestamp_")) and col not in cols_to_drop:
            cols_to_drop.append(col)
    
    # 存在するカラムのみ削除
    cols_to_drop = [col for col in cols_to_drop if col in result_df.columns]
    result_df = result_df.drop(columns=cols_to_drop)
    
    # corrected_questionをoriginal_questionにリネーム
    if "corrected_question" in result_df.columns:
        result_df = result_df.rename(columns={"corrected_question": "original_question"})
    
    # original_questionの右に新しい空のcorrected_questionカラムを追加
    if "original_question" in result_df.columns:
        # original_questionの位置を取得
        original_question_idx = list(result_df.columns).index("original_question")
        # 新しい空のcorrected_questionカラムを追加
        result_df.insert(original_question_idx + 1, "corrected_question", "")
    
    return result_df


def print_summary_statistics(dataframes: list, annotator_names: list):
    """
    全体の統計情報を出力
    
    Args:
        dataframes: 処理されたDataFrameのリスト
        annotator_names: アノテータ名のリスト
    """
    print("\n=== Summary Statistics ===")
    
    for df, annotator_name in zip(dataframes, annotator_names):
        print(f"\n--- {annotator_name} ---")
        print(f"  Total: {len(df)}")
        print(f"  Correct: {df['match_answer'].sum()} ({df['match_answer'].sum()/len(df)*100:.1f}%)")
        print(f"  Quality=1: {len(df[df['quality']])} ({len(df[df['quality']])/len(df)*100:.1f}%)")
        print(f"  Test score: {df[df['is_test']]['match_answer'].sum()} / {len(df[df['is_test']])}")
        print(f"  Answer distribution:")
        print(f"    {df['answer'].value_counts().to_dict()}")
        print(f"  Predicted distribution:")
        print(f"    {df['predicted_answer'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(
        description="フォルダ内の複数の評価アノテーションJSONファイルを処理して統合TSVを出力"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="処理するJSONファイルが入ったフォルダのパス"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="中間TSVファイルも保存する"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="グラフを生成しない"
    )
    
    args = parser.parse_args()
    
    # 入力フォルダのパス
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: フォルダが見つかりません: {input_folder}")
        return
    
    # JSONファイルを検索
    json_files = sorted(list(input_folder.glob("*.json")))
    if len(json_files) == 0:
        print(f"Error: JSONファイルが見つかりません: {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON file(s):")
    for json_file in json_files:
        print(f"  - {json_file.name}")
    
    # 各JSONファイルを処理
    dataframes = []
    annotator_names = []
    intermediate_dir = input_folder / "intermediate" if args.save_intermediate else None
    
    for json_file in json_files:
        df = process_single_json(json_file, intermediate_dir)
        annotator_name = extract_annotator_name(json_file, df)
        dataframes.append(df)
        annotator_names.append(annotator_name)
        print(f"  Annotator: {annotator_name}")
    
    # 統計情報を出力
    print_summary_statistics(dataframes, annotator_names)
    
    # データフレームをマージ
    # original_qa_idとproblem_typeが同じデータを横に並べる
    print("\n=== Merging DataFrames ===")
    merged_data = merge_dataframes(dataframes, annotator_names)
    print(f"Merged data shape: {merged_data.shape}")
    
    # 出力パスを決定
    output_path = input_folder.parent / "evaluations_merged.tsv"
    
    # TSVファイルとして出力
    merged_data.to_csv(output_path, sep="\t", index=False, encoding='utf-8')
    print(f"\nSaved merged TSV: {output_path}")
    
    # グラフを生成（オプション）
    if not args.no_plots and len(dataframes) > 0:
        print("\n=== Generating Plots ===")
        # 最初のデータフレームを使ってグラフを生成
        df = dataframes[0]
        
        # Answer Distribution
        plt.figure(figsize=(10, 6))
        plt.bar(df['answer'].value_counts().index, df['answer'].value_counts().values)
        plt.xlabel('Answer')
        plt.ylabel('Count')
        plt.title('Answer Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = input_folder / "answer_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved: {plot_path}")
        
        # Correct vs Predicted (with Comment Breakdown)
        categories = OPTION_LIST
        x = np.arange(len(categories))
        width = 0.35
        
        correct_counts = []
        predicted_wo_comment_counts = []
        predicted_w_comment_counts = []
        
        for category in categories:
            correct_count = len(df[df['answer'] == category])
            correct_counts.append(correct_count)
            
            predicted_wo_comment = len(df[
                (df['predicted_answer'] == category) & 
                (~df['has_comment'])
            ])
            predicted_wo_comment_counts.append(predicted_wo_comment)
            
            predicted_w_comment = len(df[
                (df['predicted_answer'] == category) & 
                (df['has_comment'])
            ])
            predicted_w_comment_counts.append(predicted_w_comment)
        
        _, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, correct_counts, width, label='Correct', color='#1f77b4')
        ax.bar(x + width/2, predicted_wo_comment_counts, width, label='Human Annotated (w/o comment)', color='#aec7e8')
        ax.bar(x + width/2, predicted_w_comment_counts, width, bottom=predicted_wo_comment_counts, label='Human Annotated (w/ comment)', color='#ff7f0e')
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Correct vs Human Annotated')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plot_path = input_folder / "correct_vs_predicted.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    main()

