"""
<data_root>/<language>-framenet/frames.jsonl
にあるフレームデータを後続の処理で利用しやすいようにlexical unit中心のデータに再編成して、
<data_root>/<language>-framenet/lexical_units.jsonl
に保存する。

Usage:
python src/1-2_lu_driven_edit.py --data_root <data_root> --language <language>

"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import argparse

def reorganize_by_lexical_units(frames_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    フレーム中心のデータをlexical unit中心のデータに再編成し、同じluはまとめる

    Args:
        frames_data: frames.jsonlから読み込んだフレーム中心のデータリスト

    Returns:
        lexical unit中心のデータ辞書 (同じluがまとめられている)
    """
    # 各lexical unitのデータを格納する辞書
    lex_unit_data = {}
    
    # 全フレームを処理
    for frame in frames_data:
        frame_name = frame['frame_name']
        frame_definition = frame['definition']
        core_elements = frame['core_frame_elements']
        
        # フレーム内の各lexical unitを処理
        for lu in frame['lex_units']:
            lu_name = lu['name']
            lu_pos = lu['pos']
            
            # lexical unitが既に存在するかチェック
            if lu_name not in lex_unit_data:
                # 新しいlexical unitの作成
                lex_unit_data[lu_name] = {
                    'lex_unit_name': lu_name,
                    'lex_unit_pos': lu_pos,
                    'frames': []
                }
            
            # このlexical unitに関連するフレーム情報を追加
            lex_unit_data[lu_name]['frames'].append({
                'frame_name': frame_name,
                'frame_definition': frame_definition,
                'core_elements': core_elements
            })
    
    return lex_unit_data

def main(args):
    # ファイルパスの設定
    frames_path = Path(args.data_root) / f'{args.language}-framenet' / 'frames.jsonl'
    output_path = Path(args.data_root) / f'{args.language}-framenet' / 'lexical_units.jsonl'
    
    # frames.jsonlファイルが存在するか確認
    if not frames_path.exists():
        print(f"エラー: {frames_path} が見つかりません。先にframe_parse.pyを実行してください。")
        return
    
    # frames.jsonlファイルを読み込む
    frames_data = []
    with open(frames_path, 'r', encoding='utf-8') as f:
        for line in f:
            frames_data.append(json.loads(line))
    
    print(f"{len(frames_data)}個のフレームを読み込みました")
    
    # lexical unit中心のデータに再編成（同じluはまとめる）
    lex_unit_data = reorganize_by_lexical_units(frames_data)
    
    # 結果をリストに変換
    lu_records = list(lex_unit_data.values())
    
    # jsonl形式で保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in lu_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"語彙ユニット中心のデータを {output_path} に保存しました")
    
    # 統計情報を表示
    print("\n統計情報:")
    print(f"語彙ユニット数: {len(lex_unit_data)}")
    
    # 複数のフレームに関連する語彙ユニット数
    multi_frame_lus = [lu for lu, data in lex_unit_data.items() if len(data['frames']) > 1]
    print(f"複数のフレームに関連する語彙ユニット数: {len(multi_frame_lus)}")
    
    # フレーム数でソートした場合のトップ5
    lu_with_frame_counts = [(lu, len(data['frames'])) for lu, data in lex_unit_data.items()]
    top_lus = sorted(lu_with_frame_counts, key=lambda x: x[1], reverse=True)[:5]
    
    print("\n最も多くのフレームに関連する語彙ユニットトップ5:")
    for lu, count in top_lus:
        print(f"{lu}: {count}フレーム")
    
    # 品詞ごとの分布
    pos_counts = {}
    for data in lex_unit_data.values():
        pos = data['lex_unit_pos']
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    print("\n品詞分布:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{pos}: {count}")
    
    # pandasでの分析例
    df_lu = pd.DataFrame(lu_records)
    
    # フレーム数のカラムを追加
    df_lu['frame_count'] = df_lu['frames'].apply(len)
    
    # フレーム数の分布
    print("\nフレーム数分布:")
    frame_count_dist = df_lu['frame_count'].value_counts().sort_index()
    for count, freq in frame_count_dist.items():
        print(f"{count}フレーム: {freq}語彙ユニット")
    
    # フレーム数が最も多い語彙ユニットの詳細を表示
    if not df_lu.empty:
        max_frames_lu = df_lu.loc[df_lu['frame_count'].idxmax()]
        print(f"\n最も多くのフレームに関連する語彙ユニット: {max_frames_lu['lex_unit_name']} ({max_frames_lu['lex_unit_pos']})")
        print(f"関連フレーム数: {max_frames_lu['frame_count']}")
        print("関連フレーム:")
        for i, frame in enumerate(max_frames_lu['frames'], 1):
            print(f"{i}. {frame['frame_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='データのルートディレクトリ')
    parser.add_argument('--language', type=str, default='en', help='生成する文の言語')


    args = parser.parse_args()
    main(args) 