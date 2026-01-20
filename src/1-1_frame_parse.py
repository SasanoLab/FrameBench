"""
<data_root>/<language>-framenet/raw_frame
にあるXMLファイル群をパースして、
<data_root>/<language>-framenet/frames.jsonl
にJSONL形式で保存する。

Usage:
python src/1-1_frame_parse.py --data_root <data_root> --language <language>
"""

import xml.etree.ElementTree as ET
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import argparse


def clean_definition(text: str) -> str:
    """
    定義文から例文とHTMLタグを削除する
    
    Args:
        text: 元の定義文
    
    Returns:
        クリーンアップされた定義文
    """
    if not text:
        return ""
    
    # 1. 先に例文部分（<ex>...</ex>）を削除
    # 正規表現で例文タグとその内容を削除（改行を含む場合も考慮）
    text = re.sub(r'<ex>.*?</ex>', '', text, flags=re.DOTALL)
    
    # 2. その他のXMLタグも削除
    text = re.sub(r'<def-root>', '', text)
    text = re.sub(r'</def-root>', '', text)
    text = re.sub(r'<fen>.*?</fen>', '', text, flags=re.DOTALL)
    text = re.sub(r'<fex[^>]*>.*?</fex>', '', text, flags=re.DOTALL)
    text = re.sub(r'<t>.*?</t>', '', text, flags=re.DOTALL)
    text = re.sub(r'<target>.*?</target>', '', text, flags=re.DOTALL)
    
    # 3. 残りのHTMLタグを削除
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. 余分な空白や改行を整理
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # デバッグ用：処理後のテキストを表示
    # print(f"After cleaning: {text[:200]}...")
    
    return text


def parse_xml_file(filepath: str | Path) -> Dict[str, Any]:
    """
    XMLファイルをパースし、フレーム名、定義、Coreタイプのフレーム要素、lexUnitを抽出する

    Args:
        filepath: XMLファイルのパス

    Returns:
        抽出された情報を含む辞書
    """
    try:
        # XMLファイルをパース
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # フレーム名を取得
        frame_name = root.get('name')
        
        # 名前空間を取得
        namespace = re.match(r'{.*}', root.tag)
        ns = namespace.group(0) if namespace else ''
        
        # 定義を取得
        definition_elem = root.find(f'{ns}definition')
        definition_text = definition_elem.text if definition_elem is not None and definition_elem.text else ""
        
        # HTMLタグを削除（フレーム自体の定義は元の実装を使用）
        definition_text = re.sub(r'<[^>]+>', '', definition_text)
        
        # Core typeのフレーム要素を抽出
        core_frame_elements = []
        for fe in root.findall(f'{ns}FE'):
            core_type = fe.get('coreType')
            if core_type == 'Core':
                name = fe.get('name')
                
                # 定義を取得
                fe_definition_elem = fe.find(f'{ns}definition')
                fe_definition = fe_definition_elem.text if fe_definition_elem is not None and fe_definition_elem.text else ""
                fe_definition = clean_definition(fe_definition)
                
                core_frame_elements.append({
                    'name': name,
                    'definition': fe_definition
                })
        
        # lexUnitを抽出
        lex_units = []
        for lu in root.findall(f'{ns}lexUnit'):
            lu_name = lu.get('name')
            lu_pos = lu.get('POS')  # 品詞情報も取得
            
            # sentenceCount要素からannotationの数を取得
            sentence_count_elem = lu.find(f'{ns}sentenceCount')
            if sentence_count_elem is not None:
                annotated_count = sentence_count_elem.get('annotated')
                if annotated_count is None or int(annotated_count) < 1:
                    continue  # annotationが1未満の場合はスキップ
                annotation_count = int(annotated_count)
            else:
                continue  # sentenceCount要素がない場合はスキップ
            
            # 定義を取得
            lu_def_elem = lu.find(f'{ns}definition')
            lu_definition = lu_def_elem.text if lu_def_elem is not None and lu_def_elem.text else ""
            
            lex_units.append({
                'name': lu_name,
                'pos': lu_pos,
                'definition': lu_definition,
                'annotation_count': annotation_count
            })
        
        return {
            'frame_name': frame_name,
            'definition': definition_text,
            'core_frame_elements': core_frame_elements,
            'lex_units': lex_units
        }
    except (ET.ParseError, FileNotFoundError, PermissionError, OSError, ValueError) as e:
        print(f"エラー: {filepath}の処理中に問題が発生しました: {e}")
        return {
            'frame_name': Path(filepath).stem,
            'definition': '',
            'core_frame_elements': [],
            'lex_units': []
        }


def process_directory(directory_path: str | Path) -> List[Dict[str, Any]]:
    """
    指定されたディレクトリ内のすべてのXMLファイルを処理する

    Args:
        directory_path: XMLファイルが格納されているディレクトリパス

    Returns:
        すべてのフレーム情報を含むリスト
    """
    # ディレクトリ内のすべてのXMLファイルを取得
    xml_files = list(Path(directory_path).glob('*.xml'))
    
    # マルチスレッドでファイルを処理
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(parse_xml_file, xml_files))
    
    return results


def main(args):
    """
    メイン処理：FrameNetのXMLファイルをパースして構造化データに変換する
    
    data/raw_fndata ディレクトリ内のすべてのXMLファイルを処理し、
    フレーム情報を抽出して data/frames.jsonl に保存する。
    処理完了後に統計情報も表示する。
    """
    # データディレクトリのパス
    data_dir = Path(args.data_root) / f'{args.language}-framenet' / 'raw_frame'
    output_path = Path(args.data_root) / f'{args.language}-framenet' / 'frames.jsonl'
    
    # すべてのXMLファイルを処理
    frames = process_directory(data_dir)
    
    # フレーム情報をDataFrameに変換
    df_frames = pd.DataFrame([
        {
            'frame_name': frame['frame_name'],
            'definition': frame['definition'],
            'core_elements': ', '.join([fe['name'] for fe in frame['core_frame_elements']]),
            'core_elements_definitions': '; '.join([f"{fe['name']}: {fe['definition']}" for fe in frame['core_frame_elements']]),
            'lex_units': [lu['name'] for lu in frame['lex_units']],
            'lex_pos': [ lu['pos'] for lu in frame['lex_units']]
        }
        for frame in frames
    ])
    
    pd.DataFrame(frames).to_json(output_path, orient='records', force_ascii=False, lines=True)
    
    print(f"saved to {output_path}")
    
    # データの統計情報を表示
    print("\n統計情報:")
    print(f"フレーム数: {len(frames)}")
    print(f"Coreタイプのフレーム要素を持つフレーム数: {sum(1 for frame in frames if frame['core_frame_elements'])}")
    print(f"lexUnitを持つフレーム数: {sum(1 for frame in frames if frame['lex_units'])}")
    
    # トップ5のフレームと要素数の表示
    print("\nトップ5のフレーム (Core要素数順):")
    df_frames['core_count'] = df_frames['core_elements'].apply(lambda x: len(x.split(', ')) if x else 0)
    print(df_frames.sort_values('core_count', ascending=False).head(5)[['frame_name', 'core_count']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='データのルートディレクトリ')
    parser.add_argument('--language', type=str, default='en', help='生成する文の言語')

    args = parser.parse_args()
    main(args) 