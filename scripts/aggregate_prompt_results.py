#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数プロンプトの実験結果を集計するスクリプト

各モデルディレクトリ内のプロンプトごとの結果を集計し、
モデルごとの平均値と標準偏差を計算してTSV形式で出力する。

出力:
- モデルごとの aggregated_summary.tsv と aggregated_summary.txt を更新
- 全体の集計結果を prompt_aggregated_results.tsv として出力

使用例:
  # 全プロンプトで集計
  python scripts/aggregate_prompt_results.py
  
  # 特定のプロンプトのみ集計（v1, v3, v5のみ）
  python scripts/aggregate_prompt_results.py --prompts prompt_v1 prompt_v3 prompt_v5
  
  # 出力ファイル名を指定
  python scripts/aggregate_prompt_results.py --output output/custom_results.tsv
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import statistics

# CSVフィールドサイズ制限を増やす
csv.field_size_limit(sys.maxsize)


class PromptResult:
    """プロンプト結果を格納するクラス"""
    
    def __init__(self, prompt_name: str):
        self.prompt_name = prompt_name
        self.accuracy = 0.0
        self.filtered_accuracy = 0.0  # フィルタされた正答率
        self.correct = 0
        self.total = 0
        self.errors = 0
        self.extraction_failures = 0  # 抽出失敗数
        
    def __repr__(self):
        return f"PromptResult({self.prompt_name}, acc={self.accuracy:.1f}%, filtered_acc={self.filtered_accuracy:.1f}%)"


class ModelResults:
    """モデルの全プロンプト結果を格納するクラス"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_results: Dict[str, PromptResult] = {}
        
    def add_prompt_result(self, result: PromptResult):
        self.prompt_results[result.prompt_name] = result
        
    def get_accuracy_stats(self) -> Tuple[float, float]:
        """正答率の平均と標準偏差を計算"""
        accuracies = [r.accuracy for r in self.prompt_results.values()]
        if not accuracies:
            return 0.0, 0.0
        
        mean_acc = statistics.mean(accuracies)
        std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        return mean_acc, std_acc
    
    def get_filtered_accuracy_stats(self) -> Tuple[float, float]:
        """フィルタされた正答率の平均と標準偏差を計算"""
        filtered_accuracies = [r.filtered_accuracy for r in self.prompt_results.values()]
        if not filtered_accuracies:
            return 0.0, 0.0
        
        mean_acc = statistics.mean(filtered_accuracies)
        std_acc = statistics.stdev(filtered_accuracies) if len(filtered_accuracies) > 1 else 0.0
        return mean_acc, std_acc
    
    def get_extraction_failure_average(self) -> float:
        """抽出失敗数の平均を計算"""
        extraction_failures = [r.extraction_failures for r in self.prompt_results.values()]
        if not extraction_failures:
            return 0.0
        
        return statistics.mean(extraction_failures)


def parse_summary_txt(summary_path: Path) -> Tuple[float, int, int, int, int]:
    """summary.txtファイルから正答率、正解数、総問題数、エラー数、抽出失敗数を抽出"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 正答率を抽出 (例: "正答率: 17.9%")
        accuracy_match = re.search(r'正答率:\s*([0-9.]+)%', content)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        
        # 正解数を抽出 (例: "正解数: 196")
        correct_match = re.search(r'正解数:\s*(\d+)', content)
        correct = int(correct_match.group(1)) if correct_match else 0
        
        # 総問題数を抽出 (例: "総問題数: 1098")
        total_match = re.search(r'総問題数:\s*(\d+)', content)
        total = int(total_match.group(1)) if total_match else 0
        
        # エラー数を抽出 (例: "エラー数: 0")
        error_match = re.search(r'エラー数:\s*(\d+)', content)
        errors = int(error_match.group(1)) if error_match else 0
        
        # 抽出失敗数を抽出 (例: "抽出失敗: 0/1098 (0.0%)")
        extraction_failure_match = re.search(r'抽出失敗:\s*(\d+)/\d+', content)
        extraction_failures = int(extraction_failure_match.group(1)) if extraction_failure_match else 0
        
        return accuracy, correct, total, errors, extraction_failures
        
    except Exception as e:
        print(f"警告: {summary_path} の解析に失敗しました: {e}")
        return 0.0, 0, 0, 0, 0


def calculate_filtered_accuracy(result_tsv_path: Path) -> float:
    """result.tsvからフィルタされた正答率を計算"""
    try:
        with open(result_tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # ヘッダーをチェックしてグローバル評価列があるか確認
            fieldnames = reader.fieldnames or []
            has_global_eval = (("グローバル人間評価_回答一致度" in fieldnames and 
                               "グローバル人間評価_品質" in fieldnames) or
                              ("matched_answer_score" in fieldnames and 
                               "quality_score" in fieldnames))
            
            if not has_global_eval:
                # グローバル評価列がない場合は、フィルタなしの正答率を返す
                # （実際には通常の正答率と同じになる）
                return 0.0  # フィルタ機能が使用できないことを示すため0を返す
            
            filtered_correct = 0
            filtered_total = 0
            
            for row in reader:
                # グローバル評価のスコアを取得（複数の列名パターンに対応）
                try:
                    # 新しい形式の列名を優先
                    global_matched = (row.get("matched_answer_score", "") or 
                                    row.get("グローバル人間評価_回答一致度", ""))
                    global_quality = (row.get("quality_score", "") or 
                                    row.get("グローバル人間評価_品質", ""))
                    
                    global_matched_num = float(global_matched) if global_matched else 0
                    global_quality_num = float(global_quality) if global_quality else 0
                except (ValueError, TypeError):
                    continue
                
                # グローバル評価が両方2以上の場合のみカウント
                if global_matched_num >= 2 and global_quality_num >= 2:
                    filtered_total += 1
                    # scoreカラムで正解判定（1が正解、0が不正解）
                    score = row.get("score", "")
                    evaluation = row.get("evaluation", "")
                    
                    # scoreが1または evaluationが"正解"の場合を正解とする
                    if score == "1" or evaluation == "正解":
                        filtered_correct += 1
            
            if filtered_total == 0:
                return 0.0
            
            return (filtered_correct / filtered_total) * 100
            
    except Exception as e:
        print(f"警告: {result_tsv_path} のフィルタ正答率計算に失敗しました: {e}")
        return 0.0


def find_model_directories(base_dir: Path) -> List[Path]:
    """モデルディレクトリを検索"""
    model_dirs = []
    
    if not base_dir.exists():
        print(f"警告: {base_dir} が存在しません")
        return model_dirs
    
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # seedが含まれているディレクトリはスキップ
            if "_seed_" not in item.name:
                model_dirs.append(item)
    
    return sorted(model_dirs)


def find_prompt_directories(model_dir: Path, target_prompts: Optional[List[str]] = None) -> List[Path]:
    """モデルディレクトリ内のプロンプトディレクトリを検索
    
    Args:
        model_dir: モデルディレクトリのパス
        target_prompts: 対象とするプロンプト名のリスト（Noneの場合は全て）
                       例: ['prompt_v1', 'prompt_v3', 'prompt_v5']
    
    Returns:
        プロンプトディレクトリのリスト
    """
    prompt_dirs = []
    
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith('prompt_v'):
            # target_promptsが指定されている場合はフィルタリング
            if target_prompts is None or item.name in target_prompts:
                prompt_dirs.append(item)
    
    return sorted(prompt_dirs)


def collect_model_results(base_dir: Path, target_prompts: Optional[List[str]] = None) -> Dict[str, ModelResults]:
    """全モデルの結果を収集
    
    Args:
        base_dir: ベースディレクトリ
        target_prompts: 対象とするプロンプト名のリスト（Noneの場合は全て）
    """
    all_results = {}
    
    model_dirs = find_model_directories(base_dir)
    print(f"発見されたモデルディレクトリ数: {len(model_dirs)}")
    
    if target_prompts:
        print(f"対象プロンプト: {', '.join(target_prompts)}")
    else:
        print("対象プロンプト: 全て")
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n処理中: {model_name}")
        
        model_results = ModelResults(model_name)
        prompt_dirs = find_prompt_directories(model_dir, target_prompts)
        
        if not prompt_dirs:
            print(f"  警告: プロンプトディレクトリが見つかりません")
            continue
        
        print(f"  発見されたプロンプト数: {len(prompt_dirs)}")
        
        for prompt_dir in prompt_dirs:
            prompt_name = prompt_dir.name
            summary_path = prompt_dir / "summary.txt"
            result_path = prompt_dir / "result.tsv"
            
            if not summary_path.exists():
                print(f"    警告: {summary_path} が見つかりません")
                continue
            
            # summary.txtから基本統計を取得
            accuracy, correct, total, errors, extraction_failures = parse_summary_txt(summary_path)
            
            # result.tsvからフィルタされた正答率を計算
            filtered_accuracy = 0.0
            if result_path.exists():
                filtered_accuracy = calculate_filtered_accuracy(result_path)
            
            # PromptResultを作成
            prompt_result = PromptResult(prompt_name)
            prompt_result.accuracy = accuracy
            prompt_result.filtered_accuracy = filtered_accuracy
            prompt_result.correct = correct
            prompt_result.total = total
            prompt_result.errors = errors
            prompt_result.extraction_failures = extraction_failures
            
            model_results.add_prompt_result(prompt_result)
            print(f"    {prompt_name}: 正答率={accuracy:.1f}%, フィルタ正答率={filtered_accuracy:.1f}%")
        
        if model_results.prompt_results:
            all_results[model_name] = model_results
    
    return all_results


def update_model_aggregated_summary(model_dir: Path, model_results: ModelResults):
    """モデルディレクトリのaggregated_summary.tsvとaggregated_summary.txtを更新"""
    
    # aggregated_summary.tsvを更新
    tsv_path = model_dir / "aggregated_summary.tsv"
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["プロンプト", "正答率", "フィルタ正答率", "正解数", "総問題数", "エラー数", "抽出失敗数"])
        
        for prompt_name in sorted(model_results.prompt_results.keys()):
            result = model_results.prompt_results[prompt_name]
            writer.writerow([
                prompt_name,
                f"{result.accuracy:.1f}%",
                f"{result.filtered_accuracy:.1f}%",
                result.correct,
                result.total,
                result.errors,
                result.extraction_failures
            ])
    
    # aggregated_summary.txtを更新
    txt_path = model_dir / "aggregated_summary.txt"
    mean_acc, std_acc = model_results.get_accuracy_stats()
    mean_filtered_acc, std_filtered_acc = model_results.get_filtered_accuracy_stats()
    mean_extraction_failures = model_results.get_extraction_failure_average()
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("複数プロンプトの実験結果集計\n")
        f.write("=" * 70 + "\n\n")
        
        # プロンプト別結果
        f.write(f"{'プロンプト':<12} {'正答率':<8} {'フィルタ正答率':<12} {'正解数':<6} {'総問題数':<8} {'エラー数':<6} {'抽出失敗':<6}\n")
        for prompt_name in sorted(model_results.prompt_results.keys()):
            result = model_results.prompt_results[prompt_name]
            f.write(f"{prompt_name:<12} {result.accuracy:>6.1f}% {result.filtered_accuracy:>10.1f}% {result.correct:>6} {result.total:>8} {result.errors:>6} {result.extraction_failures:>6}\n")
        
        f.write("\n")
        f.write(f"平均正答率: {mean_acc:.1f}% (標準偏差: {std_acc:.1f}%)\n")
        f.write(f"平均フィルタ正答率: {mean_filtered_acc:.1f}% (標準偏差: {std_filtered_acc:.1f}%)\n")
        f.write(f"平均抽出失敗数: {mean_extraction_failures:.1f}\n")
    
    print(f"  更新: {tsv_path}")
    print(f"  更新: {txt_path}")


def save_overall_summary(all_results: Dict[str, ModelResults], output_path: Path):
    """全体の集計結果をTSV形式で保存"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "モデル名",
            "プロンプト数",
            "平均正答率",
            "正答率標準偏差",
            "平均フィルタ正答率",
            "フィルタ正答率標準偏差",
            "平均抽出失敗数"
        ])
        
        for model_name in sorted(all_results.keys()):
            model_results = all_results[model_name]
            mean_acc, std_acc = model_results.get_accuracy_stats()
            mean_filtered_acc, std_filtered_acc = model_results.get_filtered_accuracy_stats()
            mean_extraction_failures = model_results.get_extraction_failure_average()
            
            writer.writerow([
                model_name,
                len(model_results.prompt_results),
                f"{mean_acc:.3f}",
                f"{std_acc:.3f}",
                f"{mean_filtered_acc:.3f}",
                f"{std_filtered_acc:.3f}",
                f"{mean_extraction_failures:.1f}"
            ])


def main():
    parser = argparse.ArgumentParser(
        description='複数プロンプトの実験結果を集計するスクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全プロンプトで集計
  python scripts/aggregate_prompt_results.py
  
  # 特定のプロンプトのみ集計（v1, v3, v5のみ）
  python scripts/aggregate_prompt_results.py --prompts prompt_v1 prompt_v3 prompt_v5
  
  # 出力ファイル名を指定
  python scripts/aggregate_prompt_results.py --output output/custom_results.tsv
  
  # ベースディレクトリと出力先を両方指定
  python scripts/aggregate_prompt_results.py --base output/ja/four_choice_tsv/0 --output output/results.tsv
        """
    )
    
    parser.add_argument('--base', type=str, default='output/ja/four_choice_tsv/0',
                        help='ベースディレクトリ（デフォルト: output/ja/four_choice_tsv/0）')
    parser.add_argument('--output', type=str, default='output/prompt_aggregated_results.tsv',
                        help='出力ファイルパス（デフォルト: output/prompt_aggregated_results.tsv）')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='集計対象のプロンプト名（例: prompt_v1 prompt_v3 prompt_v5）。指定しない場合は全プロンプトを集計')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base)
    output_path = Path(args.output)
    target_prompts = args.prompts
    
    if not base_dir.exists():
        print(f"エラー: {base_dir} が存在しません")
        return
    
    print("=" * 60)
    print("複数プロンプト実験結果の集計を開始します")
    print("=" * 60)
    print(f"ベースディレクトリ: {base_dir}")
    print(f"出力ファイル: {output_path}")
    if target_prompts:
        print(f"対象プロンプト: {', '.join(target_prompts)}")
    else:
        print("対象プロンプト: 全て")
    
    # 全モデルの結果を収集
    all_results = collect_model_results(base_dir, target_prompts)
    
    if not all_results:
        print("エラー: 処理対象のモデル結果が見つかりませんでした")
        return
    
    print(f"\n処理されたモデル数: {len(all_results)}")
    
    # 各モデルのaggregated_summaryを更新
    print("\n" + "=" * 60)
    print("各モデルのaggregated_summaryを更新中...")
    print("=" * 60)
    
    for model_name, model_results in all_results.items():
        print(f"\n更新中: {model_name}")
        model_dir = base_dir / model_name
        update_model_aggregated_summary(model_dir, model_results)
        
        mean_acc, std_acc = model_results.get_accuracy_stats()
        mean_filtered_acc, std_filtered_acc = model_results.get_filtered_accuracy_stats()
        mean_extraction_failures = model_results.get_extraction_failure_average()
        print(f"  平均正答率: {mean_acc:.1f}% ± {std_acc:.1f}%")
        print(f"  平均フィルタ正答率: {mean_filtered_acc:.1f}% ± {std_filtered_acc:.1f}%")
        print(f"  平均抽出失敗数: {mean_extraction_failures:.1f}")
    
    # 全体の集計結果を保存
    print(f"\n全体の集計結果を保存中: {output_path}")
    save_overall_summary(all_results, output_path)
    
    print("\n" + "=" * 60)
    print("集計完了!")
    print("=" * 60)
    print(f"全体結果: {output_path}")
    print("各モデルの詳細結果: output/ja/four_choice_tsv/0/[モデル名]/aggregated_summary.*")


if __name__ == "__main__":
    main()
