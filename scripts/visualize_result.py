"""output/prompt_aggregated_results.tsvの結果を可視化する
平均フィルタ正答率がメインの値で、横にモデル名、高さが平均フィルタ正答率となるような棒グラフにする
モデル名は推論時の思考モード設定をトリミングした文字列にして、_no_thinkingモデルとそれ以外で色分けする
利用するモデル一覧をこのスクリプトから指定する
"""
import matplotlib
# バックエンドがディスプレイを必要としない場合に対応
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== グラフ表示設定 ==========
# エラーバー（標準偏差）の表示
SHOW_ERROR_BARS = False

# thinkingモデルの網掛け表示
SHOW_HATCHING = True

# 数値ラベルの表示
SHOW_VALUE_LABELS = True

# 数値ラベルの背景表示
SHOW_VALUE_BACKGROUND = False

# 人間のデータを追加
SHOW_HUMAN = True
HUMAN_ACCURACY = 95.2

# グラフサイズ
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 6

# フォントサイズ
LABEL_FONTSIZE = 14
VALUE_FONTSIZE = 16
TICK_FONTSIZE = 14
# ===================================

# 利用するモデル一覧
TARGET_MODELS = [
    "gpt-5-nano_reasoning_medium_structured",
    "gpt-5_reasoning_medium_structured",
    # "openai_gpt-oss-20b_thinking_reasoning_low",
    "openai_gpt-oss-20b_thinking_reasoning_medium",
    # "openai_gpt-oss-20b_thinking_reasoning_high",
    "google_gemma-3-1b-it_no_thinking",
    "google_gemma-3-4b-it_no_thinking",
    "google_gemma-3-12b-it_no_thinking",
    "google_gemma-3-27b-it_no_thinking",
    "Qwen_Qwen3-0.6B_no_thinking",
    "Qwen_Qwen3-1.7B_no_thinking",
    "Qwen_Qwen3-4B_no_thinking",
    "Qwen_Qwen3-8B_no_thinking",
    "Qwen_Qwen3-14B_no_thinking",
    "Qwen_Qwen3-32B_no_thinking",
    "Qwen_Qwen3-0.6B_thinking",
    "Qwen_Qwen3-1.7B_thinking",
    "Qwen_Qwen3-4B_thinking",
    "Qwen_Qwen3-8B_thinking",
    "Qwen_Qwen3-14B_thinking",
    "Qwen_Qwen3-32B_thinking",
    "llm-jp_llm-jp-3.1-1.8b-instruct4_no_thinking",
    "llm-jp_llm-jp-3.1-13b-instruct4_no_thinking",
]


def trim_model_name(model_name: str) -> str:
    """モデル名から推論時の思考モード設定をトリミングする"""
    # _no_thinking, _thinking, _reasoning_* などを削除
    trimmed = model_name
    
    # パターンに一致する部分を削除
    patterns = [
        "_no_thinking",
        "_thinking_reasoning_high",
        "_thinking_reasoning_medium",
        "_thinking_reasoning_low",
        "_thinking",
        "_reasoning_medium_structured",
        "_reasoning_high_structured",
        "_reasoning_low_structured",
    ]
    
    for pattern in patterns:
        if pattern in trimmed:
            trimmed = trimmed.replace(pattern, "")
            break
    if "gpt-5" in trimmed:
        return trimmed
    if "_" in trimmed:
        trimmed = trimmed.split("_")[1]
    trimmed = trimmed.replace("b", "B")
    if len(trimmed.split("-")) > 2:
        if not trimmed.endswith("B"):
            trimmed = "-".join(trimmed.split("-")[:-1])
    
    return trimmed


def is_no_thinking_model(model_name: str) -> bool:
    """モデルが_no_thinkingかどうかを判定"""
    return "_no_thinking" in model_name


def get_model_series(model_name: str) -> str:
    """モデル名からシリーズを判定"""
    if model_name == "Human":
        return "Human"
    elif "gpt-5" in model_name:
        return "GPT-5"
    elif "gpt-oss" in model_name:
        return "GPT-OSS"
    elif "Qwen" in model_name:
        return "Qwen"
    elif "gemma" in model_name:
        return "Gemma"
    elif "llm-jp" in model_name:
        return "LLM-JP"
    else:
        return "Other"


def main():
    # TSVファイルのパス
    script_dir = Path(__file__).parent
    tsv_path = script_dir.parent / "output" / "prompt_aggregated_results.tsv"
    
    # データ読み込み
    df = pd.read_csv(tsv_path, sep="\t")
    
    # 対象モデルのみフィルタリング
    df_filtered = df[df["モデル名"].isin(TARGET_MODELS)].copy()
    
    # トリミングしたモデル名を追加
    df_filtered["表示モデル名"] = df_filtered["モデル名"].apply(trim_model_name)
    
    # no_thinkingかどうかのフラグを追加
    df_filtered["is_no_thinking"] = df_filtered["モデル名"].apply(is_no_thinking_model)
    
    # モデルシリーズを追加
    df_filtered["series"] = df_filtered["モデル名"].apply(get_model_series)
    
    # TARGET_MODELSの順番でソート
    model_order = {model: i for i, model in enumerate(TARGET_MODELS)}
    df_filtered["model_order"] = df_filtered["モデル名"].map(model_order)
    df_filtered = df_filtered.sort_values("model_order")
    
    # 人間の正答率を追加
    if SHOW_HUMAN:
        human_row = pd.DataFrame({
            "モデル名": ["Human"],
            "表示モデル名": ["人間"],
            "平均フィルタ正答率": [HUMAN_ACCURACY],
            "フィルタ正答率標準偏差": [0.0],  # 人間の標準偏差
            "is_no_thinking": [False],
            "series": ["Human"],  # シリーズを明示的に設定
            "model_order": [-1]  # 一番左に配置
        })
        df_filtered = pd.concat([df_filtered, human_row], ignore_index=True)
        df_filtered = df_filtered.sort_values("model_order")
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # シリーズごとの色設定
    series_colors = {
        "Human": "#ffc184",
        "GPT-5": "#9eff9e",      # 赤系
        "GPT-OSS": "#ceff9e",    # 青緑系
        "Qwen": "#9e9eff",       # 薄い青緑系
        "Gemma": "#9eceff",      # ピンク系
        "LLM-JP": "#ff9ece",     # 紫系
        "Other": "#CCCCCC"       # グレー
    }
    
    # 棒グラフ作成（thinkingモデルには網掛けを追加）
    bars = []
    for i, (idx, row) in enumerate(df_filtered.iterrows()):
        # シリーズに応じた色を取得
        color = series_colors.get(row["series"], series_colors["Other"])
        
        # 網掛けパターンの決定
        hatch_pattern = None
        if SHOW_HATCHING and not row["is_no_thinking"] and row["表示モデル名"] != "人間":
            hatch_pattern = "///"
        
        # バー作成
        bar = ax.bar(
            i,
            row["平均フィルタ正答率"],
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch_pattern
        )
        bars.append(bar)
    
    # エラーバーを追加（標準偏差）
    if SHOW_ERROR_BARS:
        ax.errorbar(
            range(len(df_filtered)),
            df_filtered["平均フィルタ正答率"],
            yerr=df_filtered["フィルタ正答率標準偏差"],
            fmt='none',  # マーカーなし
            ecolor='black',  # エラーバーの色
            elinewidth=1,  # エラーバーの線の太さ
            capsize=5,  # エラーバーの先端のキャップサイズ
            capthick=2,  # キャップの太さ
            zorder=2  # バーの上、数値の下
        )
    
    # 軸設定
    # ax.set_xlabel("モデル名", fontsize=18, fontweight="bold")
    # ax.set_ylabel("平均フィルタ正答率 (%)", fontsize=18, fontweight="bold")
    # ax.set_title("モデル別平均フィルタ正答率の比較", fontsize=22, fontweight="bold", pad=20)
    
    # x軸のラベル設定
    ax.set_xticks(range(len(df_filtered)))
    ax.set_xticklabels(df_filtered["表示モデル名"], rotation=45, ha="right", fontsize=LABEL_FONTSIZE)
    
    # y軸の目盛りラベルサイズ設定
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    
    # y軸の範囲設定
    ax.set_ylim(0, 100)
    
    # グリッド追加
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # 凡例は削除
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor=base_color, alpha=0.8, edgecolor="black", label="no_thinking"),
    #     Patch(facecolor=base_color, alpha=0.8, edgecolor="black", hatch="///", label="thinking/reasoning"),
    #     Patch(facecolor=human_color, alpha=0.8, edgecolor="black", label="人間")
    # ]
    # ax.legend(handles=legend_elements, loc="upper left", fontsize=14)
    
    # 各バーに値を表示
    if SHOW_VALUE_LABELS:
        for i, (idx, row) in enumerate(df_filtered.iterrows()):
            value = row["平均フィルタ正答率"]
            
            # 背景の設定
            bbox_props = None
            if SHOW_VALUE_BACKGROUND:
                if row["表示モデル名"] == "人間":
                    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
                else:
                    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.5)
            
            # 数値表示
            ax.text(
                i,
                value - 3,
                f"{value:.1f}",
                ha="center",
                va="top",
                fontsize=VALUE_FONTSIZE,
                color="black",
                zorder=10,
                bbox=bbox_props
            )
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    output_path = script_dir.parent / "output" / "model_comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"グラフを保存しました: {output_path}")


if __name__ == "__main__":
    main()