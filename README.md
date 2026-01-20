# FrameBench: フレーム意味論に基づく意味理解ベンチマーク

<p align="center">
このリポジトリにはフレーム知識を活用したLLM向けの意味理解ベンチマークを構築し、LLMを評価するコード群が含まれます。

</p>
<p align="center">
  <a href="#"><b>📄 論文</b></a> | 
  <a href="#"><b>🤗 日本語FrameBench</b></a> 
</p>
*ベンチマークの構築に利用するフレーム知識は配布していません。

*2026年3月11日現在、日本語にのみ対応しています。

## 📋 概要
このプロジェクトは以下のステップで構成されています：

- ベンチマークの構築
  - Step 1: フレーム知識の前処理
  - Step 2: QA問題の生成
  - Step 3: 追加文の生成
- 評価の実行

## 🚀 環境構築

### 1. **uvのインストール**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. **依存関係のインストール**

**評価のみ実行する場合:**
```bash
uv sync
# vLLMを使用してローカルLLMで評価する場合は --extra vllmをつける
```

**ベンチマーク生成も実行する場合:**
```bash
uv sync --extra generation
```
すべてのパッケージを一括でインストールしたい場合は、
`uv sync --extra all`を実行してください。

### 3. **環境変数の設定**
OpenAI APIを使用する場合は、`.env`ファイルを作成：
```bash
cp .env.example .env
# .envファイルを編集してAPIキーを設定
OPENAI_API_KEY=your_api_key_here
```

## 🛠️ ベンチマークの構築
利用可能モデルは'config.yaml'を参照ください。

### Step 1: フレーム知識の前処理

XMLファイルをパースしてJSONL形式に変換します。

```bash
bash scripts/step1.sh
```

**出力:**
- パース済みフレームデータ: `data/<language>-framenet/frames.jsonl`
- パース済み語彙単位データ: `data/<language>-framenet/lexical_units.jsonl` 
- 例文データ: `data/<language>-framenet/exemplars.jsonl`

### Step 2: 問題の生成

文ペアと問題文を生成します。

```bash
bash scripts/step2.sh <model>
```

**出力:**
- 生成されたQA問題: `data/<language>/<model>/step2/qa.jsonl`
- 生成された文ペア: `data/<language>/<model>/step2/sentence_pair.jsonl`

**オプション: 人手での記述修正（Step 2）**

生成されたQA問題の品質を向上させるため、人手によるテキスト修正が可能です。詳細は [`tools/README.md`](tools/README.md) を参照してください。

アノテーション後は、以下のコマンドでアノテーション結果を統合できます：

```bash
cd tools
uv run python postprocess/for_step2.py --input_file data/<language>/<model>/step2/qa.jsonl
```

修正されたQAデータは `data/<language>/<model>/step2+annotation/text_corrected_qa.jsonl` に保存されます。

### Step 3: 追加文の生成

問題の多様性を確保するため、追加の文ペアを生成します。

```bash
bash scripts/step3.sh <model>
```

**入力:**
- Step 2で生成されたQAファイル


**出力:**
- `data/<language>/<model>/step3/qa.jsonl` - 4文構造のQAデータ

**オプション: 人手アノテーション（Step 3）**

Step 3のデータについても、人手による評価が可能です。詳細は [`tools/README.md`](tools/README.md) を参照してください。

アノテーション後は、以下のコマンドでアノテーション結果を統合できます：

```bash
cd tools
uv run python postprocess/merge_eval_rounds.py --base_dir data/ja/gpt-4.1-mini/step3
```

統合されたQAデータは `data/<language>/<model>/step3/qa_annotated.jsonl` に保存されます。

## 🛠️ 評価

FrameBenchを用いてLLMを評価します。
評価結果はプロンプトによって変化するため、5つのプロンプトで評価し、平均した結果を利用することを推奨します。
<!-- TODO: HF Datasetからの評価 -->

#### 単一プロンプトでの評価

```bash
uv run python src/eval_multi_prompts.py \
    --model gpt-4o \
    --input_file data/ja/gpt-4.1-mini/step3/qa_annotated.jsonl \
    --num 100 \
    --prompt_files eval_prompt/prompt_v1.txt
```

#### 複数プロンプトでの評価

```bash
uv run python src/eval_multi_prompts.py \
    --model gpt-4o \
    --input_file data/ja/gpt-4.1-mini/step3/qa_annotated.jsonl \
    --num 100 \
    --prompt_files eval_prompt/prompt_v1.txt eval_prompt/prompt_v2.txt eval_prompt/prompt_v3.txt
```

**主要オプション:**
- `--model`: 評価に使用するLLMモデル
- `--input_file`: 評価対象のJSONLファイル（必須）
- `--num`: 評価する問題数（指定しない場合は全問）
- `--prompt_files`: 使用するプロンプトファイル（複数指定可、指定しない場合は`eval_prompt/`内の全ファイルを使用）
- `--output_dir`: 出力ディレクトリのベースパス（デフォルト: `output`）
- `--language`: 評価対象言語（デフォルト: `ja`）

**OpenAIモデル固有オプション:**
- `--reasoning_effort`: 推論の深さ（`low`, `medium`, `high`, `minimal`, `none`）
- `--use_structured_outputs`: Structured Outputsを使用

**vLLMモデル固有オプション:**
- `--enable_thinking`: 思考モードを有効にする
- `--tensor_parallel_size`: テンソル並列サイズ（デフォルト: 1）
- `--gpu_memory_utilization`: GPUメモリ使用率（デフォルト: 0.9）
- `--max_model_len`: 最大モデル長
- `--max_num_seqs`: 同時に処理する最大シーケンス数（デフォルト: 256）

**出力:**
- `output/<language>/four_choice_tsv/<num>/<model>_<suffix>/<prompt_name>/` - 各プロンプトの評価結果
  - `result.tsv` - 詳細な評価結果
  - `summary.txt` - 統計サマリー
  - `params.json` - 使用されたパラメータ
- `output/<language>/four_choice_tsv/<num>/<model>_<suffix>/aggregated_summary.txt` - 全プロンプトの集計結果
- `output/<language>/four_choice_tsv/<num>/<model>_<suffix>/aggregated_summary.tsv` - 集計結果（TSV形式）

#### バッチ評価スクリプト

複数プロンプト×複数モデルの実験を一括実行する場合：

```bash
bash scripts/run_multi_prompt_experiment.sh
```

スクリプト内でモデルとプロンプトを指定して実行します。

## 📁 ファイル構成

```
framebench/
├── src/                          # ソースコード
│   ├── 1-1_frame_parse.py        # FrameNet XMLパーサー
│   ├── 1-2_lu_driven_edit.py    # 語彙単位編集
│   ├── 2_generate_frame_qa.py    # QA生成
│   ├── 3_gen_additional_sentences.py  # 追加文生成
│   ├── eval_multi_prompts.py    # 複数プロンプト評価
│   ├── eval_common_four_choice.py  # 評価共通モジュール
│   ├── utils/                    # ユーティリティ
│   │   ├── llm.py                # LLMラッパー
│   │   └── utils.py              # 共通ユーティリティ
│   └── prompts/                  # プロンプトテンプレート
├── scripts/                      # 実行スクリプト
│   ├── step1.sh                 # Step 1実行
│   ├── step2.sh                 # Step 2実行
│   ├── step3.sh                 # Step 3実行
│   └── run_multi_prompt_experiment.sh  # バッチ評価
├── eval_prompt/                  # 評価用プロンプト
│   ├── prompt_v1.txt
│   ├── prompt_v2.txt
│   └── ...
├── data/                         # データディレクトリ
│   ├── <language>-framenet/     # フレームデータ
│   │   ├── raw_frame/           # 生FrameNet XMLファイル
│   │   ├── frames.jsonl         # パース済みフレームデータ
│   │   ├── lexical_units.jsonl  # パース済み語彙単位データ
│   │   └── exemplars.jsonl      # 例文データ
│   └── <language>/              # 生成データ
│       └── <model>/
│           ├── step2/           # Step 2の出力
│           ├── step2+annotation/  # Step 2アノテーション結果
│           └── step3/           # Step 3の出力
├── output/                       # 評価結果出力
│   └── <language>/
│       └── four_choice_tsv/
│           └── <num>/
│               └── <model>_<suffix>/
│                   ├── <prompt_name>/  # 各プロンプトの結果
│                   └── aggregated_summary.txt  # 集計結果
├── tools/                        # アノテーションツール（別README参照）
├── pyproject.toml               # プロジェクト設定
├── uv.lock                      # 依存関係ロック
└── README.md                    # このファイル
```

## 📄 ライセンス

このプロジェクトは研究目的で開発されました。
