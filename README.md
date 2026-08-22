# FrameBench: フレーム意味論に基づく意味理解ベンチマーク

<p align="center">
このリポジトリにはフレーム知識を活用したLLM向けの意味理解ベンチマークを構築し、LLMを評価するコード群が含まれます。

</p>
<p align="center">
  <a href="https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/Q4-13.pdf"><b>📄 論文</b></a> | 
  <a href="https://huggingface.co/datasets/cl-nagoya/jFrameBench"><b>🤗 日本語FrameBench</b></a> 
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
cat << EOF > .env
OPENAI_API_KEY=your_api_key_here
EOF
```

## 🛠️ ベンチマークの構築
利用可能モデルは`src/generation/config.yaml`を参照ください。

### Step 1: フレーム知識の前処理

XMLファイルをパースしてJSONL形式に変換します。
`data/<language>-framenet/raw_frame`配下に、喚起語がファイル名となっているXMLファイル群を配置してください。

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

### 評価方法
スクリプトの実行
```bash
bash scripts/run_eval.sh gpt-5-nano
```
もしくは
```bash
uv run python src/evaluation/eval_multi_prompts.py \
    --model gpt-4o \
    --language ja \
    --num 100 \
    --prompt_files eval_prompt/ja/prompt_v1.txt
```
**主要オプション:**
- `--model`: 評価に使用するLLM
- `--dataset`: 評価対象のデータセット（未指定時は`--language ja`で`cl-nagoya/jFrameBench`、`--language en`で`cl-nagoya/FrameBench`を使用。自分で構築したものを利用する場合はjsonlファイルへのパスを指定）
- `--num`: 評価する問題数（指定しない場合は全問）
- `--prompt_files`: 使用するプロンプトファイル（複数指定可、指定しない場合は`eval_prompt/<language>/`内の全ファイルを使用）
- `--output_dir`: 出力ディレクトリのベースパス（デフォルト: `output`）
- `--language`: 評価対象言語（デフォルト: `ja`）

**OpenAIモデル固有オプション:**
- `--reasoning_effort`: 推論の深さ（`low`, `medium`, `high`, `minimal`, `none`）

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

## 📁 ファイル構成

```
framebench/
├── src/
│   ├── generation/               # ベンチマーク構築コード
│   │   ├── 1-1_frame_parse.py    # FrameNet XMLパーサー
│   │   ├── 1-2_lu_driven_edit.py # 語彙単位編集
│   │   ├── 2_generate_frame_qa.py    # QA生成
│   │   ├── 3_gen_additional_sentences.py  # 追加文生成
│   │   ├── config.yaml           # 生成用モデル設定
│   │   ├── utils/                # 生成系ユーティリティ
│   │   │   ├── llm.py            # LLMラッパー
│   │   │   └── utils.py          # 共通ユーティリティ
│   │   └── prompts/              # プロンプトテンプレート
│   │       ├── __init__.py
│   │       ├── gen_other_choice_ja.toml  # 選択肢生成プロンプト
│   │       └── make_qa_ja.toml   # QA生成プロンプト（日本語）
│   └── evaluation/               # 評価コード
│       ├── eval_multi_prompts.py # 複数プロンプト評価
│       └── eval_utils.py         # 評価ユーティリティ
├── scripts/                      # 実行スクリプト
│   ├── step1.sh                  # Step 1実行
│   ├── step2.sh                  # Step 2実行
│   ├── step3.sh                  # Step 3実行
│   └── run_eval.sh               # 評価実行
├── eval_prompt/                  # 評価用プロンプト
│   ├── ja/
│   │   ├── prompt_v1.txt
│   │   └── ...
│   └── en/
│       ├── prompt_v1.txt
│       └── ...
├── data/                         # データディレクトリ
│   ├── <language>-framenet/      # フレームデータ
│   │   └── raw_frame/            # 生FrameNet XMLファイル
│   └── ja/                       # 生成データ
│       └── gpt5/
│           └── qa.jsonl          # jFrameBenchデータ
├── output/                       # 評価結果出力
│   └── <language>/
│       └── <num>/                # 評価問題数（100, allなど）
│           └── <model>_<suffix>/
│               ├── <prompt_name>/
│               │   ├── result.tsv
│               │   ├── summary.txt
│               │   └── params.json
│               ├── aggregated_summary.txt
│               └── aggregated_summary.tsv
├── tools/                        # アノテーションツール（別README参照）
│   ├── app.py
│   ├── src/
│   │   ├── common.py
│   │   ├── dataset_manager.py
│   │   ├── evaluation_app.py
│   │   ├── text_correction_app.py
│   │   ├── dataset_config.yaml
│   │   ├── evaluation_criteria.yaml
│   │   ├── auth_config.json
│   │   └── user_assignment.json
│   ├── postprocess/
│   │   ├── for_step2.py
│   │   ├── for_2nd_annotation.py
│   │   ├── for_human_eval.py
│   │   └── merge_eval_rounds.py
│   ├── README.md
│   ├── pyproject.toml
│   └── uv.lock
├── pyproject.toml                # プロジェクト設定
├── uv.lock                       # 依存関係ロック
└── README.md                     # このファイル
```