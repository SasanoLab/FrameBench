# FrameBench: フレーム意味論に基づく意味理解ベンチマーク

<p align="center">
このリポジトリにはフレーム知識を活用したLLM向けの意味理解ベンチマークを構築し、LLMを評価するコード群が含まれます。

</p>
<p align="center">
  <a href="https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/Q4-13.pdf"><b>📄 論文</b></a> | 
  <a href="https://huggingface.co/datasets/cl-nagoya/jFrameBench"><b>🤗 Japanese FrameBench</b></a> |
  <a href="https://huggingface.co/datasets/cl-nagoya/FrameBench"><b>🤗 English FrameBench</b></a>
</p>
<p align="center">
  <b>日本語</b> | <a href="README.md">English</a>
</p>

> [!NOTE]
> ベンチマークの構築に利用するFrameNet由来のフレーム知識は、このリポジトリでは配布していません。
> 公開済みベンチマークの評価は日本語・英語に対応しています。

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

# vLLMでローカルモデルも評価する場合
uv sync --extra vllm
```

**ベンチマーク生成も実行する場合:**
```bash
uv sync --extra generation
```
すべてのパッケージを一括でインストールしたい場合は、
`uv sync --extra all`を実行してください。

### 3. **環境変数の設定**
利用するAPIに応じて、プロジェクトルートに`.env`を作成します。

```bash
cat << EOF > .env
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
EOF
```

使用しないAPIの行は省略できます。ローカルモデルをvLLMで評価する場合、APIキーは不要です。

## 🛠️ ベンチマークの構築
利用可能な生成モデルは[`src/generation/config.yaml`](src/generation/config.yaml)を参照してください。
以下のスクリプトはデフォルトで日本語（`LANGUAGE=ja`）を処理します。別言語を使用する場合は、
対応するFrameNetデータと生成プロンプトを用意し、`LANGUAGE`と`PROMPT_FILE`を指定してください。

### Step 1: フレーム知識の前処理

FrameNet XMLをJSONL形式に変換します。`data/<language>-framenet/raw_frame/`に、
喚起語をファイル名とするXMLファイル群を配置してください。

```bash
# 日本語（デフォルト）
bash scripts/step1.sh

# 言語やデータルートを変更
LANGUAGE=en DATA_ROOT=data bash scripts/step1.sh
```

**出力:**
- パース済みフレームデータ: `data/<language>-framenet/frames.jsonl`
- パース済み語彙単位データ: `data/<language>-framenet/lexical_units.jsonl`

Step 2では、これらに加えて`data/<language>-framenet/exemplars.jsonl`を入力として使用します。
このファイルはStep 1では生成されないため、別途用意してください。

### Step 2: 問題の生成

FrameNetの例文から文ペアと問題文を生成します。モデルを省略した場合は`gpt-4.1-mini`を使用します。

```bash
# 少数件で確認（デフォルト: NUM=10, NUM_PAIRS=2）
bash scripts/step2.sh gpt-4.1-mini

# 全件を生成
NUM= bash scripts/step2.sh gpt-4.1-mini

# 言語・件数・プロンプトを指定
LANGUAGE=en NUM=100 PROMPT_FILE=path/to/make_qa_en.toml \
  bash scripts/step2.sh gemini-3.1-pro-preview
```

**出力:**
- QA問題（文ペアを含む）: `data/<language>/<model>/step2/qa.jsonl`
- 中間結果・入力ログ・APIレスポンス: 同じ`step2/`ディレクトリ内

**オプション: 人手での記述修正（Step 2）**

生成されたQA問題の品質を向上させるため、人手によるテキスト修正が可能です。詳細は [`tools/README.md`](tools/README.md) を参照してください。

アノテーション後は、以下のコマンドでアノテーション結果を統合できます：

```bash
cd tools
uv run python postprocess/for_step2.py --input_file data/<language>/<model>/step2/qa.jsonl
```

修正されたQAデータは `data/<language>/<model>/step2+annotation/text_corrected_qa.jsonl` に保存されます。

### Step 3: 追加文の生成

問題の多様性を確保するため、Step 2の各問題に追加の文ペアを生成します。

```bash
# 少数件で確認（デフォルト: MAX_ITEMS=5）
bash scripts/step3.sh gpt-4.1-mini

# 全件を生成
MAX_ITEMS= bash scripts/step3.sh gpt-4.1-mini

# 人手修正版QAを入力にする
QA_FILE=data/ja/gpt-4.1-mini/step2+annotation/text_corrected_qa.jsonl \
  bash scripts/step3.sh gpt-4.1-mini
```

**入力:**
- デフォルト: `data/<language>/<model>/step2/qa.jsonl`
- `QA_FILE`で任意のQAファイルを指定可能

**出力:**
- 4文構造のQAデータ: `data/<language>/<model>/step3/qa.jsonl`
- バッチ別の入力ログ・APIレスポンス: 同じ`step3/`ディレクトリ内

**オプション: 人手アノテーション（Step 3）**

Step 3のデータについても、人手による評価が可能です。詳細は [`tools/README.md`](tools/README.md) を参照してください。

アノテーション後は、以下のコマンドでアノテーション結果を統合できます：

```bash
cd tools
uv run python postprocess/merge_eval_rounds.py --base_dir data/ja/gpt-4.1-mini/step3
```

統合されたQAデータは `data/<language>/<model>/step3/qa_annotated.jsonl` に保存されます。

## 🛠️ 評価

FrameBenchを用いてLLMを四択評価します。評価結果はプロンプトによって変化するため、
同梱する5つのプロンプトで評価し、平均値を利用することを推奨します。

モデルごとのバックエンド、thinking、サンプリング、構造化出力などの設定は
[`src/evaluation/model_profiles.yaml`](src/evaluation/model_profiles.yaml)で管理しています。
登録モデルはプロファイルに従ってOpenAI互換APIまたはvLLMを自動選択します。

### モデルの追加

`model_profiles.yaml`の`models`にプロファイルを追加すると、任意のモデルを評価できます。
たとえば`gpt-5`を登録すると、`gpt-5-2026-08-01`のように`gpt-5`から始まる
バージョン付きモデル名にも同じ設定が使われます。完全に一致する設定がある場合は、そちらを優先します。

```yaml
models:
  organization/thinking-model:
    backend: vllm
    supports_thinking: true
    thinking_stop: "</think>\n\n"
    thinking_sampling:
      temperature: 0.6
      top_p: 0.95
      top_k: 20
    answer_sampling:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
    trust_remote_code: true
```

`supports_thinking: true`はthinking対応であることを示します。自動的にthinkingが有効になるわけではなく、
通常は実行時に`--enable_thinking`を指定します。`reasoning_effort`対応モデルでは、
`--reasoning_effort medium`などを指定した場合もthinkingが有効になります。

主なフィールドは次のとおりです。

- `backend`: `openai`または`vllm`
- `supports_thinking`: thinking生成に対応するか
- `reasoning_effort`: 対応する推論強度（例: `[low, medium, high]`）
- `thinking_stop`: thinkingフェーズを終了する文字列
- `sampling`: no-thinking時のサンプリング設定
- `thinking_sampling`: thinkingフェーズのサンプリング設定
- `answer_sampling`: thinking後の回答フェーズのサンプリング設定
- `thinking_generation_mode`: thinking時の生成方式。省略時の`two_phase`は思考を制約なしで生成してから回答を`1/2/3/4`に制約し、`single_json`は1回の生成で思考から`{"answer":"N"}`まで出力
- `skip_special_tokens`: vLLMのデコード時に特殊トークンを除去するか
- `no_thinking_generation_prefill`: no-thinking時だけ生成開始位置へ補う文字列
- `trust_remote_code`: Hugging Faceモデルのカスタムコードを許可するか
- `supports_multimodal`: 画像などの入力に対応するか
- `chat_template`: tokenizerに適切なchat templateがない場合の上書き

OpenAI互換APIでは、必要に応じて`base_url`、`api_key_env`、
`structured_outputs`も設定します。

別のプロファイルファイルは`--model_profile_file path/to/profiles.yaml`で指定できます。

> [!CAUTION]
> Harmony形式など、回答本文より先に制御トークンを生成するモデルをno-thinkingで評価する場合、
> 生成開始直後の構造化出力制約と競合することがあります。チャットテンプレートと少数件の出力を確認し、
> 必要なモデルだけ`no_thinking_generation_prefill`を設定してください。
> thinking時は通常、思考フェーズを制約なしで生成してから回答制約を適用します。

### 評価方法

簡易スクリプトの形式は`MODEL [ja|en] [API_CONCURRENCY] [追加オプション...]`です。
このスクリプトは文A/Bを入れ替えた問題も評価する`--swap_statements`を有効にし、
テンソル並列数を4に設定します。

```bash
# OpenAI API・日本語
bash scripts/run_eval.sh gpt-5-nano

# OpenAI API・英語・API並列数20
bash scripts/run_eval.sh gpt-5-nano en 20

# vLLM・thinking有効（単一GPUの場合はTPを上書き）
bash scripts/run_eval.sh Qwen/Qwen3-4B ja 10 \
  --enable_thinking --tensor_parallel_size 1
```

Pythonを直接実行すると、各条件を個別に指定できます。

```bash
uv run python src/evaluation/eval_multi_prompts.py \
  --model gpt-5-nano \
  --language ja \
  --num 100 \
  --swap_statements \
  --prompt_files eval_prompt/ja/prompt_v1.txt
```

**主要オプション:**
- `--model`: 評価に使用するLLM
- `--language`: `ja`または`en`（デフォルト: `ja`）
- `--dataset`: HFデータセット名またはローカルJSONL。未指定時は日本語で`cl-nagoya/jFrameBench`、英語で`cl-nagoya/FrameBench`
- `--dataset_split`: Hugging Faceデータセットのsplit（デフォルト: `train`）
- `--num`: 展開後に評価する四択問題数（未指定時は全問）
- `--prompt_files`: 使用するプロンプトファイル（複数指定可、指定しない場合は`eval_prompt/<language>/`内の全ファイルを使用）
- `--output_dir`: 出力ディレクトリのベースパス（デフォルト: `output`）
- `--swap_statements`: 文A/Bを入れ替えたミラー問題を追加
- `--reasoning_effort`: 対応モデルの推論の深さ
- `--no_quality_filter`: 人手アノテーションに基づく品質フィルタを無効化

**vLLMモデル固有オプション:**
- `--enable_thinking`: 思考モードを有効にする
- `--tensor_parallel_size`: テンソル並列サイズ（デフォルト: 1）
- `--gpu_memory_utilization`: GPUメモリ使用率（デフォルト: 0.9）
- `--max_model_len`: 最大モデル長
- `--max_num_seqs`: 同時に処理する最大シーケンス数（未指定時はvLLMのデフォルト）
- `--max_tokens`: no-thinking時の最大生成トークン数
- `--thinking_max_tokens`: thinkingフェーズの最大生成トークン数
- `--answer_max_tokens`: thinking後の回答フェーズの最大生成トークン数

品質フィルタはデフォルトで有効です。人手アノテーションで正解一致・品質ともに
3名中2名以上が肯定した問題を集計対象とし、`summary.txt`と集計ファイルには
フィルタ後の正答率を記録します。全問題の詳細は`result.tsv`に残ります。

**出力:**

```text
output/<language>/<num_or_all>/<model><suffix>/
├── run_config.json
├── aggregated_summary.txt
├── aggregated_summary.tsv
└── <prompt_name>/
    ├── result.tsv
    ├── summary.txt
    └── params.json          # OpenAI互換APIのみ
```

`<suffix>`には、vLLMの`_thinking`/`_no_thinking`やAPIモデルの
`_reasoning_medium`など、実際の評価条件が反映されます。

## 📁 ファイル構成

```
framebench/
├── src/
│   ├── generation/               # ベンチマーク構築コード
│   │   ├── 1-1_frame_parse.py
│   │   ├── 1-2_lu_driven_edit.py
│   │   ├── 2_generate_frame_qa.py
│   │   ├── 3_gen_additional_sentences.py
│   │   ├── config.yaml           # 生成用モデル設定
│   │   ├── prompts/              # 生成プロンプト
│   │   └── utils/
│   └── evaluation/               # 評価コード
│       ├── eval_multi_prompts.py # 複数プロンプト評価
│       ├── eval_utils.py
│       └── model_profiles.yaml   # モデル別評価設定
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
├── data/                         # 入力・生成データ
│   ├── <language>-framenet/      # FrameNet由来データ
│   │   └── raw_frame/
│   └── <language>/<model>/
│       ├── step2/qa.jsonl
│       └── step3/qa.jsonl
├── output/                       # 評価結果出力
│   └── <language>/
│       └── <num_or_all>/
│           └── <model><suffix>/
│               ├── run_config.json
│               ├── <prompt_name>/
│               │   ├── result.tsv
│               │   ├── summary.txt
│               │   └── params.json  # OpenAI互換APIのみ
│               ├── aggregated_summary.txt
│               └── aggregated_summary.tsv
├── tools/                        # アノテーションツール
├── pyproject.toml                # プロジェクト設定
├── README.md                     # 英語版README
└── README_JA.md                  # このファイル
```
