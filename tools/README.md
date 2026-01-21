---
title: Frame QA アノテーションツール
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: apache2.0
python_version: 3.12
short_description: Frame QAデータセットの質問品質を評価・修正するためのツール群
---

# Frame QA アノテーションツール

Frame QAデータセットの質問品質を評価・修正するためのツール群です。GradioベースのWebインターフェースを提供し、人手によるアノテーション作業を効率化します。

## 📋 概要

このツールは、FrameBenchで生成されたQAデータに対して以下の作業をサポートします：

1. **テキスト修正** - 生成された質問文や文の誤りを修正
2. **評価** - 質問の品質や正解の妥当性を評価

## 🚀 セットアップ

### 必要環境
- Python 3.12以上
- uv (Pythonパッケージマネージャー)

### インストール手順

1. **依存関係のインストール**
```bash
cd tools
uv sync
```

2. **環境変数の設定（オプション）**
認証情報とユーザー割当を設定します（後述）。

## 🛠️ 使用方法

### アプリケーションの起動

```bash
cd tools
uv run python app.py
```

ブラウザで `http://localhost:7860` にアクセスします。

### 基本的なワークフロー

1. **ログイン**: 認証情報を入力してログイン
2. **アプリケーション選択**: 2つのアプリケーションから選択
   - **テキスト修正アプリ**: 質問文や文の誤りを修正
   - **評価アプリ**: 質問の品質や正解を評価
3. **アノテーション実行**: データを確認し、修正・評価を実行
4. **結果保存**: アノテーション結果を保存（自動保存）
5. **ダウンロード**: 完了したアノテーション結果をダウンロード

## 📝 アノテーションの流れ

### Step 2: QA問題のアノテーション

生成されたQA問題（`data/<language>/<model>/step2/qa.jsonl`）に対して、以下の作業を行います：

#### 1. テキスト修正

**入力ファイル**: `data/<language>/<model>/step2/qa.jsonl`

**作業内容**:
- 質問文の誤りを修正
- 文の誤りを修正（タイポ、文法エラーなど）
- 不自然な表現を改善

**出力ファイル**: `text_corrections_<user_id>.json`

**アプリケーション**: テキスト修正アプリを選択

#### 2. 評価（オプション）

修正後のQA問題の品質を評価します。

**出力ファイル**: `evaluations_<user_id>.json`

**アプリケーション**: 評価アプリを選択

#### 3. アノテーション結果の統合

複数ユーザーのアノテーション結果を統合し、修正済みQAデータを生成します。

```bash
cd tools
uv run python postprocess/for_step2.py \
    --input_file ../data/ja/gpt-4.1-mini/step2/qa.jsonl
```

**処理内容**:
- `bad example` を除外
- 質問文の形式を統一（`質問文\nSentence 1: ...\nSentence 2: ...`）
- 文の末尾を正規化（`.` → `。`）
- 修正された質問文を反映

**出力ファイル**: `data/<language>/<model>/step2+annotation/text_corrected_qa.jsonl`

### Step 3: 追加文生成後のアノテーション

Step 3で生成された4文構造のQAデータ（`data/<language>/<model>/step3/qa.jsonl`）に対して、評価を行います。

#### 1. 評価ラウンド1

**入力ファイル**: `data/<language>/<model>/step3/qa.jsonl`

**作業内容**:
- 各問題の正解を評価
- 質問の品質を評価
- コメントを追加（必要に応じて）

**出力ファイル**: `evaluations_<user_id>_step3_qa_<timestamp>.json`

**アプリケーション**: 評価アプリを選択

#### 2. 評価結果の統合（ラウンド1）

```bash
cd tools
uv run python postprocess/for_human_eval.py \
    --input_folder ../data/ja/gpt-4.1-mini/step3/eval_round1/annotated
```

**出力ファイル**:
- `data/<language>/<model>/step3/eval_round1/evaluations_merged.tsv` - 統合評価結果
- `data/<language>/<model>/step3/eval_round1/annotated/*.png` - 統計グラフ

#### 3. 評価ラウンド2（必要に応じて）

ラウンド1で問題があったデータを再評価します。

**入力ファイル**: ラウンド1で問題があったデータのみ

**出力ファイル**: `evaluations_<user_id>_eval_round2_qa_<timestamp>.json`

#### 4. 最終統合

複数ラウンドの評価結果を統合し、最終的なQAデータを生成します。

```bash
cd tools
uv run python postprocess/merge_eval_rounds.py \
    --base_dir ../data/ja/gpt-4.1-mini/step3
```

**処理内容**:
- ラウンド2のデータを優先
- ラウンド1のデータで補完
- `bad example` を除外
- QA JSONL形式に変換

**出力ファイル**: `data/<language>/<model>/step3/qa_annotated.jsonl`

このファイルが最終的な評価用データとして使用されます。

## 🔐 環境変数の設定

### ユーザーの認証情報

**ローカル開発環境**:
`src/auth_config.json` ファイルで管理:

```json
{
  "admin": "0000",
  "user1": "1234"
}
```

**Hugging Face Spaces**:
環境変数 `FRAME_QA_AUTH_USERS` にJSON形式で設定:
```json
{
  "admin": "0000",
  "user1": "1234"
}
```

### ユーザーへのデータ割当

**ローカル開発環境**:
`src/user_assignment.json` ファイルで管理:

```json
{
  "admin": {
    "name": "admin",
    "data_range": {
      "start": 0,
      "end": 9999
    },
    "description": "管理者用のデータ範囲（全データアクセス可能）"
  },
  "user1": {
    "name": "user1",
    "data_range": {
      "start": 0,
      "end": 100
    },
    "description": "ユーザー1のデータ範囲"
  }
}
```

**Hugging Face Spaces**:
環境変数 `FRAME_QA_USER_ASSIGNMENTS` にJSON形式で設定:
```json
{
  "admin": {
    "name": "admin",
    "data_range": {
      "start": 0,
      "end": 9999
    },
    "description": "管理者用のデータ範囲（全データアクセス可能）"
  }
}
```

## 📁 ファイル構成

```
tools/
├── app.py                        # メインアプリケーション
├── src/
│   ├── common.py                # 共通処理のベースクラス
│   ├── text_correction_app.py   # テキスト修正専用アプリ
│   ├── evaluation_app.py        # 評価専用アプリ
│   ├── dataset_manager.py       # データセット管理
│   ├── auth_config.json         # 認証設定（.gitignore）
│   ├── user_assignment.json     # ユーザー割当設定（.gitignore）
│   ├── dataset_config.yaml      # データセット設定
│   └── evaluation_criteria.yaml # 評価基準
├── postprocess/
│   ├── for_step2.py             # Step 2アノテーション結果の統合
│   ├── for_human_eval.py        # Step 3評価結果の統合
│   └── merge_eval_rounds.py     # 複数ラウンドの評価結果を統合
├── data/                        # データディレクトリ（シンボリックリンク）
│   └── frame-definition/
│       └── ja/
│           └── frames_translated.jsonl  # フレーム定義
├── text_corrections_<user_id>.json  # テキスト修正結果（自動生成）
├── evaluations_<user_id>.json      # 評価結果（自動生成）
└── README.md                    # このファイル
```

## 📊 データの保存とダウンロード

### 自動保存
- アノテーション結果は各セッションで自動的にローカルに保存されます
- ファイル名: `text_corrections_<user_id>.json`, `evaluations_<user_id>_<timestamp>.json`

### ダウンロード
- 「アノテーション結果をダウンロード」ボタンでJSONファイルをダウンロード可能
- 各ユーザーのアノテーション結果のみがダウンロードされます

### データ形式

#### テキスト修正結果 (`text_corrections_<user_id>.json`)

```json
{
  "<annotation_key>": {
    "corrected_question": "修正された質問文\nSentence 1: ...\nSentence 2: ...",
    "original_question": "元の質問文\nSentence 1: ...\nSentence 2: ...",
    "annotation_type": "text_correction"
  }
}
```

#### 評価結果 (`evaluations_<user_id>_<timestamp>.json`)

```json
{
  "<annotation_key>": {
    "corrected_question": "質問文\nSentence 1: ...\nSentence 2: ...",
    "evaluations": {
      "回答": 2,
      "日本語の品質": 1
    },
    "comments": "コメント（任意）",
    "annotation_type": "evaluation"
  }
}
```

## 🔄 ワークフロー例

### Step 2のアノテーション

```bash
# 1. アプリケーションを起動
cd tools
uv run python app.py

# 2. ブラウザで http://localhost:7860 にアクセス
# 3. ログインしてテキスト修正アプリを選択
# 4. データを確認し、修正を実行
# 5. 結果をダウンロード（または自動保存されたファイルを使用）

# 6. アノテーション結果を統合
uv run python postprocess/for_step2.py \
    --input_file ../data/ja/gpt-4.1-mini/step2/qa.jsonl

# 7. 修正済みデータが生成される
# data/ja/gpt-4.1-mini/step2+annotation/text_corrected_qa.jsonl
```

### Step 3のアノテーション

```bash
# 1. アプリケーションを起動
cd tools
uv run python app.py

# 2. 評価アプリを選択
# 3. データを確認し、評価を実行
# 4. 結果をダウンロード

# 5. 評価結果を統合（ラウンド1）
uv run python postprocess/for_human_eval.py \
    --input_folder ../data/ja/gpt-4.1-mini/step3/eval_round1/annotated

# 6. （必要に応じて）ラウンド2を実行
# 7. 最終統合
uv run python postprocess/merge_eval_rounds.py \
    --base_dir ../data/ja/gpt-4.1-mini/step3

# 8. 最終データが生成される
# data/ja/gpt-4.1-mini/step3/qa_annotated.jsonl
```