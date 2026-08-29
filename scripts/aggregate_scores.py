#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""モデル別の複数プロンプト集計を横持ちテーブルにまとめる。

使用例:
  uv run python scripts/aggregate_scores.py output/en/all
  uv run python scripts/aggregate_scores.py output/ja/all --output output/ja/all/model_scores.tsv
  uv run python scripts/aggregate_scores.py output/en/all --format markdown
  uv run python scripts/aggregate_scores.py output/en/all --include-prompt-errors
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_MMLU_PRO_SCORES_PATH = Path(__file__).resolve().parents[1] / "src/evaluation/mmlu_pro_scores.yaml"
DEFAULT_JAPANESE_BENCHMARK_SCORES_PATH = (
    Path(__file__).resolve().parents[1] / "src/evaluation/japanese_benchmark_scores.yaml"
)
DEFAULT_MODEL_PROFILES_PATH = Path(__file__).resolve().parents[1] / "src/evaluation/model_profiles.yaml"
DEFAULT_BENCHMARK_SCORE_PATHS_BY_LANGUAGE = {
    "en": (DEFAULT_MMLU_PRO_SCORES_PATH,),
    "ja": (DEFAULT_JAPANESE_BENCHMARK_SCORES_PATH,),
}
BENCHMARK_COLUMNS = {
    "mmlu_pro": "MMLU-Pro Overall",
    "gpqa": "GPQA",
    "hle": "HLE",
    "mmmu_pro": "MMMU-Pro Overall",
    "jamc_qa": "JamC-QA",
    "mmlu_prox": "MMLU-ProX",
}
DATA_SOURCE_MARKS = {
    "Self-Reported": "a",
    "TIGER-Lab": "b",
    "Artificial Analysis API": "c",
    "MMMU Leaderboard (author)": "a",
    "MMMU Leaderboard": "d",
    "Swallow LLM Leaderboard": "e",
    "OUR EVAL": "f",
}


@dataclass(frozen=True)
class PromptScore:
    prompt: str
    accuracy: float
    correct: int | None
    total: int | None
    errors: int


@dataclass(frozen=True)
class ModelSummary:
    model: str
    avg_accuracy: float
    std_accuracy: float
    prompt_scores: dict[str, PromptScore]
    error_sum: int


@dataclass(frozen=True)
class ExternalBenchmarkScore:
    benchmark: str
    overall: float
    source_model: str | None
    data_source: str | None


def _natural_key(text: str) -> list[object]:
    """prompt_v10 が prompt_v2 より後に来るようにソートする。"""
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _split_model_variant(model_name: str) -> tuple[str, int]:
    """thinking/no_thinking などの実行モードをモデル本体から切り離す。"""
    variant_patterns = (
        (r"_no_thinking$", 1),
        (r"_thinking(?:_reasoning_[^_]+)?$", 0),
        (r"_reasoning_[^_]+$", 2),
    )
    for pattern, order in variant_patterns:
        if re.search(pattern, model_name):
            return re.sub(pattern, "", model_name), order
    return model_name, 3


def _thinking_label(model_name: str) -> str:
    if re.search(r"_no_thinking$", model_name):
        return ""

    reasoning_match = re.search(r"(?:_thinking)?_reasoning_([^_]+)$", model_name)
    if reasoning_match:
        return f"✓ {reasoning_match.group(1)}"

    return "✓" if re.search(r"_thinking$", model_name) else ""


def _sanitize_model_profile_key(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _read_model_profiles(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_profiles = data.get("models", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"models はマッピング形式で指定してください: {path}")
    return {str(model_name): profile for model_name, profile in raw_profiles.items() if isinstance(profile, dict)}


def _get_model_profile_for_result_model(model_name: str, profiles: dict[str, dict]) -> dict | None:
    """評価結果ディレクトリ名から model_profiles.yaml の定義を探す。"""
    base_model_name, _ = _split_model_variant(model_name)
    candidates = (model_name, base_model_name)
    for candidate in candidates:
        if candidate in profiles:
            return profiles[candidate]

        for profile_model_name, profile in profiles.items():
            profile_candidates = (
                profile_model_name,
                _sanitize_model_profile_key(profile_model_name),
            )
            if any(candidate == key or candidate.startswith(key) for key in profile_candidates):
                return profile
    return None


def _is_multimodal_model(model_name: str, profiles: dict[str, dict]) -> bool:
    profile = _get_model_profile_for_result_model(model_name, profiles)
    return profile is not None and profile.get("supports_multimodal") is True


def _display_model_name(model_name: str) -> str:
    base_model_name, _ = _split_model_variant(model_name)
    return base_model_name.split("_", 1)[1] if "_" in base_model_name else base_model_name


def _model_family_key(base_model_name: str) -> list[object]:
    """Qwen3-4B と Qwen3-32B が同じファミリーにまとまるキーを作る。"""
    match = re.search(r"(?i)(?:^|[-_])(?:[ea]?\d+(?:\.\d+)?b)", base_model_name)
    if not match:
        return _natural_key(base_model_name)
    family = base_model_name[: match.start()].rstrip("-_")
    return _natural_key(family)


def _model_size_key(base_model_name: str) -> tuple[float, ...]:
    sizes = [
        float(match.group("size"))
        for match in re.finditer(
            r"(?i)(?<![a-z0-9])(?:[ea])?(?P<size>\d+(?:\.\d+)?)b(?![a-z0-9])",
            base_model_name,
        )
    ]
    return tuple(sizes) if sizes else (float("inf"),)


def _preferred_model_family_order(base_model_name: str) -> int:
    normalized = base_model_name.lower().replace("/", "_")
    family_order_patterns = (
        (0, r"^gpt-5(?:[._-]|$)"),
        (0, r"^gpt-5-nano(?:[._-]|$)"),
        (1, r"^gemini-"),
        (2, r"^(?:openai_)?gpt-oss-"),
        (3, r"^(?:google_)?gemma-3-"),
        (4, r"^(?:google_)?gemma-4-"),
        (5, r"^qwen_qwen3(?!\\.5)"),
        (5, r"^qwen3(?!\\.5)"),
        (6, r"^qwen_qwen3\\.5"),
        (6, r"^qwen3\\.5"),
        (7, r"^llm-jp_"),
        (7, r"^llm-jp/"),
        (8, r"^tokyotech-llm_.*swallow"),
        (8, r"^tokyotech-llm/.*swallow"),
        (8, r"^tokyotech-llm_.*gpt-oss-swallow"),
        (8, r"^tokyotech-llm/.*gpt-oss-swallow"),
    )
    for order, pattern in family_order_patterns:
        if re.search(pattern, normalized):
            return order
    return 9


def _model_sort_key(model_name: str) -> tuple[int, list[object], int, tuple[float, ...], list[object], list[object]]:
    base_model_name, variant_order = _split_model_variant(model_name)
    return (
        _preferred_model_family_order(base_model_name),
        _model_family_key(base_model_name),
        variant_order,
        _model_size_key(base_model_name),
        _natural_key(base_model_name),
        _natural_key(model_name),
    )


def _parse_percent(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1]
    return float(value)


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _format_score_ratio(value: float) -> str:
    """0.886 のようなベンチマーク値を既存列と同じパーセント表記にする。"""
    percent = value * 100 if value <= 1.0 else value
    return _format_percent(percent)


def _format_benchmark_score(score: ExternalBenchmarkScore) -> str:
    mark = DATA_SOURCE_MARKS.get(score.data_source or "", "")
    return f"{_format_score_ratio(score.overall)}{mark}"


def _read_prompt_summary_txt(path: Path, prompt_name: str) -> PromptScore:
    """prompt_vN/summary.txt から品質フィルタ後の正答率などを読む。"""
    accuracy: float | None = None
    correct: int | None = None
    total: int | None = None
    errors = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("正答率:"):
            accuracy = _parse_percent(stripped.split(":", 1)[1])
        elif stripped.startswith("正解数:"):
            correct = int(stripped.split(":", 1)[1])
        elif stripped.startswith("品質フィルタ後問題数:"):
            total = int(stripped.split(":", 1)[1])
        elif stripped.startswith("総問題数:") and total is None:
            total = int(stripped.split(":", 1)[1])
        elif stripped.startswith("エラー数:"):
            errors = int(stripped.split(":", 1)[1])

    if accuracy is None:
        raise ValueError(f"正答率を読み取れませんでした: {path}")

    return PromptScore(
        prompt=prompt_name,
        accuracy=accuracy,
        correct=correct,
        total=total,
        errors=errors,
    )


def _read_tsv_table(path: Path) -> list[PromptScore]:
    rows: list[PromptScore] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("プロンプト"):
                continue
            rows.append(
                PromptScore(
                    prompt=row["プロンプト"],
                    accuracy=_parse_percent(row["正答率"]),
                    correct=int(row["正解数"]) if row.get("正解数") else None,
                    total=int(row["総問題数"]) if row.get("総問題数") else None,
                    errors=int(row.get("エラー数") or 0),
                )
            )
    return rows


def _read_txt_table(path: Path) -> tuple[list[PromptScore], float | None, float | None]:
    rows: list[PromptScore] = []
    avg_accuracy: float | None = None
    std_accuracy: float | None = None
    row_pattern = re.compile(
        r"^(?P<prompt>\S+)\s+"
        r"(?P<accuracy>\d+(?:\.\d+)?)%\s+"
        r"(?P<correct>\d+)\s+"
        r"(?P<total>\d+)\s+"
        r"(?P<errors>\d+)\s*$"
    )

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        row_match = row_pattern.match(stripped)
        if row_match:
            rows.append(
                PromptScore(
                    prompt=row_match.group("prompt"),
                    accuracy=float(row_match.group("accuracy")),
                    correct=int(row_match.group("correct")),
                    total=int(row_match.group("total")),
                    errors=int(row_match.group("errors")),
                )
            )
            continue

        if stripped.startswith("平均正答率:"):
            avg_accuracy = _parse_percent(stripped.split(":", 1)[1])
        elif stripped.startswith("標準偏差:"):
            std_accuracy = _parse_percent(stripped.split(":", 1)[1])

    return rows, avg_accuracy, std_accuracy


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance**0.5


def _normalize_prompt_filters(
    include_prompts: list[str] | None,
    exclude_prompts: list[str] | None,
) -> tuple[frozenset[str] | None, frozenset[str]]:
    include = frozenset(include_prompts) if include_prompts else None
    exclude = frozenset(exclude_prompts or ())
    return include, exclude


def _prompt_name_from_dir(prompt_dir: Path) -> str:
    return prompt_dir.name


def _prompt_matches_filters(
    prompt_name: str,
    include: frozenset[str] | None,
    exclude: frozenset[str],
) -> bool:
    if prompt_name in exclude:
        return False
    if include is not None:
        return prompt_name in include
    return prompt_name.startswith("prompt_")


def _read_model_summary_from_prompt_dirs(
    model_dir: Path,
    include_prompts: list[str] | None = None,
    exclude_prompts: list[str] | None = None,
) -> ModelSummary:
    include, exclude = _normalize_prompt_filters(include_prompts, exclude_prompts)
    rows: list[PromptScore] = []

    for prompt_dir in sorted(model_dir.iterdir(), key=lambda path: _natural_key(path.name)):
        if not prompt_dir.is_dir():
            continue
        prompt_name = _prompt_name_from_dir(prompt_dir)
        if not _prompt_matches_filters(prompt_name, include, exclude):
            continue
        summary_path = prompt_dir / "summary.txt"
        if not summary_path.exists():
            continue
        rows.append(_read_prompt_summary_txt(summary_path, prompt_name))

    if not rows:
        raise ValueError(f"対象プロンプトの summary.txt が見つかりません: {model_dir}")

    accuracies = [row.accuracy for row in rows]
    return ModelSummary(
        model=model_dir.name,
        avg_accuracy=sum(accuracies) / len(accuracies),
        std_accuracy=_population_std(accuracies),
        prompt_scores={row.prompt: row for row in rows},
        error_sum=sum(row.errors for row in rows),
    )


def _read_model_summary(model_dir: Path) -> ModelSummary:
    tsv_path = model_dir / "aggregated_summary.tsv"
    txt_path = model_dir / "aggregated_summary.txt"

    txt_rows: list[PromptScore] = []
    avg_accuracy: float | None = None
    std_accuracy: float | None = None
    if txt_path.exists():
        txt_rows, avg_accuracy, std_accuracy = _read_txt_table(txt_path)

    if tsv_path.exists():
        rows = _read_tsv_table(tsv_path)
    else:
        rows = txt_rows

    if not rows:
        raise ValueError(f"プロンプト別スコアを読み取れませんでした: {model_dir}")

    accuracies = [row.accuracy for row in rows]
    if avg_accuracy is None:
        avg_accuracy = sum(accuracies) / len(accuracies)
    if std_accuracy is None:
        std_accuracy = _population_std(accuracies)

    prompt_scores = {row.prompt: row for row in rows}
    return ModelSummary(
        model=model_dir.name,
        avg_accuracy=avg_accuracy,
        std_accuracy=std_accuracy,
        prompt_scores=prompt_scores,
        error_sum=sum(row.errors for row in rows),
    )


def _discover_model_dirs(
    results_dir: Path,
    *,
    from_prompt_dirs: bool = False,
    include_prompts: list[str] | None = None,
    exclude_prompts: list[str] | None = None,
) -> list[Path]:
    include, exclude = _normalize_prompt_filters(include_prompts, exclude_prompts)
    model_dirs: list[Path] = []

    for path in results_dir.iterdir():
        if not path.is_dir() or path.name == "analysis":
            continue
        if from_prompt_dirs:
            has_prompt = any(
                child.is_dir()
                and _prompt_matches_filters(_prompt_name_from_dir(child), include, exclude)
                and (child / "summary.txt").exists()
                for child in path.iterdir()
            )
            if has_prompt:
                model_dirs.append(path)
        elif (path / "aggregated_summary.tsv").exists() or (path / "aggregated_summary.txt").exists():
            model_dirs.append(path)

    return sorted(model_dirs, key=lambda path: _model_sort_key(path.name))


def _read_external_benchmark_scores(path: Path) -> dict[str, dict[str, ExternalBenchmarkScore]]:
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise ValueError(f"scores はマッピング形式で指定してください: {path}")

    scores: dict[str, dict[str, ExternalBenchmarkScore]] = {}
    for model_name, value in raw_scores.items():
        benchmark_values = {"mmlu_pro": value} if not isinstance(value, dict) or "overall" in value else value
        if not isinstance(benchmark_values, dict):
            raise ValueError(f"スコアはマッピング形式で指定してください: {path} ({model_name})")

        scores[str(model_name)] = {}
        for benchmark, benchmark_value in benchmark_values.items():
            if benchmark not in BENCHMARK_COLUMNS:
                continue
            if isinstance(benchmark_value, dict):
                if "overall" not in benchmark_value:
                    raise ValueError(f"overall がありません: {path} ({model_name}.{benchmark})")
                overall = float(benchmark_value["overall"])
                source_model = benchmark_value.get("source_model") or benchmark_value.get("leaderboard_model")
                data_source = benchmark_value.get("data_source")
            else:
                overall = float(benchmark_value)
                source_model = None
                data_source = None
            scores[str(model_name)][str(benchmark)] = ExternalBenchmarkScore(
                benchmark=str(benchmark),
                overall=overall,
                source_model=str(source_model) if source_model is not None else None,
                data_source=str(data_source) if data_source is not None else None,
            )
    return scores


def _read_external_benchmark_score_files(paths: Iterable[Path]) -> dict[str, dict[str, ExternalBenchmarkScore]]:
    scores: dict[str, dict[str, ExternalBenchmarkScore]] = {}
    for path in paths:
        for model_name, benchmark_scores in _read_external_benchmark_scores(path).items():
            scores.setdefault(model_name, {}).update(benchmark_scores)
    return scores


def _default_benchmark_score_paths(results_dir: Path) -> list[Path]:
    for part in results_dir.parts:
        if part in DEFAULT_BENCHMARK_SCORE_PATHS_BY_LANGUAGE:
            return list(DEFAULT_BENCHMARK_SCORE_PATHS_BY_LANGUAGE[part])
    return []


def _build_rows(
    summaries: list[ModelSummary],
    prompts: list[str],
    include_prompt_errors: bool,
    benchmark_scores: dict[str, dict[str, ExternalBenchmarkScore]] | None = None,
    model_profiles: dict[str, dict] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    benchmark_scores = benchmark_scores or {}
    model_profiles = model_profiles or {}
    available_benchmarks = {
        benchmark
        for model_benchmark_scores in benchmark_scores.values()
        for benchmark in model_benchmark_scores
    }
    base_model_counts: dict[str, int] = {}
    for summary in summaries:
        base_model_name, _ = _split_model_variant(summary.model)
        base_model_counts[base_model_name] = base_model_counts.get(base_model_name, 0) + 1

    for summary in summaries:
        row: dict[str, str] = {
            "モデル": _display_model_name(summary.model),
            "Thinking": _thinking_label(summary.model),
            "Multimodal": "✓" if _is_multimodal_model(summary.model, model_profiles) else "",
            "平均正答率": _format_percent(summary.avg_accuracy),
            "標準偏差": _format_percent(summary.std_accuracy),
        }
        if benchmark_scores:
            base_model_name, _ = _split_model_variant(summary.model)
            model_benchmark_scores: dict[str, ExternalBenchmarkScore] = {}
            if base_model_counts[base_model_name] == 1:
                model_benchmark_scores.update(benchmark_scores.get(base_model_name, {}))
            model_benchmark_scores.update(benchmark_scores.get(summary.model, {}))
            for benchmark, column in BENCHMARK_COLUMNS.items():
                if benchmark not in available_benchmarks:
                    continue
                benchmark_score = model_benchmark_scores.get(benchmark)
                row[column] = _format_benchmark_score(benchmark_score) if benchmark_score is not None else ""

        for prompt in prompts:
            prompt_score = summary.prompt_scores.get(prompt)
            if prompt_score is None:
                row[f"{prompt}_正答率"] = ""
                if include_prompt_errors:
                    row[f"{prompt}_エラー数"] = ""
            else:
                row[f"{prompt}_正答率"] = _format_percent(prompt_score.accuracy)
                if include_prompt_errors:
                    row[f"{prompt}_エラー数"] = str(prompt_score.errors)

        row["エラー数Sum"] = str(summary.error_sum)
        rows.append(row)
    return rows


def _write_delimited(rows: list[dict[str, str]], columns: list[str], output: Path, delimiter: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join(lines)


def _format_markdown_report(
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    title: str | None = None,
    results_dir: Path | None = None,
    prompts: list[str] | None = None,
    exclude_prompts: list[str] | None = None,
    include_benchmark_footnotes: bool = False,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    meta: list[str] = []
    if results_dir is not None:
        meta.append(f"- 結果ディレクトリ: `{results_dir}`")
    if prompts:
        meta.append(f"- 集計プロンプト: {', '.join(f'`{p}`' for p in prompts)}")
    if exclude_prompts:
        meta.append(f"- 除外プロンプト: {', '.join(f'`{p}`' for p in exclude_prompts)}")
    if meta:
        parts.append("\n".join(meta) + "\n")

    parts.append(_markdown_table(rows, columns) + "\n")

    if include_benchmark_footnotes:
        parts.append(
            "ベンチマーク列の末尾記号: "
            + ", ".join(f"{mark} = {name}" for name, mark in DATA_SOURCE_MARKS.items())
            + "\n"
        )
    return "\n".join(parts)


def _print_delimited(rows: list[dict[str, str]], columns: list[str], delimiter: str) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _collect_prompts(
    summaries: Iterable[ModelSummary],
    include_prompts: list[str] | None = None,
    exclude_prompts: list[str] | None = None,
) -> list[str]:
    include, exclude = _normalize_prompt_filters(include_prompts, exclude_prompts)
    if include is not None:
        prompts = {prompt for prompt in include if prompt not in exclude}
    else:
        prompts = {prompt for summary in summaries for prompt in summary.prompt_scores}
        prompts -= exclude
    return sorted(prompts, key=_natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aggregated_summary.{tsv,txt} をモデル別の横持ち表に集計します。",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="モデル別ディレクトリが並ぶ結果ディレクトリ（例: output/en/all）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="保存先。未指定の場合は標準出力に表示します。",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "csv", "markdown"),
        default="tsv",
        help="出力形式（デフォルト: tsv）",
    )
    parser.add_argument(
        "--include-prompt-errors",
        action="store_true",
        help="各プロンプトのエラー数列も出力します。未指定時は右端のエラー数Sumのみ出力します。",
    )
    parser.add_argument(
        "--mmlu-pro-scores",
        type=Path,
        default=None,
        help=(
            "外部ベンチマークスコアを結合するYAML。"
            "互換性のため残しているオプションで、指定時はこのファイルだけを使います。"
        ),
    )
    parser.add_argument(
        "--benchmark-scores",
        type=Path,
        action="append",
        default=None,
        help=(
            "外部ベンチマークスコアを結合するYAML。複数回指定できます。"
            "未指定時は results_dir の言語に応じて en なら英語スコア、ja なら日本語スコアを使います。"
        ),
    )
    parser.add_argument(
        "--from-prompt-dirs",
        action="store_true",
        help="各モデル配下の prompt_vN/summary.txt から読み込みます（aggregated_summary を使いません）。",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        metavar="PROMPT",
        help="集計対象のプロンプト名（例: prompt_v1 prompt_v2）。--from-prompt-dirs と併用。",
    )
    parser.add_argument(
        "--exclude-prompts",
        nargs="+",
        default=None,
        metavar="PROMPT",
        help="除外するプロンプト名（例: prompt_v4）。",
    )
    parser.add_argument(
        "--markdown-title",
        type=str,
        default=None,
        help="--format markdown 時に表の前へ付ける見出し（例: 日本語 FrameBench スコア）。",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        parser.error(f"results_dir が存在しません: {results_dir}")
    if not results_dir.is_dir():
        parser.error(f"results_dir はディレクトリを指定してください: {results_dir}")

    model_dirs = _discover_model_dirs(
        results_dir,
        from_prompt_dirs=args.from_prompt_dirs,
        include_prompts=args.prompts,
        exclude_prompts=args.exclude_prompts,
    )
    if not model_dirs:
        if args.from_prompt_dirs:
            parser.error(f"対象プロンプトの summary.txt を含むモデルディレクトリが見つかりません: {results_dir}")
        parser.error(f"aggregated_summary.tsv/txt を含むモデルディレクトリが見つかりません: {results_dir}")

    summaries: list[ModelSummary] = []
    read_errors: list[str] = []
    for model_dir in model_dirs:
        try:
            if args.from_prompt_dirs:
                summaries.append(
                    _read_model_summary_from_prompt_dirs(
                        model_dir,
                        include_prompts=args.prompts,
                        exclude_prompts=args.exclude_prompts,
                    )
                )
            else:
                summaries.append(_read_model_summary(model_dir))
        except ValueError as exc:
            read_errors.append(f"{model_dir.name}: {exc}")

    if not summaries:
        parser.error("読み取れたモデルがありません。\n" + "\n".join(read_errors))

    if read_errors:
        print("警告: 以下のモデルはスキップしました:", file=sys.stderr)
        for message in read_errors:
            print(f"  - {message}", file=sys.stderr)

    prompts = _collect_prompts(summaries, args.prompts, args.exclude_prompts)
    prompt_columns = (
        [column for prompt in prompts for column in (f"{prompt}_正答率", f"{prompt}_エラー数")]
        if args.include_prompt_errors
        else [f"{prompt}_正答率" for prompt in prompts]
    )
    if args.benchmark_scores is not None:
        benchmark_score_paths = args.benchmark_scores
        if args.mmlu_pro_scores is not None:
            benchmark_score_paths = [args.mmlu_pro_scores] + benchmark_score_paths
    elif args.mmlu_pro_scores is not None:
        benchmark_score_paths = [args.mmlu_pro_scores]
    else:
        benchmark_score_paths = _default_benchmark_score_paths(results_dir)
    benchmark_scores = _read_external_benchmark_score_files(benchmark_score_paths)
    model_profiles = _read_model_profiles(DEFAULT_MODEL_PROFILES_PATH)
    available_benchmarks = {
        benchmark
        for model_benchmark_scores in benchmark_scores.values()
        for benchmark in model_benchmark_scores
    }
    external_columns = [
        column
        for benchmark, column in BENCHMARK_COLUMNS.items()
        if benchmark in available_benchmarks
    ]
    columns = ["モデル", "Thinking", "Multimodal", "平均正答率", "標準偏差"] + external_columns + prompt_columns + ["エラー数Sum"]
    rows = _build_rows(summaries, prompts, args.include_prompt_errors, benchmark_scores, model_profiles)

    if args.format == "markdown":
        content = _format_markdown_report(
            rows,
            columns,
            title=args.markdown_title,
            results_dir=results_dir,
            prompts=args.prompts,
            exclude_prompts=args.exclude_prompts,
            include_benchmark_footnotes=bool(available_benchmarks),
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
        else:
            print(content, end="")
        return

    delimiter = "\t" if args.format == "tsv" else ","
    if args.output:
        _write_delimited(rows, columns, args.output, delimiter)
    else:
        _print_delimited(rows, columns, delimiter)


if __name__ == "__main__":
    main()
