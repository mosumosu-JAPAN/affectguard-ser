#!/usr/bin/env python3
"""Summarize local-agent realization quality by model.

Reads the latest model summary CSV produced by the app and augments it with
realization metrics derived from frontier judge rows in data/model_outputs.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY_CSV = Path("results/summary_by_model.csv")
DEFAULT_REVIEW_CSV = Path("results/model_outputs_review.csv")
DEFAULT_JSONL = Path("data/model_outputs.jsonl")
DEFAULT_OUTPUT_CSV = Path("results/summary_by_model_with_realization.csv")


def load_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def resolve_summary_csv(path: Path) -> Path:
    if path.exists():
        return path
    if DEFAULT_REVIEW_CSV.exists():
        return DEFAULT_REVIEW_CSV
    return path


def load_realization_eval_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("row_type") == "evaluation":
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "target_model_provider" not in df.columns or "target_model_name" not in df.columns:
        return pd.DataFrame()

    if "action_realization_quality" not in df.columns:
        df["action_realization_quality"] = pd.NA

    df["model_provider"] = df["target_model_provider"]
    df["model_name"] = df["target_model_name"]
    return df


def summarize_realization(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_provider",
                "model_name",
                "realization_strong_rate",
                "realization_partial_rate",
                "realization_weak_rate",
                "realization_wrong_rate",
            ]
        )

    summary = (
        df.groupby(["model_provider", "model_name"], dropna=False)
        .agg(
            realization_strong_rate=(
                "action_realization_quality",
                lambda s: (s == "strong").mean(),
            ),
            realization_partial_rate=(
                "action_realization_quality",
                lambda s: (s == "partial").mean(),
            ),
            realization_weak_rate=(
                "action_realization_quality",
                lambda s: (s == "weak").mean(),
            ),
            realization_wrong_rate=(
                "action_realization_quality",
                lambda s: (s == "wrong").mean(),
            ),
        )
        .reset_index()
    )
    return summary


def build_summary(summary_csv: Path, jsonl_path: Path) -> pd.DataFrame:
    base = load_summary_csv(summary_csv)
    realization_rows = load_realization_eval_rows(jsonl_path)
    realization_summary = summarize_realization(realization_rows)

    merged = base.merge(
        realization_summary,
        on=["model_provider", "model_name"],
        how="left",
    )

    preferred_cols = [
        "model_provider",
        "model_name",
        "n_outputs",
        "action_accuracy",
        "unsafe_confidence_rate",
        "avg_confidence",
        "realization_strong_rate",
        "realization_partial_rate",
        "realization_weak_rate",
        "realization_wrong_rate",
    ]
    existing_cols = [col for col in preferred_cols if col in merged.columns]
    return merged[existing_cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report local-agent realization quality grouped by model."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to results/summary_by_model.csv",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to data/model_outputs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Optional CSV output path for the merged summary.",
    )
    args = parser.parse_args()

    summary_csv = resolve_summary_csv(args.summary_csv)
    summary = build_summary(summary_csv, args.jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print(f"\nInput summary: {summary_csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
