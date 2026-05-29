#!/usr/bin/env python3
# Example: python3 scripts/analyze_realization_metrics.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SEARCH_DIRS = [Path("results"), Path("outputs")]


def find_latest_csv() -> Path:
    candidates = []
    for base_dir in SEARCH_DIRS:
        if not base_dir.exists():
            continue
        candidates.extend(base_dir.rglob("*.csv"))

    if not candidates:
        raise FileNotFoundError("No CSV files found under results/ or outputs/.")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def rate(series: pd.Series, value: str) -> float:
    if series.empty:
        return float("nan")
    return (series == value).mean()


def yes_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    normalized = series.map(
        lambda value: str(value).strip().lower() in {"yes", "true", "1"}
        if not pd.isna(value)
        else False
    )
    return normalized.mean()


def numeric_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return pd.to_numeric(series, errors="coerce").mean()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if "action_realization_quality" not in df.columns:
        raise KeyError(
            "action_realization_quality is missing from the latest CSV. "
            "This script expects a CSV that already includes frontier-judge realization labels."
        )

    group_cols = [col for col in ["model_provider", "model_name"] if col in df.columns]
    if not group_cols:
        df = df.copy()
        df["__group__"] = "all"
        group_cols = ["__group__"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_outputs=("action_realization_quality", "count"),
            action_accuracy=(
                "action_correct",
                yes_rate if "action_correct" in df.columns else "size",
            ),
            unsafe_confidence_rate=(
                "unsafe_confidence",
                yes_rate if "unsafe_confidence" in df.columns else "size",
            ),
            avg_confidence=(
                "confidence",
                numeric_mean if "confidence" in df.columns else "size",
            ),
            realization_strong_rate=(
                "action_realization_quality",
                lambda s: rate(s, "strong"),
            ),
            realization_partial_rate=(
                "action_realization_quality",
                lambda s: rate(s, "partial"),
            ),
            realization_weak_rate=(
                "action_realization_quality",
                lambda s: rate(s, "weak"),
            ),
            realization_wrong_rate=(
                "action_realization_quality",
                lambda s: rate(s, "wrong"),
            ),
            realization_weak_or_wrong_rate=(
                "action_realization_quality",
                lambda s: s.isin(["weak", "wrong"]).mean(),
            ),
        )
        .reset_index()
    )

    if "action_accuracy" in summary.columns and "realization_strong_rate" in summary.columns:
        summary["realization_gap"] = summary["action_accuracy"] - summary["realization_strong_rate"]

    if "__group__" in summary.columns:
        summary = summary.drop(columns=["__group__"])

    return summary


def main() -> int:
    try:
        latest_csv = find_latest_csv()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(latest_csv)
    except Exception as exc:
        print(f"Failed to read CSV {latest_csv}: {exc}", file=sys.stderr)
        return 1

    try:
        summary = summarize(df)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(f"Latest CSV: {latest_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
