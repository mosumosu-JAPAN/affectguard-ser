#!/usr/bin/env python3
"""Build a gold-action vs predicted-action confusion matrix for the 50-case run.

Example:
  .venv/bin/python scripts/confusion_matrix_50.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("results/local_agent_realization_50.csv")
OUTPUT_MD = Path("docs/local_agent_realization_50_confusion_matrix.md")
ACTION_ORDER = ["repair", "clarify", "support", "handoff"]


def main() -> int:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"gold_action", "recommended_action"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    predicted_order = ACTION_ORDER + [
        label
        for label in sorted(df["recommended_action"].dropna().astype(str).unique().tolist())
        if label not in ACTION_ORDER
    ]

    matrix = pd.crosstab(
        df["gold_action"],
        df["recommended_action"],
        dropna=False,
    ).reindex(index=ACTION_ORDER, columns=predicted_order, fill_value=0)

    row_totals = matrix.sum(axis=1)
    row_pct = matrix.div(row_totals.replace(0, pd.NA), axis=0).fillna(0)

    lines = []
    lines.append("# Local Agent Realization Confusion Matrix")
    lines.append("")
    lines.append(f"- Source: `{INPUT_CSV}`")
    lines.append("- Metric: gold action vs predicted action (`recommended_action`)")
    lines.append("- Percentages are row-normalized within each gold action.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")

    header = ["gold_action"] + predicted_order + ["total"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for gold in ACTION_ORDER:
        row = [gold] + [str(int(matrix.loc[gold, pred])) for pred in ACTION_ORDER] + [str(int(row_totals.loc[gold]))]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Row Percentages")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for gold in ACTION_ORDER:
        row = [gold] + [f"{row_pct.loc[gold, pred] * 100:.1f}%" for pred in predicted_order] + ["100.0%"]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Cell View")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for gold in ACTION_ORDER:
        row = [gold]
        for pred in predicted_order:
            count = int(matrix.loc[gold, pred])
            pct = row_pct.loc[gold, pred] * 100
            row.append(f"{count} ({pct:.1f}%)")
        row.append(str(int(row_totals.loc[gold])))
        lines.append("| " + " | ".join(row) + " |")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
