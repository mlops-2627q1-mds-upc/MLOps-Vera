"""DVC stage `logo_summary`: aggregate the leave-one-generator-out runs and check MR-4.

Collects the unseen-generator metrics of every `logo@<generator>` stage into one table (also a
DVC plot) and checks the cross-generator target on the worst held-out generator, since a
detector is only as robust as its weakest case.
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mlops_vera.config import METRICS_DIR, load_params

app = typer.Typer()


def summarise(table: pd.DataFrame, target: float) -> dict:
    """Mean and worst-case unseen balanced accuracy, and whether the worst case meets `target`."""
    worst = table.loc[table["balanced_accuracy"].idxmin()]
    return {
        "mean_balanced_accuracy": float(table["balanced_accuracy"].mean()),
        "min_balanced_accuracy": float(worst["balanced_accuracy"]),
        "worst_generator": str(worst["holdout"]),
        "target_balanced_accuracy": target,
        "meets_mr4": bool(worst["balanced_accuracy"] >= target),
    }


@app.command()
def main(logo_dir: Path = METRICS_DIR / "logo", output_dir: Path = METRICS_DIR):
    p = load_params("logo")
    rows = []
    for name in p["holdout"]:
        m = json.loads((logo_dir / f"{name}.json").read_text())
        rows.append({"holdout": name, "threshold": m["threshold"]} | m["unseen"])
    table = pd.DataFrame(rows)
    summary = summarise(table, p["target_balanced_accuracy"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logo_summary.json").write_text(json.dumps(summary, indent=2), newline="\n")
    table.to_csv(output_dir / "logo_table.csv", index=False, lineterminator="\n")
    logger.success(f"Leave-one-generator-out summary: {summary}")


if __name__ == "__main__":
    app()
