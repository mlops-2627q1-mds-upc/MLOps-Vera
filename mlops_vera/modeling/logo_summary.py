"""DVC stage `logo_summary`: aggregate the leave-one-generator-out runs and check MR-4.

Collects the unseen-generator metrics of every `logo@<generator>` stage into one table (also a
DVC plot) and checks the cross-generator target on the worst held-out generator, since a
detector is only as robust as its weakest case. The run is logged to MLflow next to the
per-generator runs (experiment `vera-logo`), with a chart of the unseen balanced accuracies.
"""

import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import pandas as pd
import typer

from mlops_vera.config import METRICS_DIR, load_params
from mlops_vera.tracking import stage_run

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


def plot_unseen(table: pd.DataFrame, target: float) -> Figure:
    """Unseen balanced accuracy per held-out generator, with its 95% CI and the MR-4 target."""
    fig = Figure(figsize=(5, 3.3), layout="constrained")
    ax = fig.subplots()
    ba = table["balanced_accuracy"]
    xerr = None
    if {"balanced_accuracy_ci_low", "balanced_accuracy_ci_high"} <= set(table):
        xerr = [ba - table["balanced_accuracy_ci_low"], table["balanced_accuracy_ci_high"] - ba]
    ax.barh(table["holdout"], ba, xerr=xerr, capsize=3)
    ax.axvline(target, color="tab:red", linestyle="--", label=f"MR-4 target ({target:g})")
    ax.set(xlabel="Balanced accuracy on the unseen generator", xlim=(0.5, 1))
    fig.legend(loc="outside upper right", frameon=False)
    return fig


@app.command()
def main(
    logo_dir: Path = METRICS_DIR / "logo",
    output_dir: Path = METRICS_DIR,
    experiment: str = "vera-logo",
):
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

    with stage_run("logo_summary", experiment):
        mlflow.log_params({f"logo.{k}": p[k] for k in ("holdout", "target_balanced_accuracy")})
        mlflow.set_tag("worst_generator", summary["worst_generator"])
        metrics = ("mean_balanced_accuracy", "min_balanced_accuracy", "meets_mr4")
        mlflow.log_metrics({k: float(summary[k]) for k in metrics})  # meets_mr4: 1 or 0
        mlflow.log_artifact(output_dir / "logo_table.csv")
        mlflow.log_figure(plot_unseen(table, summary["target_balanced_accuracy"]), "logo.png")
    logger.success(f"Leave-one-generator-out summary: {summary}")


if __name__ == "__main__":
    app()
