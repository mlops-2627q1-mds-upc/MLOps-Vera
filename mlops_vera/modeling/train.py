"""DVC stage `train`: fit a classifier on the frozen embeddings and track the run in MLflow.

- Model: standardisation + logistic regression, class-weighted against the 5:1 fake:real
  imbalance (hyper-parameters in the `train` section of params.yaml).
- The decision threshold on P(AI) is tuned on validation to maximise balanced accuracy instead
  of being fixed at 0.5, as the model card specifies.
- Validation metrics follow the model card: balanced accuracy, macro-F1, PR-AUC and recall of
  the minority (real) class, plus the detection rate of each generator.
- Params, metrics, plots and the model are logged to MLflow (DagsHub when `.env` is set, else
  the local ./mlflow.db); the model and metrics are also written as DVC outputs.

The test split is not used here: it is reserved for the final evaluation, so that choosing
between experiments cannot overfit it.
"""

import json
from pathlib import Path

import joblib
from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import typer
import yaml

from mlops_vera.config import (
    CLASSIFIER_PATH,
    EMBEDDINGS_DIR,
    METRICS_DIR,
    PROJ_ROOT,
    load_params,
)

app = typer.Typer()

# Label_B values of the Defactify dataset (0 = real photo)
GENERATORS = {1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}


def load_split(embeddings_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, generator) of one split; y is 1 for AI-generated, 0 for real."""
    with np.load(embeddings_dir / f"{split}.npz") as d:
        return d["X"], d["y"], d["generator"]


def build_model(p: dict) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=p["C"],
            class_weight=p["class_weight"],
            max_iter=p["max_iter"],
            random_state=p["seed"],
        ),
    )


def tune_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on P(AI) maximising balanced accuracy, i.e. (TPR + TNR) / 2 = Youden's J."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    best = np.argmax((tpr - fpr)[1:]) + 1  # thresholds[0] is +inf (predicts nothing as AI)
    return float(thresholds[best])


def evaluate(y: np.ndarray, generator: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Model-card metrics for P(AI) `scores` classified at `threshold`."""
    pred = (scores >= threshold).astype(int)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc_real": average_precision_score(y == 0, 1 - scores),  # minority class
        "recall_real": recall_score(y, pred, pos_label=0),
        "recall_ai": recall_score(y, pred, pos_label=1),
    }
    # Share of each generator's images flagged as AI (generators absent from `y` are skipped)
    for g, name in GENERATORS.items():
        if (mask := generator == g).any():
            metrics[f"recall_{name}"] = pred[mask].mean()
    return {k: float(v) for k, v in metrics.items()}


def plot_confusion_matrix(y: np.ndarray, pred: np.ndarray) -> Figure:
    fig = Figure(figsize=(4, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["real", "AI"], colorbar=False, ax=fig.subplots()
    )
    return fig


def plot_pr_curve(y: np.ndarray, scores: np.ndarray) -> Figure:
    fig = Figure(figsize=(5, 4), layout="constrained")
    PrecisionRecallDisplay.from_predictions(
        y == 0, 1 - scores, name="real (minority)", ax=fig.subplots()
    )
    return fig


def dvc_hash(path: str) -> str | None:
    """md5 that dvc.lock records for output `path`: ties the run to an exact data version."""
    lock = yaml.safe_load((PROJ_ROOT / "dvc.lock").read_text())
    for stage in lock["stages"].values():
        for out in stage.get("outs", []):
            if out["path"] == path:
                return out["md5"]
    return None


@app.command()
def main(
    embeddings_dir: Path = EMBEDDINGS_DIR,
    model_path: Path = CLASSIFIER_PATH,
    metrics_path: Path = METRICS_DIR / "train_metrics.json",
    experiment: str = "vera-baselines",
):
    p_embed, p = load_params("embed"), load_params("train")
    info = json.loads((embeddings_dir / "info.json").read_text())
    X_train, y_train, gen_train = load_split(embeddings_dir, "train")
    X_val, y_val, gen_val = load_split(embeddings_dir, "val")

    model = build_model(p).fit(X_train, y_train)
    scores_train = model.predict_proba(X_train)[:, 1]
    scores_val = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, scores_val)
    metrics = {
        "threshold": threshold,
        "train": evaluate(y_train, gen_train, scores_train, threshold),
        "val": evaluate(y_val, gen_val, scores_val, threshold),
    }
    logger.info(f"Validation metrics (threshold {threshold:.3f}): {metrics['val']}")

    # DVC outputs. The threshold and backbone travel with the model so serving can reproduce
    # the full image -> embedding -> decision path.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "threshold": threshold, "backbone": info["backbone"]}
    joblib.dump(bundle, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), newline="\n")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{info['backbone']}-logreg") as run:
        mlflow.log_params({f"embed.{k}": v for k, v in p_embed.items()})
        mlflow.log_params({f"train.{k}": v for k, v in p.items()})
        mlflow.set_tags(
            {
                "backbone_weights": info["weights"],
                "data_revision": load_params("data")["revision"],
                "embeddings_md5": dvc_hash("data/processed/embeddings"),
            }
        )
        mlflow.log_metric("threshold", threshold)
        for split in ("train", "val"):
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics[split].items()})

        pred_val = (scores_val >= threshold).astype(int)
        mlflow.log_figure(plot_confusion_matrix(y_val, pred_val), "val_confusion_matrix.png")
        mlflow.log_figure(plot_pr_curve(y_val, scores_val), "val_pr_curve.png")
        mlflow.log_dict(info, "embeddings_info.json")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_val, model.predict(X_val)),
            # Explicit requirements: MLflow's inference re-imports the model in a subprocess
            # and scans every installed package, which takes over a minute.
            pip_requirements=mlflow.sklearn.get_default_pip_requirements(include_skops=True),
        )

    logger.success(
        f"Model -> {model_path}; MLflow run {run.info.run_id} in experiment '{experiment}'"
    )


if __name__ == "__main__":
    app()
