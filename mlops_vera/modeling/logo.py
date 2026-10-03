"""DVC stage `logo@<generator>`: leave-one-generator-out evaluation (requirement MR-4).

Simulates a new generator appearing in production, the drift monitored in Milestone 6:

- The held-out generator is removed from train and val, so neither the classifier head nor the
  tuned decision threshold ever see it. The other four generators are kept, and the head is the
  one of the `train` stage (same hyper-parameters, class-weighted against the new ratio).
- The test set keeps the real images of the test split plus the held-out generator's images
  only. Both come from the caption-grouped test split, so no caption seen in training appears
  in it, and every caption has a real photo and an unseen generated image (content-controlled).
- The unseen balanced accuracy gets a bootstrap 95% confidence interval: each test set has only
  ~190 images per class, so small differences between generators can be noise.

Runs on the frozen embeddings of `embed`: no image is processed again, so each run takes seconds.
DVC runs one stage per generator in params.yaml (`foreach`), and `logo_summary` aggregates them.
"""

import json
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import typer

from mlops_vera.config import EMBEDDINGS_DIR, METRICS_DIR, load_params
from mlops_vera.modeling.train import (
    GENERATORS,
    build_model,
    dvc_hash,
    evaluate,
    load_split,
    tune_threshold,
)

app = typer.Typer()
GENERATOR_IDS = {name: g for g, name in GENERATORS.items()}

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def without_generator(split: Split, g: int) -> Split:
    """Drop every image of generator `g` from (X, y, generator)."""
    X, y, generator = split
    keep = generator != g
    return X[keep], y[keep], generator[keep]


def real_and_generator(split: Split, g: int) -> Split:
    """Keep the real images and the images of generator `g` only."""
    X, y, generator = split
    keep = (generator == 0) | (generator == g)
    return X[keep], y[keep], generator[keep]


def bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the balanced accuracy, resampling within each class
    so every resample keeps both classes (and the same class sizes)."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in (0, 1)]
    stats = [
        balanced_accuracy_score(y[idx], pred[idx])
        for idx in (
            np.concatenate([rng.choice(i, size=len(i)) for i in by_class])
            for _ in range(n_resamples)
        )
    ]
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


@app.command()
def main(
    holdout: str,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    output_dir: Path = METRICS_DIR / "logo",
    experiment: str = "vera-logo",
):
    if holdout not in GENERATOR_IDS:
        raise typer.BadParameter(f"unknown generator {holdout!r}, expected {list(GENERATOR_IDS)}")
    g = GENERATOR_IDS[holdout]
    p, p_logo = load_params("train"), load_params("logo")
    info = json.loads((embeddings_dir / "info.json").read_text())

    X_train, y_train, _ = without_generator(load_split(embeddings_dir, "train"), g)
    X_val, y_val, gen_val = without_generator(load_split(embeddings_dir, "val"), g)
    X_test, y_test, gen_test = real_and_generator(load_split(embeddings_dir, "test"), g)

    model = build_model(p).fit(X_train, y_train)
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(y_val, scores_val)
    unseen = evaluate(y_test, gen_test, scores_test, threshold)
    del unseen[f"recall_{holdout}"]  # the only AI images are the held-out ones: = recall_ai
    low, high = bootstrap_ci(
        y_test, (scores_test >= threshold).astype(int), p_logo["n_bootstrap"], p_logo["seed"]
    )
    unseen |= {"balanced_accuracy_ci_low": low, "balanced_accuracy_ci_high": high}
    metrics = {
        "holdout": holdout,
        "threshold": threshold,
        "n_unseen": {"real": int((y_test == 0).sum()), "ai": int((y_test == 1).sum())},
        "val": evaluate(y_val, gen_val, scores_val, threshold),
        "unseen": unseen,
    }
    logger.info(f"Held-out {holdout}: unseen balanced accuracy {unseen['balanced_accuracy']:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{holdout}.json").write_text(json.dumps(metrics, indent=2), newline="\n")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{info['backbone']}-logo-{holdout}"):
        mlflow.log_params({f"train.{k}": v for k, v in p.items()})
        mlflow.log_params({"holdout": holdout, "embed.backbone": info["backbone"]})
        mlflow.set_tags(
            {
                "backbone_weights": info["weights"],
                "data_revision": load_params("data")["revision"],
                "embeddings_md5": dvc_hash("data/processed/embeddings"),
            }
        )
        mlflow.log_metric("threshold", threshold)
        for part in ("val", "unseen"):
            mlflow.log_metrics({f"{part}_{k}": v for k, v in metrics[part].items()})

    logger.success(f"Leave-one-generator-out metrics written to {output_dir / holdout}.json")


if __name__ == "__main__":
    app()
