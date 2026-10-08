"""DVC stage `logo@<generator>`: leave-one-generator-out evaluation (requirement MR-4).

Simulates a new generator appearing in production, the drift monitored in Milestone 6:

- The held-out generator is removed from train and val, so neither the classifier head nor the
  tuned decision threshold ever see it. The other four generators are kept, and the head is the
  one of the `train` stage (same hyper-parameters, class-weighted against the new ratio).
- The test set keeps the real images of the test split plus the held-out generator's images
  only. Both come from the caption-grouped test split, so no caption seen in training appears
  in it, and every caption has a real photo and an unseen generated image (content-controlled).
- The unseen balanced accuracy gets a bootstrap 95% confidence interval: each test set has only
  ~190 images per class, so small differences between generators can be noise. The bootstrap
  resamples whole captions, the unit the split is grouped by: images of the same caption show the
  same scene, so their errors are correlated and resampling single images would understate the
  uncertainty.

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

from mlops_vera.config import EMBEDDINGS_DIR, METRICS_DIR, SPLITS_DIR, load_params
from mlops_vera.modeling.captions import load_captions
from mlops_vera.modeling.train import (
    GENERATORS,
    build_model,
    evaluate,
    lineage_tags,
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


def real_or_generator(generator: np.ndarray, g: int) -> np.ndarray:
    """Mask of the real images and the images of generator `g`."""
    return (generator == 0) | (generator == g)


def real_and_generator(split: Split, g: int) -> Split:
    """Keep the real images and the images of generator `g` only."""
    X, y, generator = split
    keep = real_or_generator(generator, g)
    return X[keep], y[keep], generator[keep]


def bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the balanced accuracy, resampling whole `groups`
    (captions) with replacement. Resamples that miss a class are skipped."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == c) for c in np.unique(groups)]
    stats = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) == 2:
            stats.append(balanced_accuracy_score(y[idx], pred[idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


@app.command()
def main(
    holdout: str,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_dir: Path = METRICS_DIR / "logo",
    experiment: str = "vera-logo",
):  # pylint: disable=too-many-locals  # one linear script: filter, fit, evaluate, log
    """Train without generator `holdout` and evaluate on it (one `logo@<generator>` stage)."""
    if holdout not in GENERATOR_IDS:
        raise typer.BadParameter(f"unknown generator {holdout!r}, expected {list(GENERATOR_IDS)}")
    g = GENERATOR_IDS[holdout]
    p, p_logo = load_params("train"), load_params("logo")
    info = json.loads((embeddings_dir / "info.json").read_text())

    X_train, y_train, _ = without_generator(load_split(embeddings_dir, "train"), g)
    X_val, y_val, gen_val = without_generator(load_split(embeddings_dir, "val"), g)
    test = load_split(embeddings_dir, "test")
    X_test, y_test, gen_test = real_and_generator(test, g)
    captions = load_captions(embeddings_dir, splits_dir, "test")[real_or_generator(test[2], g)]

    model = build_model(p).fit(X_train, y_train)
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(y_val, scores_val)
    unseen = evaluate(y_test, gen_test, scores_test, threshold)
    del unseen[f"recall_{holdout}"]  # the only AI images are the held-out ones: = recall_ai
    low, high = bootstrap_ci(
        y_test,
        (scores_test >= threshold).astype(int),
        captions,
        p_logo["n_bootstrap"],
        p_logo["seed"],
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
        mlflow.set_tags(lineage_tags(info))
        mlflow.log_metric("threshold", threshold)
        for part in ("val", "unseen"):
            mlflow.log_metrics({f"{part}_{k}": v for k, v in metrics[part].items()})

    logger.success(f"Leave-one-generator-out metrics written to {output_dir / holdout}.json")


if __name__ == "__main__":
    app()
