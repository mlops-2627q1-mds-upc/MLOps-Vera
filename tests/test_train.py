"""Tests for the `train` stage (threshold tuning, metrics, end-to-end run with local MLflow).

Uses small synthetic embeddings and a throw-away SQLite MLflow store, so no network is needed.
"""

import json

import joblib
import mlflow
import numpy as np
import pytest

from mlops_vera.config import load_params
from mlops_vera.modeling import train
from mlops_vera.modeling.train import evaluate, main, tune_threshold


def _synthetic_split(n_per_source: int, rng: np.random.Generator, dim: int = 8):
    """Real images (generator 0) centred at -1, AI images (generators 1-5) at +1."""
    generator = np.repeat(np.arange(6), n_per_source)
    y = (generator > 0).astype(int)
    X = rng.normal(loc=np.where(y, 1.0, -1.0)[:, None], scale=0.5, size=(len(y), dim))
    return X.astype(np.float32), y, generator


def test_tune_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])
    t = tune_threshold(y, scores)
    assert 0.3 < t <= 0.6
    assert evaluate(y, np.where(y, 1, 0), scores, t)["balanced_accuracy"] == 1.0


def test_tune_threshold_beats_default_under_imbalance():
    # All scores are high (a model biased towards the 5x majority class): 0.5 labels
    # everything as AI, while the tuned threshold still recovers the real images.
    y = np.array([0, 0] + [1] * 10)
    scores = np.array([0.6, 0.65] + [0.8] * 10)
    t = tune_threshold(y, scores)
    gen = np.where(y, 1, 0)
    assert evaluate(y, gen, scores, 0.5)["recall_real"] == 0.0
    assert evaluate(y, gen, scores, t)["balanced_accuracy"] == 1.0


def test_evaluate_reports_model_card_metrics():
    y = np.array([0, 0, 1, 1, 1, 1])
    generator = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array([0.2, 0.7, 0.9, 0.8, 0.4, 0.9])  # 1 real and 1 SDXL image misclassified
    m = evaluate(y, generator, scores, threshold=0.5)
    assert m["recall_real"] == 0.5
    assert m["recall_ai"] == 0.75
    assert m["balanced_accuracy"] == pytest.approx(0.625)
    assert m["recall_sd21"] == 1.0 and m["recall_sdxl"] == 0.5
    assert "recall_sd3" not in m  # generator absent from the data -> no NaN metric
    assert all(isinstance(v, float) for v in m.values())


@pytest.mark.parametrize(
    ("class_weight", "run_name"),
    [("balanced", "toy-logreg-balanced"), (None, "toy-logreg-unweighted")],
)
def test_main_trains_and_logs_to_mlflow(tmp_path, monkeypatch, class_weight, run_name):
    # Pin the head's class weighting so the test doesn't depend on the value chosen in params.yaml
    params = load_params()
    params["train"]["class_weight"] = class_weight
    monkeypatch.setattr(train, "load_params", lambda section: params[section])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(0)
    emb = tmp_path / "embeddings"
    emb.mkdir()
    for split, n in (("train", 30), ("val", 10)):
        X, y, generator = _synthetic_split(n, rng)
        np.savez(emb / f"{split}.npz", X=X, y=y, generator=generator)
    (emb / "info.json").write_text(json.dumps({"backbone": "toy", "weights": None}))

    main(
        embeddings_dir=emb,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        experiment="test",
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["val"]["balanced_accuracy"] > 0.95
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["backbone"] == "toy" and bundle["threshold"] == metrics["threshold"]

    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.info.run_name == run_name
    assert run.data.params["train.class_weight"] == str(class_weight)
    assert run.data.metrics["val_balanced_accuracy"] == metrics["val"]["balanced_accuracy"]
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"val_confusion_matrix.png", "val_pr_curve.png"} <= artifacts
