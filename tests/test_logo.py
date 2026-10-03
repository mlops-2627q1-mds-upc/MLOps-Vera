"""Tests for the leave-one-generator-out stages (`logo@<generator>` and `logo_summary`).

Uses small synthetic embeddings and a throw-away SQLite MLflow store, so no network is needed.
"""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import typer

from mlops_vera.modeling import logo, logo_summary
from mlops_vera.modeling.logo import bootstrap_ci, real_and_generator, without_generator
from tests.test_train import _synthetic_split


def test_without_generator_drops_only_that_generator():
    X, y, gen = _synthetic_split(5, np.random.default_rng(0))
    X_k, y_k, gen_k = without_generator((X, y, gen), 3)
    assert set(gen_k) == {0, 1, 2, 4, 5}
    assert len(X_k) == len(y_k) == len(gen) - 5


def test_real_and_generator_keeps_real_and_holdout_only():
    X, y, gen = _synthetic_split(5, np.random.default_rng(0))
    X_k, y_k, gen_k = real_and_generator((X, y, gen), 3)
    assert set(gen_k) == {0, 3}
    np.testing.assert_array_equal(y_k, gen_k == 3)  # real -> 0, held-out generator -> 1
    np.testing.assert_array_equal(X_k, X[(gen == 0) | (gen == 3)])


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 100)
    pred = np.where(rng.random(200) < 0.8, y, 1 - y)  # ~80% correct
    low, high = bootstrap_ci(y, pred, n_resamples=500, seed=0)
    point = (pred[y == 0] == 0).mean() / 2 + (pred[y == 1] == 1).mean() / 2
    assert low < point < high
    assert bootstrap_ci(y, pred, n_resamples=500, seed=0) == (low, high)  # seeded


def test_main_never_sees_the_holdout_before_testing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(0)
    emb = tmp_path / "embeddings"
    emb.mkdir()
    for split, n in (("train", 30), ("val", 10), ("test", 10)):
        X, y, generator = _synthetic_split(n, rng)
        np.savez(emb / f"{split}.npz", X=X, y=y, generator=generator)
    (emb / "info.json").write_text(json.dumps({"backbone": "toy", "weights": None}))

    logo.main("sd3", embeddings_dir=emb, output_dir=tmp_path / "logo", experiment="test")

    m = json.loads((tmp_path / "logo" / "sd3.json").read_text())
    assert m["holdout"] == "sd3"
    assert m["n_unseen"] == {"real": 10, "ai": 10}
    assert "recall_sd3" not in m["val"]  # threshold tuned without the held-out generator
    u = m["unseen"]
    # Only real + held-out images are tested, so the AI recall is the held-out generator's recall
    assert {k for k in u if k.startswith("recall_")} == {"recall_real", "recall_ai"}
    assert u["balanced_accuracy_ci_low"] <= u["balanced_accuracy"] <= u["balanced_accuracy_ci_high"]
    assert u["balanced_accuracy"] >= 0.9  # separable toy data

    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.info.run_name == "toy-logo-sd3"
    assert run.data.params["holdout"] == "sd3"
    assert run.data.metrics["unseen_balanced_accuracy"] == u["balanced_accuracy"]


def test_main_rejects_unknown_generator(tmp_path):
    with pytest.raises(typer.BadParameter):
        logo.main("stable-diffusion-9", embeddings_dir=tmp_path, output_dir=tmp_path)


def test_summary_checks_mr4_on_the_worst_generator():
    table = pd.DataFrame({"holdout": ["sd21", "sd3", "dalle3"], "balanced_accuracy": [0.9, 0.65, 0.95]})
    s = logo_summary.summarise(table, target=0.70)
    assert s["worst_generator"] == "sd3" and s["min_balanced_accuracy"] == 0.65
    assert s["mean_balanced_accuracy"] == pytest.approx(0.8333, abs=1e-4)
    assert s["meets_mr4"] is False  # the mean passes, the worst case does not
    assert logo_summary.summarise(table.assign(balanced_accuracy=0.8), 0.70)["meets_mr4"] is True


def test_summary_main_writes_table_and_summary(tmp_path, monkeypatch):
    params = {"holdout": ["sd21", "sd3"], "target_balanced_accuracy": 0.70}
    monkeypatch.setattr(logo_summary, "load_params", lambda section: params)
    logo_dir = tmp_path / "logo"
    logo_dir.mkdir()
    for name, ba in (("sd21", 0.9), ("sd3", 0.8)):
        m = {"threshold": 0.5, "unseen": {"balanced_accuracy": ba, "recall_real": 0.9}}
        (logo_dir / f"{name}.json").write_text(json.dumps(m))

    logo_summary.main(logo_dir=logo_dir, output_dir=tmp_path)

    table = pd.read_csv(tmp_path / "logo_table.csv")
    assert list(table["holdout"]) == ["sd21", "sd3"]
    summary = json.loads((tmp_path / "logo_summary.json").read_text())
    assert summary["worst_generator"] == "sd3" and summary["meets_mr4"] is True
