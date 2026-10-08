"""Tests for the `learning_curve` stage (head re-trained on growing caption fractions).

Uses small synthetic embeddings, so no network is needed.
"""

import mlflow
import numpy as np
import pandas as pd

from mlops_vera.modeling import learning_curve as lc
from tests.test_train import _synthetic_split

PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "seed": 0}


def _with_captions(n_per_source: int, rng: np.random.Generator):
    """Synthetic split where caption j has one image per source (real + 5 generators)."""
    X, y, generator = _synthetic_split(n_per_source, rng)
    captions = np.array([f"caption {j}" for j in np.tile(np.arange(n_per_source), 6)])
    return (X, y, generator), captions


def test_caption_subsample_keeps_whole_captions():
    captions = np.repeat([f"c{i}" for i in range(10)], 4)
    keep = lc.caption_subsample(captions, 0.3, seed=0)
    assert keep.sum() == 12  # 3 of 10 captions, all 4 images of each
    kept = set(captions[keep])
    assert len(kept) == 3 and all(keep[captions == c].all() for c in kept)
    np.testing.assert_array_equal(keep, lc.caption_subsample(captions, 0.3, seed=0))  # seeded


def test_learning_curve_averages_draws_per_fraction():
    rng = np.random.default_rng(0)
    train, captions = _with_captions(20, rng)
    val, _ = _with_captions(10, rng)
    table = lc.learning_curve(train, val, captions, PARAMS, [0.25, 0.5, 1.0], n_seeds=3)
    assert list(table["fraction"]) == [0.25, 0.5, 1.0]
    assert list(table["n_images"]) == [30, 60, 120]  # 5, 10 and 20 captions x 6 images
    assert (table["n_real"] == table["n_images"] / 6).all()
    assert table.loc[table["fraction"] == 1.0, "balanced_accuracy_sd"].item() == 0.0  # one fit
    assert (table["balanced_accuracy"] > 0.9).all()  # separable toy data


def test_main_writes_the_learning_curve(tmp_path, monkeypatch):
    params = {"train": PARAMS, "learning_curve": {"fractions": [0.5, 1.0], "n_seeds": 2}}
    monkeypatch.setattr(lc, "load_params", lambda section: params[section])
    emb, splits = tmp_path / "embeddings", tmp_path / "splits"
    emb.mkdir()
    splits.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 20), ("val", 10)):
        (X, y, generator), captions = _with_captions(n, rng)
        image_id = np.array([f"{split}_{i}" for i in range(len(y))])
        np.savez(emb / f"{split}.npz", X=X, y=y, generator=generator, image_id=image_id)
        pd.DataFrame({"image_id": image_id, "caption": captions}).to_csv(
            splits / f"{split}.csv", index=False
        )

    lc.main(embeddings_dir=emb, splits_dir=splits, output_dir=tmp_path, experiment="test")

    table = pd.read_csv(tmp_path / "learning_curve.csv")
    assert list(table["fraction"]) == [0.5, 1.0]
    assert {"n_images", "balanced_accuracy", "roc_auc", "pr_auc_real", "f1_macro"} <= set(table)
    assert {"balanced_accuracy_sd", "pr_auc_real_sd"} <= set(table)

    # One MLflow step per training-caption percentage
    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.data.tags["dvc.stage"] == "learning_curve"
    assert run.data.params["learning_curve.n_seeds"] == "2"
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_balanced_accuracy")
    assert [m.step for m in history] == [50, 100]
    assert [m.value for m in history] == list(table["balanced_accuracy"])
    sizes = mlflow.MlflowClient().get_metric_history(run.info.run_id, "train_n_images")
    assert [m.value for m in sizes] == list(table["n_images"])
