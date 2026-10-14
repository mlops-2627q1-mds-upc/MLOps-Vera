"""Tests for the MLflow tracking shared by the DVC stages (throw-away store, see conftest)."""

import time

from codecarbon import OfflineEmissionsTracker
import mlflow
import pandas as pd
import pytest
import yaml

from mlops_vera import tracking
from mlops_vera.tracking import dvc_hash, flatten, stage_run, track_emissions

LOCK = {
    "stages": {
        "embed": {"outs": [{"path": "data/processed/embeddings", "md5": "abc.dir"}]},
        "train": {"outs": [{"path": "models/classifier.joblib", "md5": "def"}]},
    }
}


def test_dvc_hash_reads_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "PROJ_ROOT", tmp_path)
    assert dvc_hash("data/processed/embeddings") is None  # no dvc.lock yet
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(LOCK))
    assert dvc_hash("data/processed/embeddings") == "abc.dir"
    assert dvc_hash("data/processed/splits") is None


def test_flatten_joins_nested_keys():
    assert flatten({"train": {"n": 1, "x": {"y": 2}}, "dim": 3}) == {
        "train_n": 1,
        "train_x_y": 2,
        "dim": 3,
    }


def test_stage_run_tags_the_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "PROJ_ROOT", tmp_path)
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(LOCK))

    with stage_run("embed", "test", inputs=("data/processed/embeddings",)) as run:
        pass

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert run.info.run_name == "embed"
    assert tags["dvc.stage"] == "embed"
    assert tags["embeddings_md5"] == "abc.dir"
    assert tags["data_revision"] == tracking.load_params("data")["revision"]
    assert "dvc.exp_name" not in tags


def test_stage_run_follows_dvc_environment(monkeypatch):
    # Set by DVC: the stage address (foreach item included) and the `dvc exp run -n` name
    monkeypatch.setenv("DVC_STAGE", "logo@sd3")
    monkeypatch.setenv("DVC_EXP_NAME", "clip-C0.1")

    with stage_run("logo", "test", run_name="toy-logo-sd3") as run:
        pass

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert run.info.run_name == "toy-logo-sd3"
    assert tags["dvc.stage"] == "logo@sd3"
    assert tags["dvc.exp_name"] == "clip-C0.1"


@track_emissions
def toy_stage(fail_before_run=False, fail_in_run=False, seconds=0.0):
    """Like a pipeline stage: work first, then log the MLflow run."""
    if fail_before_run:
        raise ValueError("bad input")
    time.sleep(seconds)
    with stage_run("toy", "test") as run:
        if fail_in_run:
            raise RuntimeError("stage crashed")
    return run.info.run_id


def test_track_emissions_logs_to_the_stage_run():
    run = mlflow.get_run(toy_stage())

    assert run.info.status == "FINISHED"
    assert run.data.metrics["codecarbon.emissions_g_co2eq"] == pytest.approx(2e-3)  # kg -> g
    assert run.data.metrics["codecarbon.energy_kwh"] == pytest.approx(1e-5)
    assert run.data.metrics["codecarbon.duration_s"] == pytest.approx(3.0)
    assert run.data.tags["codecarbon.country"] == "ESP"


def test_track_emissions_measures_failed_stages():
    with pytest.raises(RuntimeError):
        toy_stage(fail_in_run=True)
    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.info.status == "FAILED"  # logging the emissions keeps the status
    assert "codecarbon.energy_kwh" in run.data.metrics

    with pytest.raises(ValueError):  # no MLflow run to log to: only the CSV gets the row
        toy_stage(fail_before_run=True)


def test_track_emissions_with_codecarbon(tmp_path, monkeypatch):
    # The real tracker (offline: no network), to catch changes in the CodeCarbon API
    monkeypatch.setattr(tracking, "OfflineEmissionsTracker", OfflineEmissionsTracker)

    metrics = mlflow.get_run(toy_stage(seconds=1.0)).data.metrics

    assert metrics["codecarbon.duration_s"] >= 1.0  # the work before `stage_run` is measured
    assert metrics["codecarbon.energy_kwh"] > 0
    assert metrics["codecarbon.emissions_g_co2eq"] > 0
    emissions = pd.read_csv(tmp_path / ".codecarbon" / "emissions.csv")
    assert emissions["project_name"].tolist() == ["test_tracking"]
