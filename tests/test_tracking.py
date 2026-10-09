"""Tests for the MLflow tracking shared by the DVC stages (throw-away store, see conftest)."""

import mlflow
import yaml

from mlops_vera import tracking
from mlops_vera.tracking import dvc_hash, flatten, stage_run

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
