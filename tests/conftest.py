"""Shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    """Log every MLflow run of a test to a throw-away SQLite store, never to DagsHub (`.env`)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("DVC_STAGE", raising=False)
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
