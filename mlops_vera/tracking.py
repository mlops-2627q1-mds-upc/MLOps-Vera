"""MLflow tracking shared by every DVC stage.

Each stage logs one run, so an execution of the pipeline can be followed end to end in MLflow:

- the experiment groups the stages by purpose (`vera-data`, `vera-baselines`, `vera-logo`,
  `vera-learning-curve`);
- the `dvc.stage` tag names the stage (`logo@sd3`, ...), taken from the `DVC_STAGE` variable
  that DVC sets for every command;
- the `<input>_md5` tags record the DVC hash of each input, so a run is tied to the exact data
  version it read (MLflow adds the Git commit, i.e. the code and params.yaml);
- `dvc.exp_name` names the DVC experiment when the stage runs inside `dvc exp run -n <name>`,
  linking the run to its row in `dvc exp show`.

Runs go to DagsHub when `.env` sets MLFLOW_TRACKING_URI, else to the local ./mlflow.db.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

import mlflow
from mlflow.entities import Run
import yaml

from mlops_vera.config import PROJ_ROOT, load_params


def dvc_hash(path: str) -> str | None:
    """md5 that dvc.lock records for output `path`: ties the run to an exact data version.

    `dvc repro` updates dvc.lock after each stage, so the hash of an upstream output is current
    while a downstream stage runs."""
    lock_path = PROJ_ROOT / "dvc.lock"
    if not lock_path.exists():
        return None
    lock = yaml.safe_load(lock_path.read_text())
    for stage in lock["stages"].values():
        for out in stage.get("outs", []):
            if out["path"] == path:
                return out["md5"]
    return None


def flatten(d: dict, prefix: str = "") -> dict:
    """{"train": {"n": 1}} -> {"train_n": 1}, for logging nested metrics."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat |= flatten(v, f"{key}_")
        else:
            flat[key] = v
    return flat


@contextmanager
def stage_run(
    stage: str, experiment: str, run_name: str | None = None, inputs: tuple[str, ...] = ()
) -> Iterator[Run]:
    """Start the MLflow run of DVC stage `stage`, tagged with its lineage.

    `inputs` are DVC-tracked paths the stage reads; each is tagged as `<name>_md5`."""
    mlflow.set_experiment(experiment)
    tags = {
        "dvc.stage": os.environ.get("DVC_STAGE", stage),
        "data_revision": load_params("data")["revision"],
    }
    tags |= {f"{Path(path).name}_md5": dvc_hash(path) for path in inputs}
    if exp_name := os.environ.get("DVC_EXP_NAME"):
        tags["dvc.exp_name"] = exp_name
    with mlflow.start_run(run_name=run_name or stage, tags=tags) as run:
        yield run
