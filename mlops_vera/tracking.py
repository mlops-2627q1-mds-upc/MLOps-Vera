"""MLflow tracking shared by every DVC stage.

Each stage logs one run, so an execution of the pipeline can be followed end to end in MLflow:

- the experiment groups the stages by purpose (`vera-data`, `vera-baselines`, `vera-logo`,
  `vera-learning-curve`);
- the `dvc.stage` tag names the stage (`logo@sd3`, ...), taken from the `DVC_STAGE` variable
  that DVC sets for every command;
- the `<input>_md5` tags record the DVC hash of each input, so a run is tied to the exact data
  version it read (MLflow adds the Git commit, i.e. the code and params.yaml);
- `dvc.exp_name` names the DVC experiment when the stage runs inside `dvc exp run -n <name>`,
  linking the run to its row in `dvc exp show`;
- the `codecarbon.*` metrics record the energy and CO2 emissions of the whole stage (NFR-4),
  measured with CodeCarbon by decorating its `main` with `track_emissions`. Each stage also
  appends a row to `.codecarbon/emissions.csv`.

Runs go to DagsHub when `.env` sets MLFLOW_TRACKING_URI, else to the local ./mlflow.db.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import functools
import os
from pathlib import Path
import time

from codecarbon import OfflineEmissionsTracker
from codecarbon.output import EmissionsData
from loguru import logger
import mlflow
from mlflow.entities import Metric, Run, RunTag
from mlflow.tracking import MlflowClient
import yaml

from mlops_vera.config import PROJ_ROOT, load_params

EMISSIONS_DIR = PROJ_ROOT / ".codecarbon"  # git-ignored: one CSV row per stage run

_stage_runs: list[str] = []  # MLflow runs opened by the stage that `track_emissions` measures


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
        _stage_runs.append(run.info.run_id)
        yield run


def track_emissions[**P, T](main: Callable[P, T]) -> Callable[P, T]:
    """Measure the energy and CO2 of a whole stage: decorates its `main`.

    The stages log to MLflow only once their work is done, so the measurement spans the whole
    `main` and is then added to the run that `stage_run` opened. Offline mode: the carbon
    intensity comes from `codecarbon.country_iso_code` in params.yaml instead of an IP
    geolocation request. A failed stage is measured too."""

    @functools.wraps(main)
    def measured(*args: P.args, **kwargs: P.kwargs) -> T:
        stage = os.environ.get("DVC_STAGE", main.__module__.rsplit(".", 1)[-1])
        p = load_params("codecarbon")
        EMISSIONS_DIR.mkdir(exist_ok=True)
        tracker = OfflineEmissionsTracker(
            project_name=stage,
            country_iso_code=p["country_iso_code"],
            measure_power_secs=p["measure_power_secs"],
            output_dir=str(EMISSIONS_DIR),
            log_level="warning",
        )
        _stage_runs.clear()
        tracker.start()
        try:
            return main(*args, **kwargs)
        finally:
            tracker.stop()
            log_emissions(stage, tracker.final_emissions_data, _stage_runs)

    return measured


def log_emissions(stage: str, data: EmissionsData, run_ids: list[str]) -> None:
    """Add a CodeCarbon measurement to MLflow runs `run_ids` as `codecarbon.*` metrics and tags.

    The runs may have ended: the client logs to them without changing their status."""
    logger.info(
        f"{stage}: {data.emissions * 1000:.3f} g CO2eq, {data.energy_consumed * 1000:.3f} Wh "
        f"in {data.duration:.0f} s"
    )
    metrics = {
        "codecarbon.emissions_g_co2eq": data.emissions * 1000,
        "codecarbon.energy_kwh": data.energy_consumed,
        "codecarbon.cpu_energy_kwh": data.cpu_energy,
        "codecarbon.gpu_energy_kwh": data.gpu_energy,
        "codecarbon.ram_energy_kwh": data.ram_energy,
        "codecarbon.cpu_power_w": data.cpu_power,
        "codecarbon.duration_s": data.duration,
    }
    tags = {
        "codecarbon.country": data.country_iso_code,
        "codecarbon.cpu_model": data.cpu_model,
        "codecarbon.version": data.codecarbon_version,
    }
    timestamp = int(time.time() * 1000)
    client = MlflowClient()
    for run_id in run_ids:
        client.log_batch(
            run_id,
            metrics=[Metric(k, float(v), timestamp, 0) for k, v in metrics.items()],
            tags=[RunTag(k, str(v)) for k, v in tags.items()],
        )
