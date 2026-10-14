"""Shared fixtures."""

from types import SimpleNamespace

import pytest

from mlops_vera import tracking

# CodeCarbon measurement returned by FakeTracker (only the fields that `log_emissions` reads)
FAKE_EMISSIONS = SimpleNamespace(
    emissions=2e-6,  # kg CO2eq
    energy_consumed=1e-5,  # kWh
    cpu_energy=6e-6,
    gpu_energy=0.0,
    ram_energy=4e-6,
    cpu_power=7.5,
    duration=3.0,
    country_iso_code="ESP",
    cpu_model="Toy CPU",
    codecarbon_version="0.0.0",
)


class FakeTracker:
    """Stands in for CodeCarbon's tracker, which spends seconds detecting the hardware."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.final_emissions_data = None

    def start(self):
        """Nothing to measure."""

    def stop(self):
        """Return the canned emissions (kg CO2eq), like `EmissionsTracker.stop`."""
        self.final_emissions_data = FAKE_EMISSIONS
        return FAKE_EMISSIONS.emissions


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    """Log every MLflow run of a test to a throw-away SQLite store, never to DagsHub (`.env`)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("DVC_STAGE", raising=False)
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)


@pytest.fixture(autouse=True)
def fake_codecarbon(tmp_path, monkeypatch):
    """Replace CodeCarbon with FakeTracker and keep its CSV out of the project."""
    monkeypatch.setattr(tracking, "OfflineEmissionsTracker", FakeTracker)
    monkeypatch.setattr(tracking, "EMISSIONS_DIR", tmp_path / ".codecarbon")
