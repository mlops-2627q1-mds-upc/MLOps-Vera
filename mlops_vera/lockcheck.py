"""Check that dvc.lock is up to date with the code and the parameters in Git (`make lockcheck`).

If a stage's code, parameters or command changed after the stage last ran, the outputs on the
DVC remote (the model, the metrics) no longer come from the code in Git: someone forgot
`dvc repro`, or `dvc commit` after a change that leaves the outputs as they are (a comment, for
example). Two checks find this without the DVC data, so they also run on a fresh clone in CI:

- `dvc status`, for the files that Git versions. The data that DVC manages is ignored by Git: it
  is missing in a fresh clone and may be out of date on a laptop, so its status is left out;
- dvc.lock itself: each data dependency must have the hash that dvc.lock records for the stage
  that writes it, otherwise that stage ran again but the stages after it did not.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import typer
import yaml

from mlops_vera.config import PROJ_ROOT

app = typer.Typer()

# Entries of `dvc status` that do not mean a mismatch with dvc.lock
NOT_STALE = {"always changed"}


def _file_changes(status: dict) -> list[tuple[str, str, str | dict]]:
    """(stage, path, state) of every changed dependency and output in `dvc status --json`."""
    return [
        (stage, Path(path).as_posix(), state)
        for stage, changes in status.items()
        for change in changes
        if isinstance(change, dict)
        for files in change.values()
        for path, state in files.items()
    ]


def stale_entries(status: dict, ignored: set[str]) -> list[str]:
    """What changed since dvc.lock was written, leaving out the paths in `ignored`."""
    entries = [
        f"{stage}: {change}"
        for stage, changes in status.items()
        for change in changes
        if isinstance(change, str) and change not in NOT_STALE
    ]
    for stage, path, state in _file_changes(status):
        if isinstance(state, dict):  # parameters: {section or key: state}
            entries += [f"{stage}: {path} {key} {value}" for key, value in state.items()]
        elif path not in ignored:
            entries.append(f"{stage}: {path} {state}")
    return entries


def lock_mismatches(lock: dict) -> list[str]:
    """Data dependencies whose hash in dvc.lock is not that of the stage that writes them."""
    stages = (lock or {}).get("stages", {})
    written = {
        out["path"]: out.get("md5") for stage in stages.values() for out in stage.get("outs", [])
    }
    return [
        f"{name}: {dep['path']} changed upstream"
        for name, stage in stages.items()
        for dep in stage.get("deps", [])
        if dep["path"] in written and dep.get("md5") != written[dep["path"]]
    ]


def _dvc_status(repo: Path) -> dict:
    """`dvc status --json` of the project in `repo`."""
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "--json"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip() or "{}")


def _git_ignored(repo: Path, paths: list[str]) -> set[str]:
    """The `paths` that Git ignores, i.e. the data that DVC manages."""
    result = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"],
        cwd=repo,
        input="\0".join(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):  # 1: none of the paths is ignored
        raise RuntimeError(f"git check-ignore failed: {result.stderr.strip()}")
    return {Path(path).as_posix() for path in result.stdout.split("\0") if path}


def stale_stages(repo: Path = PROJ_ROOT) -> list[str]:
    """Everything in `repo` that dvc.lock does not match, as `stage: what changed` lines."""
    status = _dvc_status(repo)
    paths = sorted({path for _, path, state in _file_changes(status) if isinstance(state, str)})
    ignored = _git_ignored(repo, paths) if paths else set()
    lock_path = repo / "dvc.lock"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    return stale_entries(status, ignored) + lock_mismatches(lock)


@app.command()
def main(repo: Path = PROJ_ROOT):
    """Fail if a stage's code, parameters or inputs changed without `dvc repro`."""
    entries = stale_stages(repo)
    if not entries:
        print("dvc.lock is up to date with the code and the parameters in Git.")
        return
    # On GitHub Actions, an error annotation shows the failure on the pull request
    prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") == "true" else ""
    print(f"{prefix}dvc.lock is out of date with the code or the parameters in Git:")
    for entry in entries:
        print(f"  {entry}")
    print(
        "Run `uv run dvc repro` (or `uv run dvc commit` if the outputs stay the same), "
        "then commit dvc.lock."
    )
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
