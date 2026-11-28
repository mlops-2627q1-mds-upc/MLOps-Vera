"""Tests for the dvc.lock check (`make lockcheck`).

The unit tests feed it each kind of change, as `dvc status --json` and dvc.lock show it; the
end-to-end tests run it with Git and DVC on a small project, cloned without its DVC data, as in CI.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import typer

from mlops_vera import lockcheck
from mlops_vera.lockcheck import lock_mismatches, main, stale_entries, stale_stages

# Paths that Git ignores in the unit tests: the data that DVC manages
IGNORED = {"data/raw", "data/splits", "models/classifier.joblib"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(
            {
                "split": [
                    {"changed deps": {"data/raw": "deleted"}},
                    {"changed outs": {"data/splits": "not in cache"}},
                ]
            },
            [],
            id="data missing, as in a fresh clone",
        ),
        pytest.param(
            {"train": [{"changed outs": {"models/classifier.joblib": "modified"}}]},
            [],
            id="data out of date, as on a laptop",
        ),
        pytest.param({"download": ["always changed"]}, [], id="stage that always runs"),
        pytest.param(
            {
                "split": [
                    {"changed deps": {"mlops_vera/split.py": "modified", "data/raw": "deleted"}}
                ]
            },
            ["split: mlops_vera/split.py modified"],
            id="code changed",
        ),
        pytest.param(
            {"split": [{"changed deps": {"mlops_vera/split.py": "deleted"}}]},
            ["split: mlops_vera/split.py deleted"],
            id="code deleted",
        ),
        pytest.param(
            {"preprocess": [{"changed deps": {"params.yaml": {"preprocess": "modified"}}}]},
            ["preprocess: params.yaml preprocess modified"],
            id="parameters changed",
        ),
        pytest.param(
            {"split": [{"changed outs": {"reports/metrics/split_summary.json": "modified"}}]},
            ["split: reports/metrics/split_summary.json modified"],
            id="metrics in Git changed",
        ),
        pytest.param(
            {"split": [{"changed deps": {"data/raw": "deleted"}}, "changed command"]},
            ["split: changed command"],
            id="command changed",
        ),
    ],
)
def test_only_changes_to_files_in_git_are_stale(status, expected):
    assert stale_entries(status, IGNORED) == expected


def _lock(raw_md5: str, split_dep_md5: str) -> dict:
    """dvc.lock of two stages: `download` writes data/raw, which `split` reads."""
    return {
        "schema": "2.0",
        "stages": {
            "download": {"deps": [], "outs": [{"path": "data/raw", "md5": raw_md5}]},
            "split": {
                "deps": [
                    {"path": "data/raw", "md5": split_dep_md5},
                    {"path": "data/raw/source.json", "md5": "f1"},  # inside an output: skipped
                    {"path": "mlops_vera/split.py", "md5": "c1"},
                ],
                "outs": [{"path": "data/splits", "md5": "s1.dir"}],
            },
        },
    }


def test_lock_with_matching_hashes_has_no_mismatch():
    assert lock_mismatches(_lock("a1.dir", "a1.dir")) == []


def test_stage_that_did_not_rerun_after_its_input_changed_is_stale():
    assert lock_mismatches(_lock("a2.dir", "a1.dir")) == ["split: data/raw changed upstream"]


def test_missing_or_empty_lock_has_no_mismatch():
    assert lock_mismatches({}) == []
    assert lock_mismatches(None) == []


def test_main_lists_the_changes_and_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        lockcheck, "stale_stages", lambda repo: ["train: params.yaml train modified"]
    )
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(typer.Exit) as exc:
        main(Path("."))
    out = capsys.readouterr().out
    assert exc.value.exit_code == 1
    assert out.startswith("dvc.lock is out of date")
    assert "  train: params.yaml train modified\n" in out
    assert "dvc repro" in out


def test_main_annotates_the_failure_on_github_actions(monkeypatch, capsys):
    monkeypatch.setattr(lockcheck, "stale_stages", lambda repo: ["train: changed command"])
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(typer.Exit):
        main(Path("."))
    assert capsys.readouterr().out.startswith("::error::dvc.lock is out of date")


# ----- End to end: Git and DVC on a small project -----

MAKE_CODE = """\
import json

n = json.load(open("params.yaml"))["make"]["n"]
with open("data.txt", "w") as f:
    f.write("x" * n)
with open("metrics.json", "w") as f:
    json.dump({"n": n}, f)
"""

COUNT_CODE = """\
import json

with open("count.json", "w") as f:
    json.dump({"chars": len(open("data.txt").read())}, f)
"""

# `make` writes DVC data (data.txt) and metrics in Git; `count` reads the data.
# JSON is valid YAML, so the commands can quote the interpreter's path safely.
DVC_YAML = """\
stages:
  make:
    cmd: {make}
    deps:
      - make.py
    params:
      - make
    outs:
      - data.txt
    metrics:
      - metrics.json:
          cache: false
  count:
    cmd: {count}
    deps:
      - count.py
      - data.txt
    metrics:
      - count.json:
          cache: false
"""


def _dvc(cwd: Path, *args: str) -> None:
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=cwd, check=True, capture_output=True)


def _cmd(script: str) -> str:
    return json.dumps(f'"{sys.executable}" {script}')


@pytest.fixture(autouse=True)
def no_dvc_analytics(monkeypatch):
    """Keep the DVC commands of these tests offline."""
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")


@pytest.fixture(name="template", scope="module")
def fixture_template(tmp_path_factory) -> Path:
    """The project with both stages run once, so its dvc.lock is up to date."""
    root = tmp_path_factory.mktemp("project")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DVC_NO_ANALYTICS", "1")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
        _dvc(root, "init", "-q")
        (root / "params.yaml").write_text('{"make": {"n": 3}}\n', encoding="utf-8")
        (root / "make.py").write_text(MAKE_CODE, encoding="utf-8")
        (root / "count.py").write_text(COUNT_CODE, encoding="utf-8")
        yaml_text = DVC_YAML.format(make=_cmd("make.py"), count=_cmd("count.py"))
        (root / "dvc.yaml").write_text(yaml_text, encoding="utf-8")
        _dvc(root, "repro", "-q")
    return root


@pytest.fixture(name="clone")
def fixture_clone(template, tmp_path) -> Path:
    """Copy of the project without its DVC data, like a fresh clone in CI."""
    clone = Path(shutil.copytree(template, tmp_path / "clone"))
    (clone / "data.txt").unlink()
    shutil.rmtree(clone / ".dvc" / "cache")
    return clone


def test_check_passes_on_a_clone_without_the_dvc_data(clone, capsys):
    main(clone)
    assert "up to date" in capsys.readouterr().out


def test_check_fails_when_the_code_changes_without_dvc_repro(clone, capsys):
    with open(clone / "make.py", "a", encoding="utf-8") as f:
        f.write("# a comment changes the hash too\n")
    with pytest.raises(typer.Exit) as exc:
        main(clone)
    assert exc.value.exit_code == 1
    assert "  make: make.py modified\n" in capsys.readouterr().out


def test_check_fails_when_a_parameter_changes(clone):
    (clone / "params.yaml").write_text('{"make": {"n": 4}}\n', encoding="utf-8")
    assert stale_stages(clone) == ["make: params.yaml make modified"]


def test_check_fails_when_metrics_in_git_change(clone):
    (clone / "metrics.json").write_text('{"n": 99}', encoding="utf-8")
    assert stale_stages(clone) == ["make: metrics.json modified"]


def test_check_fails_when_only_the_first_stage_runs_again(clone):
    (clone / "params.yaml").write_text('{"make": {"n": 4}}\n', encoding="utf-8")
    _dvc(clone, "repro", "-q", "--single-item", "make")
    assert stale_stages(clone) == ["count: data.txt changed upstream"]


def test_check_passes_again_after_dvc_repro(clone):
    (clone / "params.yaml").write_text('{"make": {"n": 4}}\n', encoding="utf-8")
    _dvc(clone, "repro", "-q")
    assert stale_stages(clone) == []
