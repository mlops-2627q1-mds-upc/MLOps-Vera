"""Tests for the data validation stages (`image_metadata` and `validate_data`).

The expectation suites are checked on a small synthetic metadata table: valid data must pass
every suite, and each kind of broken data must make the suite meant to catch it fail.
"""

import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from mlops_vera import validate
from mlops_vera.config import load_params
from mlops_vera.metadata import build_metadata
from mlops_vera.validate import build_suites, build_tables, validate_tables

IMG_SIZE = 224
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


@pytest.fixture(name="meta")
def fixture_meta() -> pd.DataFrame:
    """20 captions x 6 sources (real + 5 generators), split 14/3/3 captions as in params.yaml.

    Brightness, contrast and file size depend on the caption only, so they carry no
    information about the label (no shortcut)."""
    rows = []
    for c in range(20):
        split = "train" if c < 14 else "val" if c < 17 else "test"
        for g in range(6):
            rows.append(
                {
                    "image_id": f"img_{c:02d}_{g}",
                    "caption": f"caption {c}",
                    "label_a": int(g > 0),
                    "label_b": g,
                    "split": split,
                    "raw_readable": True,
                    "raw_format": "JPEG",
                    "raw_mode": "RGB",
                    "raw_width": 640 if g == 0 else 512,
                    "raw_height": 480 if g == 0 else 512,
                    "raw_aspect_ratio": 640 / 480 if g == 0 else 1.0,
                    "raw_file_kb": 80.0,
                    "readable": True,
                    "format": "JPEG",
                    "mode": "RGB",
                    "width": IMG_SIZE,
                    "height": IMG_SIZE,
                    "file_kb": 10.0 + c % 7,
                    "brightness": 100.0 + c,
                    "contrast": 40.0 + c % 5,
                    "md5": f"{c:02d}{g}".ljust(32, "0"),
                }
            )
    return pd.DataFrame(rows)


def _validate(meta: pd.DataFrame) -> dict:
    suites = build_suites(load_params("validate"), len(meta), IMG_SIZE)
    return validate_tables(build_tables(meta, FRACTIONS), suites)


def _failed(results: dict) -> set[tuple[str, str, str | None]]:
    """(suite, expectation, column) of every failed expectation."""
    return {
        (suite, f["expectation"], f["column"])
        for suite, r in results.items()
        for f in r["failed"]
    }


def test_valid_metadata_passes_every_suite(meta):
    results = _validate(meta)
    assert _failed(results) == set()
    assert all(r["success"] and r["evaluated"] > 0 for r in results.values())


def _non_square(m):
    m.loc[0, "width"] = 200


def _inconsistent_labels(m):
    m.loc[0, "label_b"] = 3  # a real photo (label_a 0) attributed to a generator


def _caption_leak(m):
    m.loc[m["caption"] == "caption 0", "split"] = ["train"] * 5 + ["test"]


def _black_image(m):
    m.loc[3, "brightness"] = 0.0


def _flat_image(m):
    m.loc[3, "contrast"] = 0.5


def _duplicate_file(m):
    m.loc[1, "md5"] = m.loc[2, "md5"]


def _unreadable_raw(m):
    m.loc[4, "raw_readable"] = False


def _missing_generator_in_test(m):
    m.drop(m.index[(m["split"] == "test") & (m["label_b"] == 5)], inplace=True)


def _size_shortcut(m):
    m["file_kb"] = np.where(m["label_a"] == 1, 8.0, 20.0)  # AI images are always smaller


@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        (_non_square, ("images", "expect_column_values_to_be_between", "width")),
        (
            _inconsistent_labels,
            ("images", "expect_column_pair_values_to_be_in_set", "label_a"),
        ),
        (_caption_leak, ("captions", "expect_column_values_to_be_between", "n_splits")),
        (_black_image, ("images", "expect_column_values_to_be_between", "brightness")),
        (_flat_image, ("images", "expect_column_values_to_be_between", "contrast")),
        (_duplicate_file, ("images", "expect_column_values_to_be_unique", "md5")),
        (_unreadable_raw, ("images", "expect_column_values_to_be_in_set", "raw_readable")),
        (
            _missing_generator_in_test,
            ("splits", "expect_column_values_to_be_between", "n_sources"),
        ),
        (_size_shortcut, ("shortcuts", "expect_column_values_to_be_between", "auc")),
    ],
)
def test_broken_metadata_fails_the_right_expectation(meta, corrupt, expected):
    corrupt(meta)
    results = _validate(meta.reset_index(drop=True))
    assert expected in _failed(results)
    assert not results[expected[0]]["success"]


def test_shortcut_table_measures_label_signal(meta):
    table = validate.shortcut_table(meta.assign(file_kb=meta["label_a"] * 10.0))
    auc = table.set_index("feature")["auc"]
    assert auc["file_kb"] == 1.0  # the label is fully given away
    assert auc["brightness"] == auc["contrast"] == 0.5  # constant within a caption


def test_main_writes_summary_and_fails_on_bad_data(meta, tmp_path):
    meta_path, source = tmp_path / "meta.csv", tmp_path / "source.json"
    source.write_text(json.dumps({"n_images": len(meta)}))
    meta.to_csv(meta_path, index=False)
    out = tmp_path / "validation.json"

    validate.main(metadata_path=meta_path, source_path=source, output_path=out, docs=False)
    summary = json.loads(out.read_text())
    assert summary["success"] and set(summary["suites"]) == set(build_tables(meta, FRACTIONS))

    _non_square(meta)
    meta.to_csv(meta_path, index=False)
    with pytest.raises(validate.typer.Exit):
        validate.main(metadata_path=meta_path, source_path=source, output_path=out, docs=False)
    assert not json.loads(out.read_text())["success"]


def test_build_metadata_describes_raw_and_processed_images(tmp_path):
    raw, processed, splits = tmp_path / "raw", tmp_path / "processed", tmp_path / "splits"
    for d in (raw / "images", processed / "images", splits):
        d.mkdir(parents=True)
    Image.new("RGB", (640, 480), (200, 10, 10)).save(raw / "images" / "a.jpg")
    (raw / "images" / "b.jpg").write_bytes(b"not an image")  # corrupt raw file
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128)).save(processed / "images" / "a.jpg")
    pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "file": ["images/a.jpg", "images/b.jpg"],
            "caption": ["c", "c"],
            "label_a": [0, 1],
            "label_b": [0, 2],
        }
    ).to_csv(raw / "metadata.csv", index=False)
    pd.DataFrame({"image_id": ["a"], "file": ["images/a.jpg"]}).to_csv(
        processed / "metadata.csv", index=False
    )
    pd.DataFrame({"image_id": ["a"]}).to_csv(splits / "train.csv", index=False)
    for s in ("val", "test"):
        pd.DataFrame({"image_id": []}).to_csv(splits / f"{s}.csv", index=False)

    meta = build_metadata(raw, processed, splits).set_index("image_id")

    a, b = meta.loc["a"], meta.loc["b"]
    assert (a["raw_width"], a["raw_height"], a["raw_format"]) == (640, 480, "JPEG")
    assert a["raw_aspect_ratio"] == pytest.approx(4 / 3)
    assert (a["width"], a["height"], a["mode"], a["readable"]) == (IMG_SIZE, IMG_SIZE, "RGB", True)
    assert a["brightness"] == pytest.approx(128, abs=1) and a["contrast"] < 1
    assert a["split"] == "train" and len(a["md5"]) == 32
    assert not b["raw_readable"] and not b["readable"] and pd.isna(b["split"])
