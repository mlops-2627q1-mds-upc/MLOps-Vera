"""Tests for the data pipeline (download selection, preprocessing, caption-grouped split).

Uses tiny synthetic parquet shards that mimic the Defactify schema, so no network is needed.
"""

import io

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlops_vera.dataset import fetch_images, read_metadata, select_captions
from mlops_vera.features import preprocess_image
from mlops_vera.split import caption_grouped_split


def _jpeg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(120, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def shards(tmp_path):
    """Two shards, 10 captions x 6 sources, the last caption missing one generator."""
    rows = [
        {"Caption": f"cap {c}", "Label_A": int(b > 0), "Label_B": b}
        for c in range(10)
        for b in range(6)
        if not (c == 9 and b == 5)
    ]
    paths = {}
    for name, part in (("train", rows[:30]), ("test", rows[30:])):
        images = [
            {"bytes": _jpeg(640, 480) if r["Label_B"] == 0 else _jpeg(512, 512), "path": None}
            for r in part
        ]
        table = pa.Table.from_pylist([{**r, "Image": im} for r, im in zip(part, images)])
        path = tmp_path / f"{name}-00000-of-00001.parquet"
        pq.write_table(table, path, row_group_size=7)  # several row groups
        paths[f"data/{path.name}"] = path
    return paths


def test_select_keeps_whole_complete_caption_groups(shards):
    meta = read_metadata(lambda s: open(shards[s], "rb"), sorted(shards))
    sel = select_captions(meta, n_captions=4, seed=0, complete_only=True)
    assert sel["Caption"].nunique() == 4
    assert "cap 9" not in set(sel["Caption"])  # incomplete group excluded
    assert (sel.groupby("Caption").size() == 6).all()


def test_select_is_deterministic(shards):
    meta = read_metadata(lambda s: open(shards[s], "rb"), sorted(shards))
    a = select_captions(meta, 3, seed=42)
    b = select_captions(meta.sample(frac=1, random_state=1), 3, seed=42)
    assert set(a["Caption"]) == set(b["Caption"])


def test_fetch_images_writes_selected_rows(shards, tmp_path):
    open_shard = lambda s: open(shards[s], "rb")  # noqa: E731
    meta = read_metadata(open_shard, sorted(shards))
    sel = select_captions(meta, 5, seed=0)
    out = fetch_images(open_shard, sel, tmp_path / "raw")
    assert len(out) == len(sel)
    assert all((tmp_path / "raw" / f).exists() for f in out["file"])
    real = out[out["label_b"] == 0]
    assert (real["width"] == 640).all() and (real["height"] == 480).all()


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (512, 512), (1024, 1024)])
def test_preprocess_outputs_fixed_square_rgb(size):
    out = preprocess_image(Image.new("L", size), 224)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_split_has_no_caption_leakage():
    meta = pd.DataFrame(
        {"caption": np.repeat([f"c{i}" for i in range(100)], 6), "label_a": [0, 1, 1, 1, 1, 1] * 100}
    )
    out = caption_grouped_split(meta, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert out.groupby("caption")["split"].nunique().max() == 1
    assert out["split"].value_counts().to_dict() == {"train": 420, "val": 90, "test": 90}
