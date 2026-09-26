"""DVC stage `split`: caption-grouped train/val/test split (requirement DR-4).

The original Defactify splits share captions (leakage found in the EDA), so we re-split:
all images generated from the same caption go to the same split.
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mlops_vera.config import METRICS_DIR, PREPROCESSED_DIR, SPLITS_DIR, load_params

app = typer.Typer()
SPLITS = ("train", "val", "test")


def caption_grouped_split(meta: pd.DataFrame, fractions: dict, seed: int) -> pd.DataFrame:
    captions = np.sort(meta["caption"].unique())
    np.random.default_rng(seed).shuffle(captions)
    n = len(captions)
    n_train = round(fractions["train"] * n)
    n_val = round(fractions["val"] * n)
    groups = {
        "train": captions[:n_train],
        "val": captions[n_train : n_train + n_val],
        "test": captions[n_train + n_val :],
    }
    assign = {c: name for name, caps in groups.items() for c in caps}
    return meta.assign(split=meta["caption"].map(assign))


def summarise(meta: pd.DataFrame) -> dict:
    summary = {}
    for name in SPLITS:
        part = meta[meta["split"] == name]
        summary[name] = {
            "n_images": len(part),
            "n_captions": int(part["caption"].nunique()),
            "n_real": int((part["label_a"] == 0).sum()),
            "n_ai": int((part["label_a"] == 1).sum()),
        }
    return summary


@app.command()
def main(input_dir: Path = PREPROCESSED_DIR, output_dir: Path = SPLITS_DIR):
    p = load_params("split")
    assert abs(p["train"] + p["val"] + p["test"] - 1) < 1e-9, "split fractions must sum to 1"
    meta = caption_grouped_split(pd.read_csv(input_dir / "metadata.csv"), p, p["seed"])

    # Guard: no caption may appear in more than one split
    leaks = meta.groupby("caption")["split"].nunique().gt(1).sum()
    assert leaks == 0, f"{leaks} captions leak across splits"

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        meta[meta["split"] == name].drop(columns="split").to_csv(
            output_dir / f"{name}.csv", index=False
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarise(meta)
    (METRICS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.success(f"Splits written to {output_dir}: {summary}")


if __name__ == "__main__":
    app()
