"""DVC stage `image_metadata`: describe every image as one row of a table (data validation input).

Great Expectations validates tabular data, so each image is turned into its metadata:

- raw image: whether it opens, format, colour mode, size, aspect ratio and file size;
- preprocessed image: the same, plus mean brightness, contrast (brightness standard deviation)
  and an md5 hash of the file (to find duplicate images);
- labels, caption and the split the image was assigned to.

`validate.py` checks this table against the expectations of the data (requirement DR-8).
"""

import hashlib
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer

from mlops_vera.config import INTERIM_DATA_DIR, PREPROCESSED_DIR, RAW_DEFACTIFY_DIR, SPLITS_DIR
from mlops_vera.tracking import stage_run

app = typer.Typer()
IMAGE_METADATA_PATH = INTERIM_DATA_DIR / "image_metadata.csv"
SPLITS = ("train", "val", "test")


def raw_image_metadata(path: Path) -> dict:
    """Format, colour mode, size and file size of a raw image, or `raw_readable=False`."""
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated or corrupt files without decoding all pixels
        with Image.open(path) as im:
            fmt, mode, (w, h) = im.format, im.mode, im.size
    except (OSError, SyntaxError):
        return {"raw_readable": False}
    return {
        "raw_readable": True,
        "raw_format": fmt,
        "raw_mode": mode,
        "raw_width": w,
        "raw_height": h,
        "raw_aspect_ratio": w / h,
        "raw_file_kb": path.stat().st_size / 1024,
    }


def processed_image_metadata(path: Path) -> dict:
    """Format, mode, size, brightness statistics and md5 of a preprocessed image."""
    try:
        data = path.read_bytes()
        with Image.open(path) as im:
            fmt, mode, (w, h) = im.format, im.mode, im.size
            gray = np.asarray(im.convert("L"), dtype=np.float32)
    except OSError:
        return {"readable": False}
    return {
        "readable": True,
        "format": fmt,
        "mode": mode,
        "width": w,
        "height": h,
        "file_kb": len(data) / 1024,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "md5": hashlib.md5(data).hexdigest(),
    }


def split_assignment(splits_dir: Path) -> pd.Series:
    """image_id -> split name, from the split stage's CSV files."""
    parts = [pd.read_csv(splits_dir / f"{s}.csv", usecols=["image_id"]) for s in SPLITS]
    return pd.concat(p.assign(split=s) for p, s in zip(parts, SPLITS)).set_index("image_id")[
        "split"
    ]


def build_metadata(raw_dir: Path, processed_dir: Path, splits_dir: Path) -> pd.DataFrame:
    """One row per image: labels, split, raw and preprocessed image metadata."""
    raw = pd.read_csv(raw_dir / "metadata.csv")
    processed = pd.read_csv(processed_dir / "metadata.csv").set_index("image_id")["file"]
    rows = []
    for r in tqdm(raw.itertuples(), total=len(raw), desc="metadata"):
        row = {
            "image_id": r.image_id,
            "caption": r.caption,
            "label_a": r.label_a,
            "label_b": r.label_b,
        }
        row |= raw_image_metadata(raw_dir / r.file)
        if r.image_id in processed.index:
            row |= processed_image_metadata(processed_dir / processed[r.image_id])
        else:
            row["readable"] = False
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta["split"] = meta["image_id"].map(split_assignment(splits_dir))
    return meta


@app.command()
def main(
    raw_dir: Path = RAW_DEFACTIFY_DIR,
    processed_dir: Path = PREPROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_path: Path = IMAGE_METADATA_PATH,
    experiment: str = "vera-data",
):
    """Write the metadata table of every raw and preprocessed image."""
    meta = build_metadata(raw_dir, processed_dir, splits_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_path, index=False, lineterminator="\n")

    inputs = ("data/raw/defactify", "data/processed/defactify_224", "data/processed/splits")
    with stage_run("image_metadata", experiment, inputs=inputs):
        mlflow.log_metrics(
            {
                "n_images": len(meta),
                "n_raw_unreadable": int((~meta["raw_readable"].astype(bool)).sum()),
                "n_unreadable": int((~meta["readable"].astype(bool)).sum()),
                "n_duplicate_files": int(meta["md5"].dropna().duplicated().sum()),
            }
        )
    logger.success(f"Metadata of {len(meta)} images written to {output_path}")


if __name__ == "__main__":
    app()
