"""DVC stage `download`: fetch a caption-grouped subsample of Defactify from Hugging Face.

1. Read only the light columns (Caption, Label_A, Label_B) of every parquet shard.
2. Sample `n_captions` captions (optionally only complete real + 5-generator groups).
3. Fetch image bytes only for the parquet row groups that contain selected rows.

Images are stored untouched (raw = immutable); preprocessing happens in `features.py`. The run
is logged to MLflow (experiment `vera-data`) with the resolved dataset revision and the size and
class balance of the sample.
"""

import io
import json
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from mlops_vera.config import RAW_DEFACTIFY_DIR, load_params
from mlops_vera.tracking import stage_run

app = typer.Typer()

META_COLS = ["Caption", "Label_A", "Label_B"]
N_SOURCES = 6  # Label_B: 0 = real, 1..5 = generators


def list_shards(api: HfApi, repo_id: str, revision: str) -> list[str]:
    """Parquet shards of the dataset at `revision`, sorted."""
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def shard_split(shard: str) -> str:
    """'data/train-00000-of-00010.parquet' -> 'train'."""
    return Path(shard).name.split("-")[0]


def read_metadata(open_shard, shards: list[str]) -> pd.DataFrame:
    """Read the light columns of every shard, remembering shard and row position."""
    frames = []
    for shard in tqdm(shards, desc="metadata"):
        with open_shard(shard) as f:
            df = pq.ParquetFile(f).read(columns=META_COLS).to_pandas()
        df["shard"] = shard
        df["row"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def select_captions(
    meta: pd.DataFrame, n_captions: int, seed: int, complete_only: bool = True
) -> pd.DataFrame:
    """Sample whole caption groups so real/fake siblings stay together."""
    if complete_only:
        n_sources = meta.groupby("Caption")["Label_B"].nunique()
        pool = n_sources[n_sources == N_SOURCES].index
    else:
        pool = meta["Caption"].unique()
    pool = np.sort(np.asarray(pool))  # sort first: sampling must not depend on read order
    n = min(n_captions, len(pool))
    if n < n_captions:
        logger.warning(f"Only {n} eligible captions available (requested {n_captions}).")
    chosen = np.random.default_rng(seed).choice(pool, size=n, replace=False)
    return meta[meta["Caption"].isin(set(chosen))].reset_index(drop=True)


def fetch_images(open_shard, selected: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Write the image bytes of the selected rows, reading only the needed row groups."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for shard, rows in tqdm(selected.groupby("shard"), desc="images"):
        wanted = rows.set_index("row")
        with open_shard(shard) as f:
            pf = pq.ParquetFile(f)
            start = 0
            for rg in range(pf.num_row_groups):
                end = start + pf.metadata.row_group(rg).num_rows
                hits = wanted.index[(wanted.index >= start) & (wanted.index < end)]
                if len(hits):
                    images = pf.read_row_group(rg, columns=["Image"]).column("Image")
                    for row in hits:
                        raw = images[int(row) - start].as_py()["bytes"]
                        records.append(_save(raw, shard, int(row), wanted.loc[row], img_dir))
                start = end
    return pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)


def summarise(records: pd.DataFrame) -> dict:
    """Size and class balance of the downloaded sample."""
    return {
        "n_images": len(records),
        "n_captions": int(records["caption"].nunique()),
        "n_real": int((records["label_a"] == 0).sum()),
        "n_ai": int((records["label_a"] == 1).sum()),
        # Raw images not square yet: the aspect-ratio shortcut that `preprocess` removes
        "share_non_square": float((records["width"] != records["height"]).mean()),
    }


def _save(raw: bytes, shard: str, row: int, meta: pd.Series, img_dir: Path) -> dict:
    with Image.open(io.BytesIO(raw)) as im:
        fmt, (w, h) = (im.format or "JPEG").lower(), im.size
    ext = "jpg" if fmt == "jpeg" else fmt
    image_id = f"{Path(shard).stem}_{row:06d}"
    (img_dir / f"{image_id}.{ext}").write_bytes(raw)
    return {
        "image_id": image_id,
        "file": f"images/{image_id}.{ext}",
        "caption": meta["Caption"],
        "label_a": int(meta["Label_A"]),
        "label_b": int(meta["Label_B"]),
        "orig_split": shard_split(shard),
        "width": w,
        "height": h,
        "format": fmt,
    }


@app.command()
def main(output_dir: Path = RAW_DEFACTIFY_DIR, experiment: str = "vera-data"):
    """Download the caption-grouped Defactify subsample defined in params.yaml (`data`)."""
    p = load_params("data")
    api, fs = HfApi(), HfFileSystem()
    sha = api.dataset_info(p["repo_id"], revision=p["revision"]).sha
    logger.info(f"{p['repo_id']} @ {sha}")

    def open_shard(shard: str):
        return fs.open(f"datasets/{p['repo_id']}@{sha}/{shard}", "rb")

    shards = list_shards(api, p["repo_id"], sha)
    meta = read_metadata(open_shard, shards)
    selected = select_captions(meta, p["n_captions"], p["seed"], p["complete_only"])
    logger.info(f"Selected {len(selected)} images from {selected['Caption'].nunique()} captions")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = fetch_images(open_shard, selected, output_dir)
    records.to_csv(output_dir / "metadata.csv", index=False)
    source = {"repo_id": p["repo_id"], "revision": sha, "n_images": len(records)}
    (output_dir / "source.json").write_text(json.dumps(source, indent=2))

    with stage_run("download", experiment):
        mlflow.log_params({f"data.{k}": v for k, v in p.items()})
        mlflow.set_tag("data_sha", sha)  # `revision` resolved by the Hub
        mlflow.log_metrics({"n_shards": len(shards), "n_candidate_images": len(meta)})
        mlflow.log_metrics(summarise(records))
        mlflow.log_dict(source, "source.json")
    logger.success(f"Saved {len(records)} raw images to {output_dir}")


if __name__ == "__main__":
    app()
