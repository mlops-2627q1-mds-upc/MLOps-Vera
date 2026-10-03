"""Captions of the embedded images.

The splits are grouped by caption: images generated from the same caption show the same scene, so
analyses that resample or subsample images (bootstrap intervals, learning curves) work on whole
captions. The embeddings store image ids and the split CSVs map them to captions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_captions(embeddings_dir: Path, splits_dir: Path, split: str) -> np.ndarray:
    """Caption of every image of `split`, in the row order of its embeddings."""
    with np.load(embeddings_dir / f"{split}.npz") as d:
        image_ids = d["image_id"]
    captions = pd.read_csv(splits_dir / f"{split}.csv").set_index("image_id")["caption"]
    return captions.loc[image_ids].to_numpy()
