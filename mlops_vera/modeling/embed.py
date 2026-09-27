"""DVC stage `embed`: turn every preprocessed image into a feature vector (transfer learning).

An ImageNet-pretrained torchvision backbone is used *frozen*: its final classification layer is
replaced by the identity, so each image maps to the backbone's penultimate activations. Extracting
them once per split keeps the expensive part (the CNN forward pass) out of `train`, so many
classifier configurations can be tried and tracked in MLflow without touching the images again.

Images are already centre-cropped and resized by `preprocess`, so only tensor conversion and
ImageNet normalisation are applied here. `build_backbone` + `image_transform` must be reused at
serving time (requirement FR-5).
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm
import typer

from mlops_vera.config import EMBEDDINGS_DIR, PREPROCESSED_DIR, SPLITS_DIR, load_params

app = typer.Typer()
SPLITS = ("train", "val", "test")

# Normalisation shared by all ImageNet-pretrained torchvision classifiers
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, str | None]:
    """Load torchvision model `name` as a frozen feature extractor.

    Returns the model (eval mode, no gradients) and the name of the loaded weights.
    """
    weights = models.get_model_weights(name).DEFAULT if pretrained else None
    model = models.get_model(name, weights=weights)
    _drop_last_linear(model)
    model.eval().requires_grad_(False)
    return model, (str(weights) if weights else None)


def _drop_last_linear(model: nn.Module) -> None:
    """Replace the final (classification) linear layer with the identity."""
    name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, nn.Identity())


def image_transform() -> v2.Compose:
    """Preprocessed PIL image -> normalised float tensor, as the backbones expect."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageFiles(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = image_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


@torch.inference_mode()
def embed_images(
    model: nn.Module, paths: list[Path], batch_size: int, num_workers: int = 0, desc: str = ""
) -> np.ndarray:
    """Embed images in order; returns a float32 array of shape (len(paths), dim)."""
    loader = DataLoader(ImageFiles(paths), batch_size=batch_size, num_workers=num_workers)
    feats = [model(batch) for batch in tqdm(loader, desc=desc)]
    return torch.cat(feats).numpy().astype(np.float32)


@app.command()
def main(
    images_dir: Path = PREPROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_dir: Path = EMBEDDINGS_DIR,
    num_workers: int = 2,
):
    p = load_params("embed")
    model, weights = build_backbone(p["backbone"])
    logger.info(f"Backbone {p['backbone']} ({weights})")

    output_dir.mkdir(parents=True, exist_ok=True)
    info = {"backbone": p["backbone"], "weights": weights, "n_images": {}}
    for split in SPLITS:
        meta = pd.read_csv(splits_dir / f"{split}.csv")
        paths = [images_dir / f for f in meta["file"]]
        X = embed_images(model, paths, p["batch_size"], num_workers, desc=split)
        np.savez(
            output_dir / f"{split}.npz",
            X=X,
            y=meta["label_a"].to_numpy(),  # 0 = real, 1 = AI-generated
            generator=meta["label_b"].to_numpy(),  # 0 = real, 1-5 = generator id
            image_id=meta["image_id"].to_numpy(dtype=str),
        )
        info["n_images"][split] = len(meta)
    info["dim"] = int(X.shape[1])

    (output_dir / "info.json").write_text(json.dumps(info, indent=2))
    logger.success(f"Embeddings ({info['dim']}-d) written to {output_dir}: {info['n_images']}")


if __name__ == "__main__":
    app()
