"""DVC stage `embed`: turn every preprocessed image into a feature vector (transfer learning).

A pretrained backbone is used *frozen* as a feature extractor. Two families are supported:

- ImageNet-pretrained torchvision classifiers (e.g. `resnet18`, `resnet50`): the final
  classification layer is replaced by the identity, so each image maps to the backbone's
  penultimate activations.
- CLIP image encoders (`clip_*`, via open_clip): each image maps to its CLIP image embedding.
  CLIP is trained on web-scale image-text pairs rather than ImageNet labels, and its features are
  known to transfer well to detecting generated images.

Extracting embeddings once per split keeps the expensive part (the backbone forward pass) out of
`train`, so many classifier configurations can be tried and tracked in MLflow without touching
the images again.

Images are already centre-cropped and resized by `preprocess`, so only tensor conversion and the
backbone's normalisation are applied here. `build_backbone` + `image_transform` must be reused at
serving time (requirement FR-5).
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import open_clip
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

# CLIP backbones: name in params.yaml -> (open_clip architecture, pretrained tag). The original
# OpenAI weights were trained with QuickGELU, hence the `-quickgelu` architecture.
CLIP_BACKBONES = {"clip_vit_b_32": ("ViT-B-32-quickgelu", "openai")}


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, str | None]:
    """Load backbone `name` (torchvision model or key of CLIP_BACKBONES) as a frozen feature
    extractor.

    Returns the model (eval mode, no gradients) and the name of the loaded weights.
    """
    if name in CLIP_BACKBONES:
        arch, tag = CLIP_BACKBONES[name]
        model = open_clip.create_model(arch, pretrained=tag if pretrained else None).visual
        weights = f"open_clip:{arch}/{tag}" if pretrained else None
    else:
        tv_weights = models.get_model_weights(name).DEFAULT if pretrained else None
        model = models.get_model(name, weights=tv_weights)
        _drop_last_linear(model)
        weights = str(tv_weights) if tv_weights else None
    model.eval().requires_grad_(False)
    return model, weights


def _drop_last_linear(model: nn.Module) -> None:
    """Replace the final (classification) linear layer with the identity."""
    name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, nn.Identity())


def image_transform(backbone: str) -> v2.Compose:
    """Preprocessed PIL image -> float tensor normalised as `backbone` expects."""
    if backbone in CLIP_BACKBONES:
        mean, std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )


class ImageFiles(Dataset):
    def __init__(self, paths: list[Path], transform: v2.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


@torch.inference_mode()
def embed_images(
    model: nn.Module,
    paths: list[Path],
    transform: v2.Compose,
    batch_size: int,
    num_workers: int = 0,
    desc: str = "",
) -> np.ndarray:
    """Embed images in order; returns a float32 array of shape (len(paths), dim)."""
    loader = DataLoader(
        ImageFiles(paths, transform), batch_size=batch_size, num_workers=num_workers
    )
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
    transform = image_transform(p["backbone"])
    logger.info(f"Backbone {p['backbone']} ({weights})")

    output_dir.mkdir(parents=True, exist_ok=True)
    info = {"backbone": p["backbone"], "weights": weights, "n_images": {}}
    for split in SPLITS:
        meta = pd.read_csv(splits_dir / f"{split}.csv")
        paths = [images_dir / f for f in meta["file"]]
        X = embed_images(model, paths, transform, p["batch_size"], num_workers, desc=split)
        np.savez(
            output_dir / f"{split}.npz",
            X=X,
            y=meta["label_a"].to_numpy(),  # 0 = real, 1 = AI-generated
            generator=meta["label_b"].to_numpy(),  # 0 = real, 1-5 = generator id
            image_id=meta["image_id"].to_numpy(dtype=str),
        )
        info["n_images"][split] = len(meta)
    info["dim"] = int(X.shape[1])

    (output_dir / "info.json").write_text(json.dumps(info, indent=2), newline="\n")
    logger.success(f"Embeddings ({info['dim']}-d) written to {output_dir}: {info['n_images']}")


if __name__ == "__main__":
    app()
