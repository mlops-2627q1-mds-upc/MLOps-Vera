"""Serving path: an uploaded image -> P(AI) and a real / AI-generated decision (FR-1, FR-2, FR-5).

Chains the pipeline's own functions, so a served image is seen exactly as the training images:

1. `preprocess_image`, then a JPEG round trip at `jpeg_quality`: the `preprocess` stage writes
   JPEGs, so the model has only ever seen JPEG-compressed images;
2. the frozen backbone and its normalisation (`build_backbone`, `image_transform`), as in `embed`;
3. the head saved by `train`, whose P(AI) is compared with the threshold tuned on validation.

Images are only processed in memory, never written to disk (NFR-7). The API loads one
`Predictor` at start-up and reuses it for every request. For a quick check from the shell:

    python -m mlops_vera.modeling.predict path/to/image.jpg
"""

from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.base import ClassifierMixin
import torch
from torch import nn
import typer

from mlops_vera.config import CLASSIFIER_PATH, load_params
from mlops_vera.features import preprocess_image
from mlops_vera.modeling.embed import build_backbone, image_transform

app = typer.Typer()

# Decision labels, indexed by the binary target of `train` (label_a: 0 = real, 1 = AI)
LABELS = ("real", "ai_generated")


@dataclass(frozen=True)
class Prediction:
    """Decision on one image: the label, P(AI) and the threshold it was compared with."""

    label: str
    score: float
    threshold: float


def to_model_input(image: Image.Image, img_size: int, jpeg_quality: int) -> Image.Image:
    """Image as the `preprocess` stage writes it and `embed` reads it back, without the disk."""
    buf = io.BytesIO()
    preprocess_image(image, img_size).save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


class Predictor:
    """Backbone + head of a trained model, applied to raw images."""

    def __init__(self, bundle: dict, backbone: nn.Module, preprocess: dict):
        """`bundle` is the dict saved by `train`; `preprocess` the params of that stage."""
        self.classifier: ClassifierMixin = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.backbone_name = bundle["backbone"]
        self.backbone = backbone
        self.transform = image_transform(self.backbone_name)
        self.img_size = preprocess["img_size"]
        self.jpeg_quality = preprocess["jpeg_quality"]

    @classmethod
    def load(cls, model_path: Path = CLASSIFIER_PATH, pretrained: bool = True) -> "Predictor":
        """Load the model saved by `train` and its backbone (downloaded on first use)."""
        bundle = joblib.load(model_path)
        backbone, _ = build_backbone(bundle["backbone"], pretrained)
        return cls(bundle, backbone, load_params("preprocess"))

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Backbone features of raw images, as the `embed` stage computes them."""
        inputs = [to_model_input(im, self.img_size, self.jpeg_quality) for im in images]
        batch = torch.stack([self.transform(im) for im in inputs])
        return self.backbone(batch).numpy().astype(np.float32)

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """Decisions on raw images, in order."""
        scores = self.classifier.predict_proba(self.embed(images))[:, 1]
        # Same rule as the saved classifier's predict(), which only returns the label
        return [
            Prediction(LABELS[int(s >= self.threshold)], float(s), self.threshold) for s in scores
        ]

    def predict(self, image: Image.Image) -> Prediction:
        """Decision on one raw image."""
        return self.predict_batch([image])[0]


@app.command()
def main(images: list[Path], model_path: Path = CLASSIFIER_PATH):
    """Print the decision on each image as JSON lines."""
    predictor = Predictor.load(model_path)
    for path in images:
        with Image.open(path) as im:
            prediction = predictor.predict(im)
        print(json.dumps({"image": str(path)} | asdict(prediction)))


if __name__ == "__main__":
    app()
