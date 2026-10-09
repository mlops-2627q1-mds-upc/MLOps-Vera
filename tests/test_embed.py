"""Tests for the `embed` stage (frozen backbone feature extraction).

Backbones are built with random weights (`pretrained=False`), so no network is needed.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from mlops_vera.modeling import embed
from mlops_vera.modeling.embed import build_backbone, embed_images, image_transform


@pytest.mark.parametrize(
    ("name", "dim"),
    [("resnet18", 512), ("resnet50", 2048), ("mobilenet_v3_small", 1024), ("clip_vit_b_32", 512)],
)
def test_backbone_outputs_features(name, dim):
    model, weights = build_backbone(name, pretrained=False)
    assert weights is None
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(isinstance(m, nn.Linear) and m.out_features == 1000 for m in model.modules())
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, dim)


@pytest.mark.parametrize(
    ("backbone", "mean", "std"),
    [("resnet18", 0.485, 0.229), ("clip_vit_b_32", 0.48145466, 0.26862954)],
)
def test_image_transform_normalises_as_backbone_expects(backbone, mean, std):
    x = image_transform(backbone)(Image.new("RGB", (224, 224), color=(255, 255, 255)))
    assert x.dtype == torch.float32
    assert x.shape == (3, 224, 224)
    assert torch.allclose(x[0], torch.full((224, 224), (1 - mean) / std))


def test_embed_images_keeps_order_and_shape(tmp_path):
    colours = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]
    paths = []
    for i, c in enumerate(colours):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (224, 224), color=c).save(paths[-1])
    model, _ = build_backbone("resnet18", pretrained=False)
    transform = image_transform("resnet18")

    X = embed_images(model, paths, transform, batch_size=2)
    assert X.shape == (3, 512)
    assert X.dtype == "float32"
    # Same result one image at a time -> batching does not reorder or mix images
    for i, p in enumerate(paths):
        assert torch.allclose(
            torch.from_numpy(embed_images(model, [p], transform, 1)[0]),
            torch.from_numpy(X[i]),
            atol=1e-5,
        )


def test_main_writes_embeddings_and_logs_to_mlflow(tmp_path, monkeypatch):
    # Toy backbone (mean colour of the image), so no weights are downloaded
    toy = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    monkeypatch.setattr(embed, "build_backbone", lambda name: (toy, "toy-weights"))
    monkeypatch.setattr(
        embed, "load_params", lambda section: {"backbone": "resnet18", "batch_size": 2}
    )
    images, splits = tmp_path / "images", tmp_path / "splits"
    images.mkdir()
    splits.mkdir()
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        rows = []
        for i in range(n):
            Image.new("RGB", (32, 32), color=(10 * i, 0, 0)).save(images / f"{split}{i}.jpg")
            rows.append({"image_id": f"{split}{i}", "file": f"{split}{i}.jpg"})
        meta = pd.DataFrame(rows).assign(label_a=[0, 1] * (n // 2), label_b=[0, 3] * (n // 2))
        meta.to_csv(splits / f"{split}.csv", index=False)

    embed.main(
        images_dir=images,
        splits_dir=splits,
        output_dir=tmp_path / "embeddings",
        num_workers=0,
        experiment="test",
    )

    info = json.loads((tmp_path / "embeddings" / "info.json").read_text())
    assert info == {
        "backbone": "resnet18",
        "weights": "toy-weights",
        "n_images": {"train": 4, "val": 2, "test": 2},
        "dim": 3,
    }
    with np.load(tmp_path / "embeddings" / "val.npz") as d:
        assert np.shape(d["X"]) == (2, 3) and list(d["generator"]) == [0, 3]

    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.info.run_name == "resnet18"
    assert run.data.tags["dvc.stage"] == "embed"
    assert run.data.tags["backbone_weights"] == "toy-weights"
    assert run.data.params["embed.batch_size"] == "2"
    assert run.data.metrics["n_images_train"] == 4 and run.data.metrics["dim"] == 3
    assert run.data.metrics["images_per_second"] > 0
