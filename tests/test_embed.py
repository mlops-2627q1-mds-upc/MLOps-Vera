"""Tests for the `embed` stage (frozen backbone feature extraction).

Backbones are built with random weights (`pretrained=False`), so no network is needed.
"""

from PIL import Image
import pytest
import torch
from torch import nn

from mlops_vera.modeling.embed import build_backbone, embed_images, image_transform


@pytest.mark.parametrize(("name", "dim"), [("resnet18", 512), ("mobilenet_v3_small", 1024)])
def test_backbone_outputs_penultimate_features(name, dim):
    model, weights = build_backbone(name, pretrained=False)
    assert weights is None
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(isinstance(m, nn.Linear) and m.out_features == 1000 for m in model.modules())
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, dim)


def test_image_transform_normalises_to_float_tensor():
    x = image_transform()(Image.new("RGB", (224, 224), color=(255, 255, 255)))
    assert x.dtype == torch.float32
    assert x.shape == (3, 224, 224)
    assert torch.allclose(x[0], torch.full((224, 224), (1 - 0.485) / 0.229))


def test_embed_images_keeps_order_and_shape(tmp_path):
    colours = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]
    paths = []
    for i, c in enumerate(colours):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (224, 224), color=c).save(paths[-1])
    model, _ = build_backbone("resnet18", pretrained=False)

    X = embed_images(model, paths, batch_size=2)
    assert X.shape == (3, 512)
    assert X.dtype == "float32"
    # Same result one image at a time -> batching does not reorder or mix images
    for i, p in enumerate(paths):
        assert torch.allclose(
            torch.from_numpy(embed_images(model, [p], 1)[0]), torch.from_numpy(X[i]), atol=1e-5
        )
