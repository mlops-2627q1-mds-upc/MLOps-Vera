"""Tests for the serving path (`predict`: raw image -> decision).

A toy backbone (mean colour of the image) and a head trained on it stand in for CLIP and the saved
model, so no weights or DVC artefacts are needed. The real model is checked in `test_model.py`.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import pytest
from torch import nn

from mlops_vera.config import load_params
from mlops_vera.features import preprocess_image
from mlops_vera.modeling import predict
from mlops_vera.modeling.predict import LABELS, Predictor, main, to_model_input
from mlops_vera.modeling.train import build_model, with_threshold

TOY_PREPROCESS = {"img_size": 32, "jpeg_quality": 90}
THRESHOLD = 0.4


def _toy_backbone() -> nn.Module:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())


def _grey(level: int) -> Image.Image:
    return Image.new("RGB", (48, 40), color=(level, level, level))


@pytest.fixture(name="bundle")
def fixture_bundle() -> dict:
    """Bundle as saved by `train`, with a head that flags bright images as AI-generated."""
    levels = range(0, 256, 8)
    y = np.array([int(level >= 128) for level in levels])
    untrained = {"model": None, "threshold": THRESHOLD, "backbone": "resnet18"}
    X = Predictor(untrained, _toy_backbone(), TOY_PREPROCESS).embed([_grey(lv) for lv in levels])
    model = build_model(load_params("train")).fit(X, y)
    return untrained | {"model": with_threshold(model, THRESHOLD, X, y)}


@pytest.fixture(name="predictor")
def fixture_predictor(bundle) -> Predictor:
    return Predictor(bundle, _toy_backbone(), TOY_PREPROCESS)


@pytest.fixture(name="model_path")
def fixture_model_path(tmp_path, monkeypatch, bundle) -> Path:
    """`bundle` saved as `train` does; loading it builds the toy backbone instead of CLIP."""
    joblib.dump(bundle, tmp_path / "classifier.joblib")
    monkeypatch.setattr(
        predict, "build_backbone", lambda name, pretrained: (_toy_backbone(), None)
    )
    return tmp_path / "classifier.joblib"


def test_model_input_matches_what_embed_reads_from_the_preprocess_stage(tmp_path):
    # Non-square, transparent and noisy: cropping, compositing and JPEG all change the pixels
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (60, 90, 4), dtype=np.uint8), mode="RGBA")
    preprocess_image(image, 32).save(tmp_path / "stage.jpg", format="JPEG", quality=90)
    with Image.open(tmp_path / "stage.jpg") as im:
        expected = np.asarray(im.convert("RGB"))

    served = to_model_input(image, img_size=32, jpeg_quality=90)
    assert served.mode == "RGB" and served.size == (32, 32)
    np.testing.assert_array_equal(np.asarray(served), expected)


def test_predict_returns_label_score_and_threshold(predictor):
    dark, bright = predictor.predict(_grey(10)), predictor.predict(_grey(245))
    assert dark.label == "real" and bright.label == "ai_generated"
    assert 0 <= dark.score < THRESHOLD <= bright.score <= 1
    assert dark.threshold == bright.threshold == THRESHOLD


def test_label_agrees_with_the_saved_classifier(predictor, bundle):
    images = [_grey(level) for level in range(0, 256, 5)]
    expected = bundle["model"].predict(predictor.embed(images))
    assert [p.label for p in predictor.predict_batch(images)] == [LABELS[c] for c in expected]


def test_predict_batch_keeps_the_order_of_the_images(predictor):
    images = [_grey(200), _grey(20), _grey(130), _grey(90)]
    assert predictor.predict_batch(images) == [predictor.predict(im) for im in images]


def test_load_restores_the_saved_model(model_path):
    predictor = Predictor.load(model_path)
    assert predictor.threshold == THRESHOLD
    assert predictor.backbone_name == "resnet18"
    assert predictor.img_size == load_params("preprocess")["img_size"]
    assert predictor.predict(_grey(245)).label == "ai_generated"


def test_main_prints_one_json_line_per_image(tmp_path, capsys, model_path):
    paths = [tmp_path / "dark.png", tmp_path / "bright.png"]
    _grey(10).save(paths[0])
    _grey(245).save(paths[1])

    main(paths, model_path=model_path)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["image"], line["label"]) for line in lines] == [
        (str(paths[0]), "real"),
        (str(paths[1]), "ai_generated"),
    ]
    assert all(line["threshold"] == THRESHOLD for line in lines)
