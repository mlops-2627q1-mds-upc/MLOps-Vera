"""Behavioural tests of the trained model (Milestone 3: model testing).

Unlike the unit tests of `train`, these run on the real pipeline artefacts: the saved classifier,
the embeddings and the images. They check that the model

- meets the model-card targets on validation (MR-1 to MR-3) and detects every generator;
- classifies at the threshold tuned on validation (MR-5), and retraining reproduces it (MR-6);
- gives the same embeddings when an image goes through the serving path (raw image ->
  `preprocess_image` -> backbone), i.e. no training/serving skew (FR-5);
- keeps its decision under perturbations that do not change whether an image is real or
  AI-generated: mirroring, JPEG re-compression, a brightness change and a smaller upload.

They are skipped when the artefacts are missing (run `uv run dvc pull` first), and can be
deselected with `pytest -m "not model"`.
"""

import io

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import pytest
import torch

from mlops_vera.config import (
    CLASSIFIER_PATH,
    EMBEDDINGS_DIR,
    RAW_DEFACTIFY_DIR,
    SPLITS_DIR,
    load_params,
)
from mlops_vera.features import preprocess_image
from mlops_vera.modeling.embed import build_backbone, image_transform
from mlops_vera.modeling.train import GENERATORS, build_model, evaluate, load_split

ARTEFACTS = [CLASSIFIER_PATH, EMBEDDINGS_DIR / "train.npz", EMBEDDINGS_DIR / "val.npz"]
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not all(p.exists() for p in ARTEFACTS), reason="model artefacts missing: dvc pull"
    ),
]

# Model-card targets on validation (MR-1 to MR-3) and minimum detection rate per generator
TARGETS = {"balanced_accuracy": 0.85, "f1_macro": 0.85, "recall_real": 0.80, "pr_auc_real": 0.90}
MIN_GENERATOR_RECALL = 0.80
# Images per class sampled from validation for the image-level tests, and the minimum share of
# decisions that must not change under each perturbation (measured: 0.94-0.98)
N_PER_CLASS = 60
MIN_AGREEMENT = 0.90


@pytest.fixture(name="bundle", scope="module")
def fixture_bundle() -> dict:
    return joblib.load(CLASSIFIER_PATH)


@pytest.fixture(name="val", scope="module")
def fixture_val():
    return load_split(EMBEDDINGS_DIR, "val")


def _scores(bundle: dict, X: np.ndarray) -> np.ndarray:
    return bundle["model"].predict_proba(X)[:, 1]


def test_model_meets_model_card_targets_on_validation(bundle, val):
    X, y, generator = val
    metrics = evaluate(y, generator, _scores(bundle, X), bundle["threshold"])
    for name, target in TARGETS.items():
        assert metrics[name] >= target, f"{name} = {metrics[name]:.3f} < {target}"


@pytest.mark.parametrize("generator", list(GENERATORS.values()))
def test_model_detects_every_generator(bundle, val, generator):
    X, y, gen = val
    recall = evaluate(y, gen, _scores(bundle, X), bundle["threshold"])[f"recall_{generator}"]
    assert recall >= MIN_GENERATOR_RECALL, f"only {recall:.1%} of {generator} images detected"


def test_saved_model_classifies_at_the_tuned_threshold(bundle, val):
    X = val[0]
    assert 0 < bundle["threshold"] < 1 and bundle["threshold"] != 0.5  # tuned, not default
    expected = (_scores(bundle, X) >= bundle["threshold"]).astype(int)
    np.testing.assert_array_equal(bundle["model"].predict(X), expected)


def test_retraining_reproduces_the_saved_model(bundle, val):
    X_train, y_train, _ = load_split(EMBEDDINGS_DIR, "train")
    retrained = build_model(load_params("train")).fit(X_train, y_train)
    # Each machine's BLAS rounds the last float32 digits differently, so a model trained
    # elsewhere differs by ~1e-6; 1e-4 absorbs that and still catches real changes.
    np.testing.assert_allclose(
        retrained.predict_proba(val[0])[:, 1], _scores(bundle, val[0]), atol=1e-4
    )


# ------------------------------------------------------------------ image-level (serving path)


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@pytest.fixture(name="encoder", scope="module")
def fixture_encoder(bundle):
    """Backbone + transform of the saved model: preprocessed PIL images -> embeddings."""
    try:
        model, _ = build_backbone(bundle["backbone"])
    except OSError as err:  # weights not cached and no network
        pytest.skip(f"backbone weights unavailable: {err}")
    transform = image_transform(bundle["backbone"])

    @torch.inference_mode()
    def encode(images: list[Image.Image]) -> np.ndarray:
        return model(torch.stack([transform(im) for im in images])).numpy()

    return encode


@pytest.fixture(name="sample", scope="module")
def fixture_sample(val):
    """Balanced sample of validation images: positions in the split, raw images, labels."""
    meta = pd.read_csv(SPLITS_DIR / "val.csv")
    if not (RAW_DEFACTIFY_DIR / meta["file"].iloc[0]).exists():
        pytest.skip("raw images missing: dvc pull data/raw/defactify")
    rng = np.random.default_rng(0)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(val[1] == c), N_PER_CLASS, replace=False) for c in (0, 1)]
    )
    raws = []
    for f in meta["file"].iloc[idx]:
        with Image.open(RAW_DEFACTIFY_DIR / f) as im:
            raws.append(im.convert("RGB"))
    return idx, raws, val[1][idx]


def _serve(raw: Image.Image) -> Image.Image:
    """What the model sees for an uploaded image: the `preprocess` stage, JPEG included."""
    p = load_params("preprocess")
    return _jpeg(preprocess_image(raw, p["img_size"]), p["jpeg_quality"])


def test_serving_path_reproduces_the_training_embeddings(encoder, sample, val):
    idx, raws, _ = sample
    np.testing.assert_allclose(encoder([_serve(r) for r in raws]), val[0][idx], atol=1e-4)


PERTURBATIONS = {
    "mirror": ImageOps.mirror,
    "jpeg_q75": lambda im: _jpeg(im, 75),
    "jpeg_q50": lambda im: _jpeg(im, 50),
    "brightness_+10%": lambda im: ImageEnhance.Brightness(im).enhance(1.1),
}


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_decision_is_invariant_to_label_preserving_perturbations(bundle, encoder, sample, name):
    _, raws, _ = sample
    images = [_serve(r) for r in raws]
    before = _scores(bundle, encoder(images)) >= bundle["threshold"]
    after = _scores(bundle, encoder([PERTURBATIONS[name](im) for im in images]))
    agreement = (before == (after >= bundle["threshold"])).mean()
    assert agreement >= MIN_AGREEMENT, f"{name} changes {1 - agreement:.1%} of the decisions"


def test_decision_is_invariant_to_upload_resolution(bundle, encoder, sample):
    _, raws, _ = sample
    full = _scores(bundle, encoder([_serve(r) for r in raws])) >= bundle["threshold"]
    halved = [r.resize((r.width // 2, r.height // 2)) for r in raws]
    half = _scores(bundle, encoder([_serve(r) for r in halved])) >= bundle["threshold"]
    assert (full == half).mean() >= MIN_AGREEMENT
