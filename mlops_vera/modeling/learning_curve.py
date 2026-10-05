"""DVC stage `learning_curve`: is the training subsample large enough?

Re-trains the selected head (the hyper-parameters of the `train` stage) on growing fractions of
the training captions and evaluates each fit on the full validation split, with the threshold
tuned on validation as in `train`. Fractions keep whole captions, like `data.n_captions` does, and
each fraction below 100% is averaged over several random draws. If the validation metrics flatten
well before 100%, more images would barely improve the model and the subsample is large enough.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mlops_vera.config import EMBEDDINGS_DIR, METRICS_DIR, SPLITS_DIR, load_params
from mlops_vera.modeling.captions import load_captions
from mlops_vera.modeling.train import build_model, evaluate, load_split, tune_threshold

app = typer.Typer()
METRICS = ["balanced_accuracy", "roc_auc", "pr_auc_real", "f1_macro"]

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def caption_subsample(captions: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Mask keeping every image of a random `fraction` of the captions."""
    unique = np.unique(captions)
    n = max(1, round(fraction * len(unique)))
    chosen = np.random.default_rng(seed).choice(unique, size=n, replace=False)
    return np.isin(captions, chosen)


def learning_curve(
    train: Split, val: Split, captions: np.ndarray, params: dict, fractions: list, n_seeds: int
) -> pd.DataFrame:
    """Validation metrics of the head trained on each fraction, averaged over `n_seeds` draws
    (a single fit at 100%, where every draw is the same)."""
    X_train, y_train, _ = train
    X_val, y_val, gen_val = val
    rows = []
    for fraction in fractions:
        for seed in range(n_seeds if fraction < 1 else 1):
            keep = caption_subsample(captions, fraction, seed)
            model = build_model(params).fit(X_train[keep], y_train[keep])
            scores = model.predict_proba(X_val)[:, 1]
            metrics = evaluate(y_val, gen_val, scores, tune_threshold(y_val, scores))
            n_real = int((y_train[keep] == 0).sum())
            rows.append({"fraction": fraction, "n_images": int(keep.sum()), "n_real": n_real})
            rows[-1] |= {k: metrics[k] for k in METRICS}
    runs = pd.DataFrame(rows)
    table = runs.groupby("fraction").mean().reset_index()
    sd = runs.groupby("fraction")[METRICS].std().fillna(0.0).add_suffix("_sd").reset_index()
    return table.merge(sd, on="fraction")


@app.command()
def main(
    embeddings_dir: Path = EMBEDDINGS_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_dir: Path = METRICS_DIR,
):
    p, p_lc = load_params("train"), load_params("learning_curve")
    table = learning_curve(
        load_split(embeddings_dir, "train"),
        load_split(embeddings_dir, "val"),
        load_captions(embeddings_dir, splits_dir, "train"),
        p,
        p_lc["fractions"],
        p_lc["n_seeds"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "learning_curve.csv", index=False, lineterminator="\n")
    logger.success(f"Learning curve written to {output_dir / 'learning_curve.csv'}:\n{table}")


if __name__ == "__main__":
    app()
