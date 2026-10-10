"""DVC stage `preprocess`: normalise every image so no non-semantic shortcut leaks the label.

- Centre-crop to a square  -> removes the aspect-ratio shortcut (fakes are ~100% square).
- Resize to `img_size`     -> removes the resolution shortcut.
- Re-encode as JPEG (fixed quality) -> removes format/compression differences.

The same `preprocess_image` function must be reused at serving time (requirement FR-5). The run
is logged to MLflow (experiment `vera-data`) with an example caption group after preprocessing.
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer

from mlops_vera.config import PREPROCESSED_DIR, RAW_DEFACTIFY_DIR, load_params
from mlops_vera.tracking import stage_run

app = typer.Typer()


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB copy of `img`. Transparent areas (RGBA, LA or palette images with transparency) are
    composited on white: converting directly would keep whatever colour they hide."""
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return img.convert("RGB")


def preprocess_image(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to square and resize to `size`x`size` RGB."""
    img = to_rgb(img)
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.Resampling.BICUBIC)


def plot_caption_group(meta: pd.DataFrame, images_dir: Path) -> Figure:
    """The preprocessed images of the first caption: one real photo and its generated siblings
    should be indistinguishable in size and format, only in content."""
    group = meta[meta["caption"] == meta["caption"].iloc[0]].sort_values("label_b")
    fig = Figure(figsize=(2 * len(group), 2.4), layout="constrained")
    for ax, row in zip(fig.subplots(1, len(group), squeeze=False)[0], group.itertuples()):
        with Image.open(images_dir / row.file) as im:
            ax.imshow(im)
        ax.set_title("real" if row.label_b == 0 else f"AI ({row.label_b})", fontsize=9)
        ax.axis("off")
    return fig


@app.command()
def main(
    input_dir: Path = RAW_DEFACTIFY_DIR,
    output_dir: Path = PREPROCESSED_DIR,
    experiment: str = "vera-data",
):
    """Crop, resize and re-encode every raw image (params.yaml: `preprocess`)."""
    p = load_params("preprocess")
    meta = pd.read_csv(input_dir / "metadata.csv")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    files = []
    for row in tqdm(meta.itertuples(), total=len(meta), desc="preprocess"):
        rel = f"images/{row.image_id}.jpg"
        with Image.open(input_dir / row.file) as im:
            out = preprocess_image(im, p["img_size"])
        out.save(output_dir / rel, format="JPEG", quality=p["jpeg_quality"])
        files.append(rel)

    meta = meta.rename(columns={"width": "orig_width", "height": "orig_height"})
    meta["file"] = files
    meta.drop(columns=["format"]).to_csv(
        output_dir / "metadata.csv", index=False, lineterminator="\n"
    )

    with stage_run("preprocess", experiment, inputs=("data/raw/defactify",)):
        mlflow.log_params({f"preprocess.{k}": v for k, v in p.items()})
        mlflow.log_metric("n_images", len(meta))
        mlflow.log_figure(plot_caption_group(meta, output_dir), "example_caption_group.png")
    logger.success(f"Preprocessed {len(meta)} images -> {output_dir}")


if __name__ == "__main__":
    app()
