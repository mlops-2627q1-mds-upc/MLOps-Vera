"""DVC stage `preprocess`: normalise every image so no non-semantic shortcut leaks the label.

- Centre-crop to a square  -> removes the aspect-ratio shortcut (fakes are ~100% square).
- Resize to `img_size`     -> removes the resolution shortcut.
- Re-encode as JPEG (fixed quality) -> removes format/compression differences.

The same `preprocess_image` function must be reused at serving time (requirement FR-5).
"""

from pathlib import Path

from loguru import logger
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer

from mlops_vera.config import PREPROCESSED_DIR, RAW_DEFACTIFY_DIR, load_params

app = typer.Typer()


def preprocess_image(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to square and resize to `size`x`size` RGB."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.Resampling.BICUBIC)


@app.command()
def main(input_dir: Path = RAW_DEFACTIFY_DIR, output_dir: Path = PREPROCESSED_DIR):
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
    meta.drop(columns=["format"]).to_csv(output_dir / "metadata.csv", index=False)
    logger.success(f"Preprocessed {len(meta)} images -> {output_dir}")


if __name__ == "__main__":
    app()
