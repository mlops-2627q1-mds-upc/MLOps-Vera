"""DVC stage `validate_data`: check the image metadata with Great Expectations (DR-4, DR-5, DR-8).

Images are not tabular, so the checks run on the table built by `image_metadata` and on three
tables derived from it. Each table has its own expectation suite:

- `images` (one row per image): unique ids, valid and consistent labels, every image readable,
  raw images with plausible sizes, preprocessed images square, RGB, JPEG and not blank, no
  duplicate files, and the 5:1 AI:real ratio of the dataset.
- `captions` (one row per caption): every caption in exactly one split (no leakage, DR-4) and
  with the real photo plus all five generators.
- `splits` (one row per split): class ratio and generators present in every split, and split
  sizes matching params.yaml.
- `shortcuts` (one row per preprocessed-image feature): no metadata feature on its own predicts
  the label, i.e. preprocessing removed non-semantic shortcuts such as the aspect ratio (DR-5).

Thresholds live in the `validate` section of params.yaml. The stage writes a summary to
reports/metrics/data_validation.json, builds the Great Expectations Data Docs (HTML) in
reports/data_docs, logs the run to MLflow and fails if any expectation fails. `embed` depends on
the summary, so DVC stops before embedding data that failed validation.
"""

import json
from pathlib import Path

import great_expectations as gx
from great_expectations import expectations as gxe
from great_expectations.data_context.types.base import ProgressBarsConfig
from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
import typer

from mlops_vera.config import METRICS_DIR, RAW_DEFACTIFY_DIR, REPORTS_DIR, load_params
from mlops_vera.metadata import IMAGE_METADATA_PATH
from mlops_vera.tracking import stage_run

app = typer.Typer()
DATA_DOCS_DIR = REPORTS_DIR / "data_docs"  # Great Expectations HTML report (git-ignored)

SPLITS = ("train", "val", "test")
N_SOURCES = 6  # real photo + 5 generators (Label_B 0..5)
# (label_a, label_b) pairs that make sense: real photos have no generator, AI images have one
CONSISTENT_LABELS = [(0, 0)] + [(1, g) for g in range(1, N_SOURCES)]
SHORTCUT_FEATURES = ("file_kb", "brightness", "contrast")


# ----------------------------------------------------------------------------- derived tables


def caption_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per caption: in how many splits it appears and how many sources it has."""
    return (
        meta.groupby("caption")
        .agg(n_splits=("split", "nunique"), n_sources=("label_b", "nunique"))
        .reset_index()
    )


def split_table(meta: pd.DataFrame, fractions: dict) -> pd.DataFrame:
    """One row per split: real share, number of sources and caption share vs. the target."""
    n_captions = meta["caption"].nunique()
    table = (
        meta.groupby("split")
        .agg(
            n_images=("image_id", "size"),
            real_share=("label_a", lambda a: float((a == 0).mean())),
            n_sources=("label_b", "nunique"),
            caption_share=("caption", lambda c: c.nunique() / n_captions),
        )
        .reset_index()
    )
    table["share_deviation"] = (table["caption_share"] - table["split"].map(fractions)).abs()
    return table


def shortcut_table(meta: pd.DataFrame, features=SHORTCUT_FEATURES) -> pd.DataFrame:
    """How well each feature alone separates real from AI images: ROC-AUC folded to [0.5, 1]
    (0.5 = no signal, 1 = the feature alone gives the label away)."""
    rows = []
    for feature in features:
        auc = roc_auc_score(meta["label_a"], meta[feature])
        rows.append({"feature": feature, "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- expectation suites


def images_suite(p: dict, n_images: int, img_size: int) -> gx.ExpectationSuite:
    """Expectations on each image: schema, labels, raw and preprocessed files, composition."""
    raw_min, raw_max = p["raw_size_range"]
    return gx.ExpectationSuite(
        name="images",
        expectations=[
            gxe.ExpectTableRowCountToEqual(value=n_images),
            gxe.ExpectColumnValuesToBeUnique(column="image_id"),
            *(
                gxe.ExpectColumnValuesToNotBeNull(column=c)
                for c in ("image_id", "caption", "split")
            ),
            gxe.ExpectColumnValuesToBeInSet(column="label_a", value_set=[0, 1]),
            gxe.ExpectColumnValuesToBeInSet(column="label_b", value_set=list(range(N_SOURCES))),
            gxe.ExpectColumnPairValuesToBeInSet(
                column_A="label_a", column_B="label_b", value_pairs_set=CONSISTENT_LABELS
            ),
            gxe.ExpectColumnValuesToBeInSet(column="split", value_set=list(SPLITS)),
            # Raw images: all open, in a supported format, with a plausible size (FR-4)
            gxe.ExpectColumnValuesToBeInSet(column="raw_readable", value_set=[True]),
            gxe.ExpectColumnValuesToBeInSet(
                column="raw_format", value_set=["JPEG", "PNG", "WEBP"]
            ),
            *(
                gxe.ExpectColumnValuesToBeBetween(column=c, min_value=raw_min, max_value=raw_max)
                for c in ("raw_width", "raw_height")
            ),
            # Preprocessed images: what the model sees (DR-5, FR-5)
            gxe.ExpectColumnValuesToBeInSet(column="readable", value_set=[True]),
            gxe.ExpectColumnValuesToBeInSet(column="format", value_set=["JPEG"]),
            gxe.ExpectColumnValuesToBeInSet(column="mode", value_set=["RGB"]),
            *(
                gxe.ExpectColumnValuesToBeBetween(column=c, min_value=img_size, max_value=img_size)
                for c in ("width", "height")
            ),
            gxe.ExpectColumnValuesToBeBetween(column="file_kb", min_value=p["min_file_kb"]),
            gxe.ExpectColumnValuesToBeBetween(
                column="brightness",
                min_value=p["brightness_range"][0],
                max_value=p["brightness_range"][1],
            ),
            gxe.ExpectColumnValuesToBeBetween(column="contrast", min_value=p["min_contrast"]),
            gxe.ExpectColumnValuesToBeUnique(column="md5"),
            # Dataset composition: ~5 AI images per real photo
            gxe.ExpectColumnMeanToBeBetween(
                column="label_a",
                min_value=p["ai_share_range"][0],
                max_value=p["ai_share_range"][1],
            ),
            gxe.ExpectColumnDistinctValuesToEqualSet(
                column="label_b", value_set=list(range(N_SOURCES))
            ),
        ],
    )


def captions_suite() -> gx.ExpectationSuite:
    """Expectations on each caption: one split only (DR-4) and all six sources."""
    return gx.ExpectationSuite(
        name="captions",
        expectations=[
            gxe.ExpectColumnValuesToBeUnique(column="caption"),
            gxe.ExpectColumnValuesToBeBetween(column="n_splits", min_value=1, max_value=1),
            gxe.ExpectColumnValuesToBeBetween(
                column="n_sources", min_value=N_SOURCES, max_value=N_SOURCES
            ),
        ],
    )


def splits_suite(p: dict) -> gx.ExpectationSuite:
    """Expectations on each split: class ratio, generators present and relative size."""
    return gx.ExpectationSuite(
        name="splits",
        expectations=[
            gxe.ExpectColumnDistinctValuesToEqualSet(column="split", value_set=list(SPLITS)),
            gxe.ExpectColumnValuesToBeBetween(
                column="real_share",
                min_value=p["real_share_range"][0],
                max_value=p["real_share_range"][1],
            ),
            gxe.ExpectColumnValuesToBeBetween(
                column="n_sources", min_value=N_SOURCES, max_value=N_SOURCES
            ),
            gxe.ExpectColumnValuesToBeBetween(
                column="share_deviation", max_value=p["max_split_share_deviation"]
            ),
        ],
    )


def shortcuts_suite(p: dict) -> gx.ExpectationSuite:
    """No preprocessed-image feature alone may predict the label (DR-5)."""
    return gx.ExpectationSuite(
        name="shortcuts",
        expectations=[
            gxe.ExpectColumnValuesToBeBetween(column="auc", max_value=p["max_shortcut_auc"]),
        ],
    )


# ------------------------------------------------------------------------------- validation


def validate_tables(tables: dict, suites: dict, docs_dir: Path | None = None) -> dict:
    """Validate each table with the suite of the same name. Returns {name: result summary}.

    Runs in an in-memory Great Expectations context (no `gx/` project folder): the suites are
    defined in code, so Git and DVC version them together with the pipeline. With `docs_dir`,
    the results are also rendered as HTML Data Docs.
    """
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    actions = []
    if docs_dir is not None:
        context.add_data_docs_site(
            site_name="local",
            site_config={
                "class_name": "SiteBuilder",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": str(Path(docs_dir).resolve()),
                },
                "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            },
        )
        actions.append(gx.checkpoint.UpdateDataDocsAction(name="data_docs"))

    source = context.data_sources.add_pandas("metadata")
    summary = {}
    for name, table in tables.items():
        batch = source.add_dataframe_asset(name).add_batch_definition_whole_dataframe(name)
        definition = context.validation_definitions.add(
            gx.ValidationDefinition(name=name, data=batch, suite=context.suites.add(suites[name]))
        )
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(name=name, validation_definitions=[definition], actions=actions)
        )
        result = checkpoint.run(batch_parameters={"dataframe": table})
        summary[name] = summarise(next(iter(result.run_results.values())))
    return summary


def summarise(result) -> dict:
    """Success, counts and the failed expectations (with a few offending values) of a run."""
    failed = []
    for r in result.results:
        if not r.success:
            cfg = r.expectation_config
            failed.append(
                {
                    "expectation": cfg.type,
                    "column": cfg.kwargs.get("column", cfg.kwargs.get("column_A")),
                    "observed": r.result.get("observed_value"),
                    "unexpected_count": r.result.get("unexpected_count"),
                    "examples": r.result.get("partial_unexpected_list", [])[:5],
                }
            )
    stats = result.statistics
    return {
        "success": bool(result.success),
        "evaluated": int(stats["evaluated_expectations"]),
        "successful": int(stats["successful_expectations"]),
        "failed": failed,
    }


def validation_metrics(summary: dict) -> dict:
    """Numeric view of the summary for MLflow: the overall result, each suite and the shortcut
    AUCs."""
    metrics = {"success": float(summary["success"])}
    for name, r in summary["suites"].items():
        metrics[f"{name}_successful"] = r["successful"]
        metrics[f"{name}_evaluated"] = r["evaluated"]
    return metrics | {f"shortcut_auc_{k}": v for k, v in summary["shortcut_auc"].items()}


def build_tables(meta: pd.DataFrame, fractions: dict) -> dict:
    """The four tables to validate, by suite name."""
    return {
        "images": meta,
        "captions": caption_table(meta),
        "splits": split_table(meta, fractions),
        "shortcuts": shortcut_table(meta),
    }


def build_suites(p: dict, n_images: int, img_size: int) -> dict:
    """The four expectation suites, by name, with the thresholds of params.yaml."""
    return {
        "images": images_suite(p, n_images, img_size),
        "captions": captions_suite(),
        "splits": splits_suite(p),
        "shortcuts": shortcuts_suite(p),
    }


@app.command()
def main(
    metadata_path: Path = IMAGE_METADATA_PATH,
    source_path: Path = RAW_DEFACTIFY_DIR / "source.json",
    output_path: Path = METRICS_DIR / "data_validation.json",
    docs_dir: Path = DATA_DOCS_DIR,
    docs: bool = True,
    experiment: str = "vera-data",
):
    """Validate the image metadata; exit with an error if any expectation fails."""
    p = load_params("validate")
    fractions = {s: load_params("split")[s] for s in SPLITS}
    meta = pd.read_csv(metadata_path)
    n_images = json.loads(source_path.read_text())["n_images"]
    tables = build_tables(meta, fractions)
    suites = build_suites(p, n_images, load_params("preprocess")["img_size"])

    results = validate_tables(tables, suites, docs_dir if docs else None)
    shortcut_auc = tables["shortcuts"].set_index("feature")["auc"].round(4).to_dict()
    # For reference: the raw aspect ratio, the shortcut that preprocessing removes
    shortcut_auc["raw_aspect_ratio"] = round(
        float(shortcut_table(meta, ["raw_aspect_ratio"])["auc"].iloc[0]), 4
    )
    summary = {
        "success": all(r["success"] for r in results.values()),
        "suites": results,
        "shortcut_auc": shortcut_auc,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), newline="\n")

    # Logged before failing, so that failed validations are tracked too
    with stage_run("validate_data", experiment, inputs=("data/interim/image_metadata.csv",)):
        mlflow.log_params({f"validate.{k}": v for k, v in p.items()})
        mlflow.log_metrics(validation_metrics(summary))
        mlflow.log_artifact(str(output_path))

    for name, r in results.items():
        log = logger.info if r["success"] else logger.error
        log(f"{name}: {r['successful']}/{r['evaluated']} expectations passed {r['failed'] or ''}")
    if not summary["success"]:
        logger.error(f"Data validation failed, see {output_path}")
        raise typer.Exit(code=1)
    logger.success(f"Data validation passed (summary in {output_path})")
    if docs:
        logger.info(f"Data Docs: {docs_dir / 'index.html'}")


if __name__ == "__main__":
    app()
