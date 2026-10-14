# MLOps-Vera

**Detecting real vs. AI-generated images** — built, deployed, and monitored following MLOps and software-engineering best practices.

> Lab project for *Machine Learning Systems in Production (MLOps)* — Master in Data Science, UPC · 2026–2027, Q1.

## Scope

Team **Vera** develops an ML component that classifies an input image as **real** or **AI-generated**, and operates it end-to-end as a production service. The emphasis of the course is the *engineering around the model* — reproducibility, quality assurance, deployment, and monitoring — rather than maximizing raw accuracy.

- **Task:** binary image classification (real vs. AI-generated)
- **Data:** [Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) — real MS COCO photos vs. images from five recent generators (SD 2.1, SDXL, SD 3, DALL·E 3, MidJourney v6); 96k images, CC BY 4.0
- **Serving:** exposed as a REST API (FastAPI) and packaged with Docker
- **Focus:** apply MLOps practices across six milestones, from inception to monitoring

## Dataset

[Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
([paper](https://arxiv.org/abs/2601.00553)), CC BY 4.0: 16k real MS COCO photos and 80k images
generated from the same captions by five recent generators. We chose it over CIFAKE (one dated
generator at 32×32) and over larger benchmarks such as OpenFake (~3.4 TB) because it uses modern
generators while staying small enough to version and train on a laptop. The pipeline works on a
caption-grouped subsample (800 captions, 7,734 images) pinned to a fixed dataset revision. See the
[dataset card](docs/dataset_card.md) and [`notebooks/data-exploration.ipynb`](notebooks/data-exploration.ipynb).

## Project structure

Based on [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

```
├── data/                  # Versioned with DVC (not Git)
│   ├── raw/defactify/     #   untouched images + metadata (stage `download`)
│   ├── interim/           #   image metadata table for data validation (stage `image_metadata`)
│   └── processed/
│       ├── defactify_224/ #   cropped, resized, re-encoded images (stage `preprocess`)
│       ├── splits/        #   caption-grouped train/val/test CSVs (stage `split`)
│       └── embeddings/    #   frozen-backbone features per split (stage `embed`)
├── docs/                  # Dataset card and model card
├── mlops_vera/            # Python package
│   ├── config.py          #   paths and params.yaml loader
│   ├── dataset.py         #   stage `download`
│   ├── features.py        #   stage `preprocess`
│   ├── split.py           #   stage `split`
│   ├── metadata.py        #   stage `image_metadata`
│   ├── validate.py        #   stage `validate_data` (Great Expectations)
│   └── modeling/          #   stages `embed`, `train`, `logo`, `learning_curve`
├── models/                # Trained classifier (stage `train`, DVC)
├── notebooks/             # Exploratory data analysis
├── reports/               # LaTeX report, milestone write-ups, metrics/ (DVC metrics)
├── tests/                 # Pytest suite: unit tests (offline) and model tests (need dvc pull)
├── dvc.yaml / dvc.lock    # Pipeline definition and its locked state
├── params.yaml            # Pipeline hyper-parameters
├── pyproject.toml / uv.lock
└── Makefile               # make requirements | data | train | validate | test | lint | qa
```

## How to run

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12).

**1. Install and set up access.** Data, models and experiments live on
[DagsHub](https://dagshub.com/AdriSegurao/MLOps-Vera), which serves DVC data only to signed-in
users. Create a free account and an access token (User settings → Tokens), then:

```bash
make requirements                                    # uv sync
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-username>
uv run dvc remote modify origin --local password <dagshub-token>
cp .env.example .env                                 # MLflow: same username and token
```

Both files are git-ignored. Without `.env`, MLflow runs are logged locally to `./mlflow.db`
(`uv run mlflow ui`). If `dvc pull` reports missing files, the DVC token is the usual cause.

**2. Get the data and the model**, either by downloading what the pipeline produced or by
rebuilding it from Hugging Face:

```bash
uv run dvc pull     # data, embeddings and trained model (~760 MB)
uv run dvc repro    # or re-run the stages whose code, params or inputs changed
uv run dvc status   # "Data and pipelines are up to date" = workspace matches dvc.lock
```

Single parts of the pipeline: `make data` (download → split), `make validate` (Great
Expectations), `make train` (embed → train), `make logo` (leave-one-generator-out). Results:
`uv run dvc metrics show` and `uv run dvc plots show`.

**3. Run experiments.** Hyper-parameters live in `params.yaml`; override them per experiment:

```bash
uv run dvc exp run -n clip-C0.1 -S train.C=0.1
uv run dvc exp show -A
```

Every stage logs a run to [MLflow](https://dagshub.com/AdriSegurao/MLOps-Vera.mlflow), including
its CodeCarbon energy and emissions.

**4. Quality checks.**

```bash
make test     # pytest with coverage (model tests need `dvc pull`; skip them with -m "not model")
make lint     # ruff + pylint
make qa       # lint, data validation, tests and notebook lint
```

## Licence

The code and the trained model are released under the [MIT licence](LICENSE). The data is not part of the repository: the
Defactify dataset is licensed under CC BY 4.0 and its real images follow MS COCO's terms (see the
[dataset card](docs/dataset_card.md)).
