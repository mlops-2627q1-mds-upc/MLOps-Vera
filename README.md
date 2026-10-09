# MLOps-Vera

**Detecting real vs. AI-generated images** — built, deployed, and monitored following MLOps and software-engineering best practices.

> Lab project for *Machine Learning Systems in Production (MLOps)* — Master in Data Science, UPC · 2026–2027, Q1.

## Scope

Team **Vera** develops an ML component that classifies an input image as **real** or **AI-generated**, and operates it end-to-end as a production service. The emphasis of the course is the *engineering around the model* — reproducibility, quality assurance, deployment, and monitoring — rather than maximizing raw accuracy.

- **Task:** binary image classification (real vs. AI-generated)
- **Data:** [Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) — real MS COCO photos vs. images from five recent generators (SD 2.1, SDXL, SD 3, DALL·E 3, MidJourney v6); 96k images, CC BY 4.0
- **Serving:** exposed as a REST API (FastAPI) and packaged with Docker
- **Focus:** apply MLOps practices across the six milestones below

## Dataset

We use **[Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)** ([paper](https://arxiv.org/abs/2601.00553)): 96,000 images — 16k real **MS COCO** photos and 80k from **five recent generators** (Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3, MidJourney v6) — released under **CC BY 4.0**. Initial exploration lives in [`notebooks/data-exploration.ipynb`](notebooks/data-exploration.ipynb).

### Why Defactify (dataset decision)

We initially scoped the project around **CIFAKE**, then switched to Defactify after evaluating the alternatives:

- **CIFAKE → Defactify.** CIFAKE provides only **one, now-dated generator** (Stable Diffusion 1.4) at 32×32 resolution. Defactify covers **five recent generators (2022–2024)** at realistic resolutions, making the real-vs-AI task far more representative of today's generated images and enabling a cross-generator drift study.
- **Defactify over larger datasets (e.g., OpenFake).** Larger, newer benchmarks such as **OpenFake** (~3.44 TB, latest generators) were rejected as the core dataset because of their **size and heavier computational needs** — they do not fit free data-versioning or laptop-class training, and add heavier licensing (CC-BY-NC-SA) and ethics constraints. Defactify's **96k images / ~7.5 GB** stays reproducible and trainable on our hardware while still using modern generators.

> The course is graded on MLOps engineering rather than raw accuracy, so we prioritised a clean, well-licensed, right-sized dataset that we can reproducibly build, deploy, and monitor.

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
│   └── modeling/          #   stages `embed`, `train`; inference
├── models/                # Trained classifier (stage `train`, DVC)
├── notebooks/             # Exploratory data analysis
├── reports/               # LaTeX report, milestone write-ups, metrics/ (DVC metrics)
├── tests/                 # Pytest suite: unit tests (offline) and model tests (need dvc pull)
├── dvc.yaml / dvc.lock    # Pipeline definition and its locked state
├── params.yaml            # Pipeline hyper-parameters
├── pyproject.toml / uv.lock
└── Makefile               # make requirements | data | train | validate | test | lint
```

## Getting the data (DVC)

The data, the intermediate outputs and the trained model are versioned with **DVC**. Git stores the
pipeline (`dvc.yaml`), its parameters (`params.yaml`) and the fingerprint of every output
(`dvc.lock`); the files themselves are in the DVC remote on DagsHub
(`https://dagshub.com/AdriSegurao/MLOps-Vera.dvc`, set in `.dvc/config`).

| Stage | What it does | Output |
| --- | --- | --- |
| `download` | Caption-grouped subsample of Defactify (800 captions → 7,734 images), pinned to a fixed dataset revision | `data/raw/defactify` (~610 MB) |
| `preprocess` | Centre-crop, resize to 224 px and re-encode as JPEG | `data/processed/defactify_224` (~130 MB) |
| `split` | 70/15/15 train/val/test split grouped by caption | `data/processed/splits` |
| `embed` | Frozen CLIP ViT-B/32 embedding of every image | `data/processed/embeddings` (~17 MB) |
| `train` | Logistic-regression head on the embeddings | `models/classifier.joblib` |
| `logo@<generator>`, `logo_summary` | Leave-one-generator-out evaluation | metrics only |

### One-time access setup

The DagsHub repository is public, but DagsHub only serves DVC data to signed-in users. Create a
free DagsHub account (signing in with GitHub works) and an access token (DagsHub → User settings →
Tokens), then store them locally; both files are git-ignored:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-username>
uv run dvc remote modify origin --local password <dagshub-token>
cp .env.example .env   # MLflow: fill in the same username and token
```

Use the same token in `.dvc/config.local` and `.env`. If `dvc pull` reports missing files while
MLflow works, the DVC token is the usual cause: DVC reports a rejected login as missing files.

### Pulling and reproducing

```bash
uv run dvc pull                            # everything (~760 MB): data, embeddings and model
uv run dvc pull data/processed/embeddings data/processed/splits  # enough to re-train or evaluate the head (~18 MB)
uv run dvc status                          # "Data and pipelines are up to date" = matches dvc.lock
uv run dvc repro                           # re-runs only the stages whose code, params or inputs changed
```

`dvc pull` downloads the exact files the pipeline produced, so nothing is re-trained. `dvc repro`
rebuilds outputs from Hugging Face and the code; after changing something, run `uv run dvc push`
and commit the updated `dvc.lock`. The amount of data is the `data.n_captions` parameter: change it
and run `dvc repro` to build a larger version, while DVC keeps both versions.

### Is the subsample large enough?

The `learning_curve` stage re-trains the selected head on growing fractions of the training
captions (5 random draws per fraction) and evaluates each fit on the validation split:

| Training captions | Images (real) | Balanced acc. | ROC-AUC | PR-AUC (real) |
| --- | --- | --- | --- | --- |
| 10% | 515 (86) | 0.905 | 0.964 | 0.876 |
| 25% | 1,250 (208) | 0.925 | 0.977 | 0.914 |
| 50% | 2,712 (452) | 0.929 | 0.978 | 0.919 |
| 75% | 4,060 (677) | 0.931 | 0.981 | 0.930 |
| 100% | 5,352 (892) | 0.935 | 0.982 | 0.933 |

The curve flattens early: four times more data (25% to 100%) adds about one point of balanced
accuracy, less than the noise of the validation set (±2 points). More captions would mainly give
larger test sets and narrower confidence intervals, not a better model.

```bash
uv run dvc repro learning_curve   # re-run it (seconds, reuses the embeddings)
uv run dvc plots show             # line chart of the curve (dvc_plots/index.html)
```

## Experiment tracking (MLflow)

Every stage of the DVC pipeline logs a run with **MLflow** to the tracking server hosted by
DagsHub: <https://dagshub.com/AdriSegurao/MLOps-Vera.mlflow>.

```bash
cp .env.example .env   # then fill in your DagsHub username and access token
```

`.env` is git-ignored and loaded automatically by `mlops_vera/config.py`. Without it, runs are
logged locally to `./mlflow.db` (browse them with `uv run mlflow ui`).

| Experiment | Stages | What each run records |
| --- | --- | --- |
| `vera-data` | `download`, `preprocess`, `split`, `embed` | sample size and class balance, resolved dataset sha, an example caption group after preprocessing, per-split counts, embedding dimension and throughput |
| `vera-baselines` | `train` | see [Running experiments](#running-experiments) |
| `vera-logo` | `logo@<generator>`, `logo_summary` | val and unseen-generator metrics with bootstrap CI; mean and worst case against MR-4, with a chart |
| `vera-learning-curve` | `learning_curve` | validation metrics per training-caption percentage (one MLflow step each), with a chart |

Each run logs its section of `params.yaml` as params, and all share the same lineage tags:
`dvc.stage` (the stage address, e.g. `logo@sd3`), `data_revision`, the DVC hash of every input
(`<input>_md5`, read from `dvc.lock`), `dvc.exp_name` when run by `dvc exp run -n <name>`, and
the Git commit (added by MLflow). The helpers live in `mlops_vera/tracking.py`.

### Running experiments

The `embed` stage extracts features with a frozen pretrained backbone; `train` fits a
logistic-regression head on them and logs the run to MLflow (experiment `vera-baselines`). An
experiment is a variation of `params.yaml`, run with DVC so that code, params, data and metrics
stay linked:

```bash
# Backbones: any torchvision classifier (resnet18, resnet50, ...) or clip_vit_b_32
uv run dvc exp run -n resnet50-balanced -S embed.backbone=resnet50 -S train.class_weight=balanced
uv run dvc exp run -n clip-balanced-C0.1 -S train.C=0.1   # head regularisation

uv run dvc exp show -A                 # compare experiments (params + metrics) in the terminal
uv run dvc exp apply resnet50-balanced # restore one into the workspace
uv run dvc exp push origin <name>      # share it; others get it with `dvc exp pull origin`
```

`embed` only re-runs when the backbone changes, so trying other head settings is cheap.

Each MLflow run, named `<backbone>-logreg-<balanced|unweighted>-C<C>`, records:

- **Params:** the `embed` and `train` sections of `params.yaml`.
- **Metrics** (train and val): balanced accuracy, macro-F1, ROC-AUC, PR-AUC and recall of the
  real class, recall of the AI class and of each generator, and the decision threshold (tuned
  on validation).
- **Tags** linking it to its exact inputs: dataset revision, DVC hash of the embeddings and
  backbone weights (MLflow adds the Git commit).
- **Artifacts:** validation confusion matrix and PR curve, and the fitted model, whose
  `predict()` applies the tuned threshold.

The test split is not used by `train`; it is kept for the final evaluation.

### Baseline results

Six baselines (3 backbones × with/without class weighting, C = 1); validation split:

| Backbone | `class_weight` | Balanced acc. | Macro-F1 | PR-AUC (real) | Recall (real) |
| --- | --- | --- | --- | --- | --- |
| **CLIP ViT-B/32** | **none** | **0.916** | **0.845** | **0.898** | 0.943 |
| CLIP ViT-B/32 | balanced | 0.911 | 0.829 | 0.896 | 0.952 |
| ResNet-50 | none | 0.821 | 0.717 | 0.582 | 0.871 |
| ResNet-50 | balanced | 0.816 | 0.712 | 0.563 | 0.866 |
| ResNet-18 | none | 0.811 | 0.686 | 0.660 | 0.904 |
| ResNet-18 | balanced | 0.803 | 0.690 | 0.653 | 0.871 |

Regularisation sweep of the selected head (CLIP ViT-B/32, balanced class weights); C = 1
overfits the embeddings (train ROC-AUC 1.000):

| C | Balanced acc. | Macro-F1 | PR-AUC (real) | Recall (real) | ROC-AUC train / val | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.911 | 0.829 | 0.896 | 0.952 | 1.000 / 0.970 | 0.994 |
| 0.3 | 0.916 | 0.850 | 0.900 | 0.938 | 1.000 / 0.973 | 0.919 |
| 0.1 | 0.919 | 0.871 | 0.910 | 0.919 | 0.999 / 0.976 | 0.707 |
| 0.03 | 0.931 | 0.877 | 0.924 | 0.943 | 0.999 / 0.980 | 0.688 |
| **0.01** | **0.935** | **0.884** | **0.933** | 0.947 | 0.998 / 0.982 | 0.570 |
| 0.003 | 0.937 | 0.887 | 0.937 | 0.947 | 0.995 / 0.984 | 0.509 |
| 0.001 | 0.935 | 0.900 | 0.937 | 0.928 | 0.992 / 0.984 | 0.427 |

Since C = 0.01 was the smallest value of the first sweep, the sweep was extended to 0.003 and
0.001. Below 0.01 the metrics level off: balanced accuracy changes by at most 0.002, far below the
noise of the validation split (±2 points with 209 real images), so C = 0.01 is kept rather than
picking a winner by noise. Every value of the sweep is stored as a DVC experiment
(`clip-vit-b-32-balanced-C<C>`; `uv run dvc exp pull origin <name>`, then `uv run dvc exp show -A`).

CLIP ViT-B/32 with balanced class weights and C = 0.01 is the selected model (current
`params.yaml`): class weighting keeps the head robust to other class ratios in future training
data, and the tuned threshold (0.57) is saved inside the model, so both the joblib bundle and
the MLflow model classify with it. See the [model card](docs/model_card.md) and the report for
the analysis.

### Cross-generator generalisation (leave-one-generator-out)

Requirement MR-4 asks how the detector copes with a generator it has never seen, the drift we
expect in production. The `logo` stage is a DVC `foreach` with one branch per generator listed in
`params.yaml` (`logo.holdout`): `logo@sd3` trains the selected head without any SD 3 image in
train or val (threshold tuning included), then tests it on the real images of the test split plus
the SD 3 images only. All branches reuse the embeddings, so each one takes seconds, and DVC only
re-runs the branches whose inputs changed. `logo_summary` joins them into one table and checks the
target (balanced accuracy ≥ 0.70) on the worst generator:

```bash
uv run dvc repro logo_summary   # or a single branch: uv run dvc repro logo@sd3
uv run dvc metrics show         # per-generator metrics and the summary
uv run dvc plots show           # bar chart of the unseen balanced accuracy (dvc_plots/index.html)
```

Each branch is also an MLflow run (`<backbone>-logo-<generator>`, experiment `vera-logo`).

| Held-out generator | Balanced acc. (95% CI) | Recall of the unseen generator | Recall (real) |
| --- | --- | --- | --- |
| SD 2.1 | 0.872 (0.827–0.914) | 0.814 | 0.931 |
| SDXL | 0.957 (0.936–0.977) | 0.984 | 0.931 |
| **SD 3** | **0.822 (0.765–0.872)** | **0.670** | 0.973 |
| DALL·E 3 | 0.923 (0.884–0.956) | 0.920 | 0.926 |
| MidJourney | 0.915 (0.880–0.948) | 0.904 | 0.926 |

Mean 0.898, worst 0.822 (SD 3): MR-4 is met. Each test set has 188 real and 188 generated images
from 120 captions; the 95% bootstrap CI resamples whole captions, since images of the same caption
show the same scene and their errors are correlated.

## Quality assurance

### Data validation (Great Expectations)

Great Expectations validates tables, and our data are images, so the `image_metadata` stage first
turns every image into one row of metadata: whether the raw and the preprocessed file open, their
format, colour mode, size and file size, plus the brightness, contrast and md5 hash of the
preprocessed image, its labels and its split. The `validate_data` stage then checks four tables
built from it, each with its own expectation suite (32 expectations, thresholds in the `validate`
section of `params.yaml`):

| Suite | One row per | Checks |
| --- | --- | --- |
| `images` | image | unique ids; valid and consistent labels (real ↔ no generator); every file opens; raw sizes plausible; preprocessed images are 224×224 RGB JPEGs, not black, white or flat; no duplicate files; ~5 AI images per real one |
| `captions` | caption | every caption in exactly one split (no leakage, DR-4) and with all six sources |
| `splits` | split | real share, all generators present, size matching the split fractions |
| `shortcuts` | image feature | no single preprocessed-image feature predicts the label: ROC-AUC ≤ 0.60 (DR-5) |

```bash
uv run dvc repro validate_data   # fails, stopping the pipeline, if any expectation fails
```

It writes a summary (`reports/metrics/data_validation.json`, a DVC metric) and the HTML Data Docs
(`reports/data_docs/index.html`, not versioned). The `shortcuts` suite verifies the purpose of
preprocessing: on the raw images the aspect ratio alone separates real from AI images with
ROC-AUC 0.75 (all AI images are square, only 2% of the real ones), while after preprocessing no
feature exceeds 0.55 (file size 0.52, brightness 0.55, contrast 0.53).

### Tests (Pytest)

```bash
uv run pytest --cov=mlops_vera   # all tests with coverage (87% of the package)
uv run pytest -m "not model"     # only the fast, offline tests
```

- **Unit tests** (offline, synthetic data) for every stage, including the expectation suites:
  valid metadata passes, and each kind of broken data (non-square or black image, label
  mismatch, caption leakage, duplicate file, unreadable file, missing generator, size shortcut)
  fails the suite meant to catch it.
- **Model tests** (`-m model`, on the real artefacts; skipped if they are not pulled): the saved
  model meets the model-card targets on validation and detects at least 80% of every generator;
  it classifies at the tuned threshold and retraining reproduces it; the serving path (raw image
  → `preprocess_image` → backbone) reproduces the training embeddings (no training/serving skew,
  FR-5); and at least 90% of its decisions survive mirroring, JPEG re-compression (q75, q50), a
  10% brightness change or a half-resolution upload (measured: 94–98%).

## Team

| Member | GitHub |
| --- | --- |
| Arman Bazarchi | [@armanbzi](https://github.com/armanbzi) |
| Adrián Segura | [@AdriSegurao](https://github.com/AdriSegurao) |
| Pablo Rodríguez | [@PabloRodriguezElvira](https://github.com/PabloRodriguezElvira) |

## Milestones

1. **Inception** — ML problem & requirements, dataset & model cards, project coordination
2. **Model building — reproducibility** — project structure (Cookiecutter DS), code & data versioning (Git + DVC), experiment tracking (MLflow)
3. **Model building — quality assurance** — energy tracking (CodeCarbon), static analysis (Pylint / Pynblint), testing (Pytest, Great Expectations)
4. **Model deployment — API** — ML system design, REST API (FastAPI), API testing
5. **Model deployment — packaging** — containerization (Docker), CI/CD (GitHub Actions)
6. **Monitoring** — resource monitoring (Prometheus + Grafana), data & model drift (Alibi Detect)
