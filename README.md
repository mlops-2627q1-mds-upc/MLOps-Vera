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
│   └── modeling/          #   stages `embed`, `train`; inference
├── models/                # Trained classifier (stage `train`, DVC)
├── notebooks/             # Exploratory data analysis
├── reports/               # LaTeX report, milestone write-ups, metrics/ (DVC metrics)
├── tests/                 # Pytest suite (offline)
├── dvc.yaml / dvc.lock    # Pipeline definition and its locked state
├── params.yaml            # Pipeline hyper-parameters
├── pyproject.toml / uv.lock
└── Makefile               # make requirements | data | train | test | lint
```

## Experiment tracking (MLflow)

Training runs are logged with **MLflow** to the tracking server hosted by DagsHub:
<https://dagshub.com/AdriSegurao/MLOps-Vera.mlflow>.

```bash
cp .env.example .env   # then fill in your DagsHub username and access token
```

`.env` is git-ignored and loaded automatically by `mlops_vera/config.py`. Without it, runs are
logged locally to `./mlflow.db` (browse them with `uv run mlflow ui`).

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

CLIP ViT-B/32 with balanced class weights and C = 0.01 is the selected model (current
`params.yaml`): class weighting keeps the head robust to other class ratios in future training
data, and the tuned threshold (0.57) is saved inside the model, so both the joblib bundle and
the MLflow model classify with it. See the [model card](docs/model_card.md) and the report for
the analysis.

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
