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

## Team

| Member | GitHub |
| --- | --- |
| Arman Bazarchi | [@armanbzi](https://github.com/armanbzi) |
| Adrián Segura | — |
| Pablo Rodríguez | — |
| Mario Prisco | — |

## Milestones

1. **Inception** — ML problem & requirements, dataset & model cards, project coordination
2. **Model building — reproducibility** — project structure (Cookiecutter DS), code & data versioning (Git + DVC), experiment tracking (MLflow)
3. **Model building — quality assurance** — energy tracking (CodeCarbon), static analysis (Pylint / Pynblint), testing (Pytest, Great Expectations)
4. **Model deployment — API** — ML system design, REST API (FastAPI), API testing
5. **Model deployment — packaging** — containerization (Docker), CI/CD (GitHub Actions)
6. **Monitoring** — resource monitoring (Prometheus + Grafana), data & model drift (Alibi Detect)
