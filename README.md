# MLOps-Vera

**Detecting real vs. AI-generated images** — built, deployed, and monitored following MLOps and software-engineering best practices.

> Lab project for *Machine Learning Systems in Production (MLOps)* — Master in Data Science, UPC · 2026–2027, Q1.

## Scope

Team **Vera** develops an ML component that classifies an input image as **real** or **AI-generated**, and operates it end-to-end as a production service. The emphasis of the course is the *engineering around the model* — reproducibility, quality assurance, deployment, and monitoring — rather than maximizing raw accuracy.

- **Task:** binary image classification (real vs. AI-generated)
- **Data:** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — real CIFAR-10 images vs. Stable-Diffusion-generated counterparts; additional generators held out to study data drift
- **Serving:** exposed as a REST API (FastAPI) and packaged with Docker
- **Focus:** apply MLOps practices across the six milestones below

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
