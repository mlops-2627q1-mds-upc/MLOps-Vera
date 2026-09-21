---
pretty_name: "Defactify / MS-COCOAI (real vs AI-generated)"
license: cc-by-4.0
language:
  - en
task_categories:
  - image-classification
size_categories:
  - 10K<n<100K
tags:
  - ai-generated-image-detection
  - real-vs-fake
  - synthetic-image
  - deepfake-detection
source_datasets:
  - Rajarshi-Roy-research/Defactify_Image_Dataset
---

# Dataset Card — Defactify / MS-COCOAI (real vs AI-generated)

> Team **Vera** — *Machine Learning Systems in Production (MLOps)*, UPC 2026–2027.
> **Source:** [`Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) ·
> **Paper:** [arXiv 2601.00553](https://arxiv.org/abs/2601.00553) · **Licence:** CC BY 4.0.

## Dataset summary

Defactify (MS-COCOAI) pairs real photographs from **MS COCO** with AI-generated images produced from the **same captions** by five recent text-to-image generators. It supports detecting whether an image is real or AI-generated and, optionally, attributing which generator produced it. Because each real image has generated counterparts from the *same* prompt, content is controlled, so as a detector must learn *how* (patterns) an image was made, not *what* it shows.

## Supported tasks

- **Task A — binary detection:** `real` vs `AI-generated` (main task).
- **Task B — generator attribution:** which of the five generators produced a fake (optional task to include).

## Languages & modality

Images, paired with English captions (from MS COCO).

## Dataset structure

- **Size:** 96,000 image–caption records.
- **Composition:** 16,000 real (MS COCO) + 80,000 AI-generated (16,000 each from Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3, MidJourney v6).
- **Fields:** `Caption` (string), `Image` (image), `Label_A` (int: 0 = real, 1 = AI-generated), `Label_B` (int: 0 = real, 1–5 = generator).
- **Splits:** train 42,000 / validation 9,000 / test 45,000.
- **Format / resolution:** JPEG, variable resolution (~270×270 up to 1024×1024).

## Dataset creation

- **Curation:** same-caption real/fake pairing to control content bias.
- **Source data:** real images from MS COCO; fake images generated from MS COCO captions by the five generators.

## Considerations for using the data

Findings from our exploratory analysis ([`notebooks/data-exploration.ipynb`](../notebooks/data-exploration.ipynb)):

- **Class imbalance:** the binary task is ~5:1 fake:real (Task B is balanced).
- **Aspect-ratio shortcut:** generated images are almost all square while real photos are not (~100% vs ~4% square in our sample) — normalise size so the model cannot cheat on shape.
- **Split leakage:** some captions appear across train/val/test; apply a **caption-grouped re-split** before evaluation.
- **Scope:** everyday COCO scenes and five specific 2022–2024 generators; not representative of deepfake face-swaps, adversarial inputs, or newer generators.
- **Privacy / representation:** MS COCO photos may contain people and reflect MS COCO's own collection biases.

## Recommended preprocessing 

Centre-crop/resize every image to a fixed square and re-encode all images to one format/quality, so neither aspect ratio nor compression leaks the label; version data and splits with DVC.

## Licensing

**CC BY 4.0** (author-declared in the paper); the real images additionally follow **MS COCO**'s terms. Attribute both the Defactify paper and MS COCO.

## Citation

- Defactify / MS-COCOAI — *A Comprehensive Dataset for Human vs. AI Generated Image Detection*, [arXiv 2601.00553](https://arxiv.org/abs/2601.00553).
- MS COCO — Lin et al., *Microsoft COCO: Common Objects in Context*, 2014.
