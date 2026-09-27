---
pretty_name: "Vera real-vs-AI image detector"
license: cc-by-4.0
pipeline_tag: image-classification
language:
  - en
tags:
  - ai-generated-image-detection
  - real-vs-fake
  - transfer-learning
datasets:
  - Rajarshi-Roy-research/Defactify_Image_Dataset
metrics:
  - balanced_accuracy
  - f1
  - pr_auc
  - recall
---

# Model Card — Vera real-vs-AI image detector

> Team **Vera** — *Machine Learning Systems in Production (MLOps)*, UPC 2026–2027.
> Structure follows Mitchell et al. (2019), *Model Cards for Model Reporting*.
>
> **Status:** Milestone 2 baseline selected (v0.1). Results below are on the **validation** split; the held-out test evaluation and the leave-one-generator-out study are completed in Milestone 3.

## Model details

- **Developed by:** Team Vera.
- **Model date / version:** September 2026; v0.1 (Milestone 2 baseline).
- **Model type:** binary image classifier (real vs AI-generated) built by **transfer learning**: the image encoder of **CLIP ViT-B/32** (OpenAI weights, via open_clip), used **frozen**, maps each 224×224 image to a 512-d embedding; a **standardisation + logistic-regression** head (L2, C = 1.0, no class weights) outputs P(AI). The image is flagged as AI when P(AI) ≥ 0.992, a threshold tuned on validation to maximise balanced accuracy. Chosen among six tracked baselines (ResNet-18, ResNet-50 and CLIP ViT-B/32, each with and without class weighting); hyper-parameters live in `params.yaml` and every run is logged to MLflow (DagsHub).
- **Training data:** [Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (easily configurable to other datasets) (see the [dataset card](dataset_card.md)).
- **Licence:** code under the repository licence; training data under CC BY 4.0.
- **Questions / comments:** Team Vera (see report cover).

## Intended use

- **Primary uses:** decide whether an input image is real or AI-generated, returning a confidence score; optionally attribute the generator.
- **Primary users:** content-platform back-ends, journalists and fact-checkers, and developers integrating the component via its API.
- **Out-of-scope:** deepfake face-swap or partial-manipulation forensics, adversarially-perturbed inputs, generators far outside the training set, and any high-stakes decision without human review.

## Factors

Relevant factors: generator family (SD 2.1, SDXL, SD 3, DALL·E 3, MidJourney v6), image resolution / aspect ratio, and content domain (COCO-like scenes). Evaluation is reported **per generator** (leave-one-generator-out) as well as overall.

## Metrics

Balanced accuracy, macro-F1, PR-AUC, and recall on the minority (real) class; the decision threshold is tuned on validation rather than fixed at 0.5. Performance variation is reported across generators. Targets (to confirm): balanced accuracy ≥ 0.85, macro-F1 ≥ 0.85, recall(real) ≥ 0.80, PR-AUC ≥ 0.90, cross-generator balanced accuracy ≥ 0.70.

## Evaluation data

Caption-grouped validation split (1,254 images from 120 captions: 209 real, 1,045 AI) for model selection and threshold tuning, and a held-out test split (1,128 images, 120 captions) for the final evaluation. No caption is shared between splits. Preprocessing: centre-crop/resize to a fixed square and uniform re-encoding, so shape and format cannot leak the label.

## Training data

A caption-grouped subsample of Defactify (800 captions with all six sources; 5,352 training images from 560 captions: 892 real, 4,460 AI), same preprocessing. The ~5:1 fake:real imbalance is handled by tuning the decision threshold on validation; class weighting was also tried and made no significant difference (see below).

## Quantitative analyses

Validation results of the selected model (test-set results follow in Milestone 3):

| Metric | Value | Target |
| --- | --- | --- |
| Balanced accuracy | 0.916 | ≥ 0.85 ✅ |
| Macro-F1 | 0.845 | ≥ 0.85 (just below) |
| Recall (real) | 0.943 | ≥ 0.80 ✅ |
| PR-AUC (real) | 0.898 | ≥ 0.90 (just below) |
| ROC-AUC | 0.971 | — |

Recall per generator (share of its images flagged as AI): SD 2.1 0.856, SDXL 0.952, **SD 3 0.789**, DALL·E 3 0.943, MidJourney 0.904. SD 3 is the hardest generator for every backbone tried.

Compared with the ResNet baselines (validation balanced accuracy 0.80–0.82), the CLIP encoder is about 10 points better and overfits less (train/val balanced accuracy 0.96/0.92 vs 0.96/0.82 for ResNet-50). The leave-one-generator-out drift experiment (cross-generator target ≥ 0.70) is pending.

## Ethical considerations

Dual-use: false positives may wrongly flag genuine images, so outputs are advisory and expect human oversight. The model's reliability is also limited by the dataset's scope and potential biases described above. Training energy / CO₂ is tracked with CodeCarbon.

## Caveats & recommendations

Guard against the aspect-ratio/format shortcut; expect accuracy to drop on unseen or newer generators (monitor drift and retrain as needed); the model is trained on modest-resolution, COCO-domain images and should not be assumed to transfer out-of-domain.
