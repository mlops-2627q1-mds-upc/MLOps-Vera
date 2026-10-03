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
> **Status:** Milestone 2 baseline selected (v0.1). Results below are on the **validation** split, plus the leave-one-generator-out study; the held-out test evaluation is completed in Milestone 3.

## Model details

- **Developed by:** Team Vera.
- **Model date / version:** September 2026; v0.1 (Milestone 2 baseline).
- **Model type:** binary image classifier (real vs AI-generated) built by **transfer learning**: the image encoder of **CLIP ViT-B/32** (OpenAI weights, via open_clip), used **frozen**, maps each 224×224 image to a 512-d embedding; a **standardisation + logistic-regression** head (L2, C = 0.01, balanced class weights) outputs P(AI). The image is flagged as AI when P(AI) ≥ 0.57, a threshold tuned on validation to maximise balanced accuracy and stored inside the saved model. Backbone chosen among six tracked baselines (ResNet-18, ResNet-50 and CLIP ViT-B/32, each with and without class weighting), then C chosen with a sweep over 0.01–1; hyper-parameters live in `params.yaml` and every run is logged to MLflow (DagsHub).
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

A caption-grouped subsample of Defactify (800 captions with all six sources; 5,352 training images from 560 captions: 892 real, 4,460 AI), same preprocessing. The ~5:1 fake:real imbalance is handled by balanced class weights, so the head does not depend on the class ratio of the training data, plus the decision threshold tuned on validation.

## Quantitative analyses

Validation results of the selected model (test-set results follow in Milestone 3):

| Metric | Value | Target |
| --- | --- | --- |
| Balanced accuracy | 0.935 | ≥ 0.85 ✅ |
| Macro-F1 | 0.884 | ≥ 0.85 ✅ |
| Recall (real) | 0.947 | ≥ 0.80 ✅ |
| PR-AUC (real) | 0.933 | ≥ 0.90 ✅ |
| ROC-AUC | 0.982 | — |

Recall per generator (share of its images flagged as AI): SD 2.1 0.876, SDXL 0.971, **SD 3 0.847**, DALL·E 3 0.981, MidJourney 0.943. SD 3 is the hardest generator for every backbone tried.

Compared with the ResNet baselines (validation balanced accuracy 0.80–0.82), the CLIP encoder is about 10 points better and overfits less (train/val balanced accuracy 0.96/0.92 vs 0.96/0.82 for ResNet-50). With C = 1 the CLIP head still overfits (train ROC-AUC 1.00 vs 0.97 on validation); stronger regularisation (C = 0.01) narrows the gap (0.998 vs 0.982) and adds about 2 points of balanced accuracy and 4 of PR-AUC.

**Cross-generator generalisation (leave-one-generator-out, MR-4).** For each generator, the same head is trained without it (train and val, threshold tuning included) and tested on the test split's real images plus that generator's images only (188 + 188 images, 95% bootstrap CI):

| Held-out generator | Balanced accuracy | Recall of the unseen generator | Recall (real) |
| --- | --- | --- | --- |
| SD 2.1 | 0.872 (0.838–0.907) | 0.814 | 0.931 |
| SDXL | 0.957 (0.936–0.976) | 0.984 | 0.931 |
| **SD 3** | **0.822 (0.787–0.856)** | **0.670** | 0.973 |
| DALL·E 3 | 0.923 (0.894–0.949) | 0.920 | 0.926 |
| MidJourney | 0.915 (0.883–0.942) | 0.904 | 0.926 |

The worst case (SD 3, 0.822) meets the cross-generator target (≥ 0.70); the mean is 0.898. The drop is concentrated in the generated class: real images stay well recognised, but an unseen SD 3 image is detected only 67% of the time, against 85% on validation when SD 3 is part of training. SD 2.1 also drops, while SDXL, DALL·E 3 and MidJourney stay close to the in-distribution results, so their artefacts seem to be shared with the other generators.

## Ethical considerations

Dual-use: false positives may wrongly flag genuine images, so outputs are advisory and expect human oversight. The model's reliability is also limited by the dataset's scope and potential biases described above. Training energy / CO₂ is tracked with CodeCarbon.

## Caveats & recommendations

Guard against the aspect-ratio/format shortcut; expect accuracy to drop on unseen or newer generators, as the leave-one-generator-out study shows for SD 3 (monitor drift and retrain as needed); the model is trained on modest-resolution, COCO-domain images and should not be assumed to transfer out-of-domain.
