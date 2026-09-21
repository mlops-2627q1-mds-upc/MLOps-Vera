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
> **Status:** The model is **not yet trained** — quantitative results are completed in Milestones 2–3. This card states the intended design and how it will be evaluated.

## Model details

- **Developed by:** Team Vera.
- **Model date / version:** 2026; v0.1 (planned).
- **Model type:** binary image classifier (real vs AI-generated) built by **transfer learning** —  Final architecture and hyper-parameters are fixed in Milestone 2.
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

The Defactify **test** split (we may re-split with better distributions), re-split by caption to be leakage-free. Preprocessing: centre-crop/resize to a fixed square and uniform re-encoding, so shape and format cannot leak the label.

## Training data

The Defactify **train** split (same preprocessing). Class imbalance (~5:1 fake:real) is handled via class weights or generator-stratified balanced sampling.

## Quantitative analyses

*To be completed in Milestones 2–3:* overall metrics on the leakage-free test set and per-generator (unitary) results, including the leave-one-generator-out drift experiment.

## Ethical considerations

Dual-use: false positives may wrongly flag genuine images, so outputs are advisory and expect human oversight. The model's reliability is also limited by the dataset's scope and potential biases described above. Training energy / CO₂ is tracked with CodeCarbon.

## Caveats & recommendations

Guard against the aspect-ratio/format shortcut; expect accuracy to drop on unseen or newer generators (monitor drift and retrain as needed); the model is trained on modest-resolution, COCO-domain images and should not be assumed to transfer out-of-domain.
