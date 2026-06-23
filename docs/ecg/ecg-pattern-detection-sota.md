# ECG Pattern Detection: SOTA Survey & Implementation Plan

> Scope agreed with requester (2026-06-23):
> **deliverable** = research + implementation plan;
> **input modality** = any/multiple (raw 12-lead, single-lead/wearable, ECG images);
> **targets** = rhythm/arrhythmia, structural ("hidden") disease, ischemia/MI, risk/prognosis;
> **stack** = TensorFlow / Keras.

This document is a wide sweep of the current state of the art for reading ECGs to detect
abnormal patterns, followed by a concrete, phased plan to build a TensorFlow/Keras system.

It was prompted by an NYT piece (2026-06-22) on AI finding "hidden" heart damage on ordinary
ECGs. The NYT page itself is paywalled/blocked from this environment, but the underlying line of
research is well documented: the marquee example is **EchoNext** (Columbia / NewYork-Presbyterian,
*Nature* 2025), an AI tool that reads a standard ECG to flag patients who likely have
**structural heart disease** and should get an echocardiogram. In a head-to-head on ~3,200 ECGs it
identified 77% of structural problems vs 64% for cardiologists. This is the same family of work as
Mayo Clinic's AI-ECG for low ejection fraction (the EAGLE trial) and AI-ECG cardiac amyloidosis
screening. The "AI sees damage humans can't" framing is exactly the structural/prognostic branch
of the taxonomy below.

---

## 1. How ECGs are read to detect abnormal patterns

There are three orthogonal axes to think about: **input modality**, **model architecture**, and
**clinical target**. SOTA systems mix and match across all three.

### 1.1 Input modalities

| Modality | What it is | Typical use | Notes |
| --- | --- | --- | --- |
| Raw 12-lead signal | 10 s waveform, 250–500 Hz, 12 channels | Clinical-grade diagnosis | Richest signal; PTB-XL / MIMIC-IV-ECG / Chapman-Shaoxing |
| Single / reduced lead | 1–3 leads from watch, patch, Holter | Screening, ambulatory monitoring | Apple Watch, KardiaMobile, Zio patch; harder, noisier |
| ECG image / paper scan | Photo or PDF of a printed strip | Low-resource, legacy archives | Needs digitization or direct vision models |

A 2024 trend is **digitization pipelines** (PhysioNet/CinC 2024 challenge) that convert paper/image
ECGs back to signals, plus **direct image models** that skip digitization and classify the picture.

### 1.2 Model architectures (signal-based)

The dominant pattern is treating the ECG as a multivariate time series.

- **CNN / 1-D ResNet** — still the workhorse. Deep residual 1-D CNNs (the Ribeiro et al. / Mayo
  lineage) underpin most clinically validated tools (low-EF detection AUC ~0.93). Strong, fast,
  well-understood; the default baseline.
- **CNN + RNN (LSTM/GRU) hybrids** — CNN front-end for local morphology, recurrent back-end for
  rhythm context. Common in MI detection and multi-centre arrhythmia work.
- **Transformers / attention** — self-attention captures long-range and inter-lead dependencies.
  Hybrid CNN+Transformer models (e.g. DeepECG-Net) report top results on MIT-BIH / PTB-XL /
  CPSC-2018. Attention also aids interpretability.
- **State-space models (S4 / S4D / Mamba)** — purpose-built for long sequences. ECGs at
  100–1000 Hz over 10 s are long; S4/S4D ("S4D-ECG", "S4ECG") match or beat Transformers,
  especially **out-of-distribution**, at lower compute. A leading 2025 direction.
- **Graph / multi-lead structure models** — model the 12 leads as a graph to exploit spatial lead
  relationships. Niche but growing.

### 1.3 Model architectures (image-based)

- **2-D CNNs** (ResNet/EfficientNet/DenseNet) on rendered or photographed strips.
- **Vision Transformers (ViT)** on ECG imagery — 2025 work on paper-ECG MI detection.
- **Ensembles for screening** — e.g. PRESENT-SHD couples per-image CNNs with an XGBoost meta-model
  for a composite structural-heart-disease screen.
- **Synthetic-data pipelines** — render clean signals to realistic "paper" (wrinkles, handwriting,
  perspective, noise) to train robust image models without large labeled image corpora.

### 1.4 Self-supervised / foundation models (the 2024–2025 frontier)

Labeled ECGs are scarce relative to the millions of unlabeled recordings, so SSL pretraining is the
biggest lever for the "hidden disease" and rare-condition tasks.

- **ECG-FM** — open transformer foundation model, hybrid SSL (masked reconstruction + contrastive
  with ECG-specific augmentation), pretrained on ~1.5 M 12-lead ECGs; fine-tunes to many tasks.
- **CREMA** — Contrastive Regularized Masked Autoencoder with a Signal Transformer; beats supervised
  and prior SSL baselines, robust across clinical domains.
- **CREMA / MAE-style masked autoencoders, contrastive (CLIP-like) and hybrid objectives** are the
  three SSL recipes that dominate.
- **Generative / "ECG language" models** — ECG-Byte tokenizes waveforms for generative
  ECG-language modeling; multimodal ECG↔text directions are emerging.
- **Scaling SSL with SSMs** — state-space backbones for representation learning from ubiquitous ECG.

Practical implication: **pretrain (or adopt a pretrained backbone) once, fine-tune per target.**

### 1.5 Clinical targets and representative SOTA

- **Rhythm / arrhythmia** — AFib, AV block, PVC/VT/VF, multi-label rhythm classification.
  Benchmarks: MIT-BIH, CPSC-2018, PTB-XL, PhysioNet/CinC 2020–2021. F1 ~0.91+ on PTB-XL with
  attention/ensemble models. The classic, most mature task.
- **Structural / "hidden" disease** — low LVEF (≤35–40%), HCM, severe LVH, valvular disease,
  cardiac amyloidosis. **EchoNext** (*Nature* 2025) and Mayo low-EF CNN (AUC ~0.93, EAGLE trial),
  PRESENT-SHD ensemble on ECG images. This is the NYT "damage humans can't see" story.
- **Ischemia / MI** — acute & silent MI, STEMI/NSTEMI, occlusion-MI (ECG-SMART-NET); multimodal
  CNN+RNN+attention on 145k ECGs with cross-hospital validation; image-based MI detection.
- **Risk / prognosis (forecasting, not current state)** — AI-ECG "biological heart age" and
  age-gap predict mortality and MACE (risk rises sharply when ECG age exceeds chronological by
  ~7 yr); ECG-surv predicts time-to-1-year mortality; future-AFib prediction from sinus-rhythm ECG.
  Serial ECGs improve risk estimates.

### 1.6 Cross-cutting techniques

- **Preprocessing** — baseline-wander & powerline filtering, R-peak detection (Pan-Tompkins),
  per-lead normalization, resampling to a common rate, fixed-window or beat segmentation.
- **Augmentation** — time/amplitude scaling, lead masking, Gaussian/baseline noise, random crop,
  mixup; SSL relies heavily on these.
- **Class imbalance / multi-label** — focal loss, class weighting, macro-F1 selection,
  threshold tuning per label.
- **Explainability (mandatory for clinical trust)** — Grad-CAM/saliency over the waveform,
  attention maps, SHAP, and concept attribution back to P-QRS-T morphology.
- **Calibration & external validation** — multi-site/OOD testing is the bar for credibility
  (most published failures are distribution shift across hospitals/devices).

---

## 2. Implementation plan (TensorFlow / Keras)

Goal: a modality-flexible ECG abnormality system that starts with a strong supervised 1-D CNN
baseline on 12-lead signals, adds Transformer/SSM variants, layers self-supervised pretraining,
then extends to single-lead and image inputs and to the structural/prognostic targets.

### 2.1 Design principles

- One **shared encoder** abstraction; swappable heads per task (multi-label rhythm, binary
  structural screen, MI, survival/age regression).
- Modality adapters in front of the encoder so raw-12-lead, single-lead, and image inputs reuse
  the same training/eval harness.
- Reproducible config-driven training; everything benchmarked on public datasets first.

### 2.2 Data layer

- **Datasets**: PTB-XL (primary, multi-label, well-curated), Chapman-Shaoxing, CPSC-2018,
  MIMIC-IV-ECG (scale + linkage to outcomes for prognosis), MIT-BIH (beat-level rhythm).
  For structural targets, ECG↔echo paired cohorts are needed (institutional / restricted).
- **Ingestion**: WFDB readers → arrays; standardize to 500 Hz, 10 s, 12 leads
  `(5000, 12)` tensors. Store as TFRecords / `tf.data` pipelines with on-the-fly augmentation.
- **Splits**: patient-level (never leak a patient across splits); reserve an external/site-held-out
  set for OOD evaluation.

> Note on this repo: it is a **dbt-athena** project. If ECG metadata/outcomes live in S3+Athena,
> dbt models can produce the curated label/cohort tables, and the Keras pipeline reads the
> resulting Parquet from S3. The ML code itself is standalone TF/Keras (per the chosen stack);
> dbt handles the tabular/label ETL only.

### 2.3 Model zoo (Keras)

1. **Baseline — 1-D ResNet** (`tf.keras`): stacked residual Conv1D blocks, GAP, multi-label sigmoid
   head. Target: reproduce PTB-XL macro-AUC in the published ~0.93 range. *This is the milestone
   that de-risks everything.*
2. **CNN + BiLSTM** hybrid for rhythm/MI context.
3. **CNN + Transformer encoder** (MultiHeadAttention) for long-range + inter-lead attention.
4. **State-space block (S4D-style)** implemented as a custom Keras layer for long-sequence,
   OOD-robust modeling.
5. **Image branch**: EfficientNet/ViT (`keras.applications` / KerasCV) for paper/photo ECGs, plus
   a synthetic paper-rendering data generator.

### 2.4 Self-supervised pretraining

- Implement a **masked-autoencoder** pretext (mask spans of the signal, reconstruct) and a
  **contrastive** pretext (ECG-specific augmentations, InfoNCE) on unlabeled MIMIC-IV-ECG.
- Hybrid loss (reconstruction + contrastive), mirroring ECG-FM / CREMA.
- Then **fine-tune** the shared encoder per target. This is the path to good rare-condition and
  structural-disease performance.

### 2.5 Task heads

- Rhythm/arrhythmia: multi-label sigmoid + focal loss, per-label threshold tuning.
- Structural screen (low-EF / SHD): binary/few-label sigmoid; report sensitivity at fixed
  specificity (screening operating point).
- MI: multi-label (location-aware) classification.
- Prognosis: ECG-age **regression** (MAE) and **survival** head (Cox / discrete-time hazard) for
  mortality time-to-event.

### 2.6 Training & evaluation

- `tf.data` + mixed precision; AdamW, cosine schedule, early stopping on macro-AUC/F1.
- Metrics: macro & per-class AUROC/AUPRC, F1, sensitivity@specificity, calibration (ECE),
  and **external/OOD** numbers reported separately.
- Explainability: Grad-CAM over Conv1D feature maps + attention visualization + SHAP on a tabular
  fusion variant.
- Track experiments (MLflow / Weights & Biases); serialize SavedModel for inference.

### 2.7 Phased roadmap

| Phase | Deliverable | Datasets | Exit criterion |
| --- | --- | --- | --- |
| 0 | Data pipeline + EDA (`tf.data`, WFDB→TFRecord) | PTB-XL | Loaders + patient-level splits verified |
| 1 | 1-D ResNet baseline + eval harness | PTB-XL | Macro-AUC in published range; reproducible |
| 2 | Transformer & S4D variants + explainability | PTB-XL, CPSC, Chapman | Match/beat baseline; OOD report |
| 3 | SSL pretrain (MAE+contrastive) + fine-tune | MIMIC-IV-ECG | Fine-tune > from-scratch on PTB-XL |
| 4 | Structural + prognosis heads | paired ECG-echo / MIMIC outcomes | Low-EF/SHD screen + ECG-age/survival |
| 5 | Single-lead + ECG-image branches | reduced-lead, synthetic images | Modality parity within tolerance |

### 2.8 Risks & guardrails

- **Distribution shift** across sites/devices is the top failure mode — bake OOD eval in from
  Phase 1, never trust single-site numbers.
- **Label leakage** via patient overlap — enforce patient-level splits.
- **Structural/prognosis data access** — paired ECG-echo and outcome-linked cohorts are usually
  institutional/IRB-gated; plan for restricted access.
- **Clinical claims** — this is research/decision-support scaffolding, not a diagnostic device;
  any deployment needs prospective validation and regulatory review.

---

## 3. References (public sources consulted)

Architectures, benchmarks & SSL

- [DeepECG-Net: hybrid transformer for real-time ECG anomaly detection (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-07781-1)
- [Interpretable DL for arrhythmia on PTB-XL (Diagnostics 2025)](https://www.mdpi.com/2075-4418/15/15/1950)
- [Interpretable multi-label ECG framework (feature attention + explainability)](https://www.sciencedirect.com/science/article/abs/pii/S1746809425009589)
- [ECG-FM: open electrocardiogram foundation model (JAMIA Open 2025)](https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827)
- [CREMA: Contrastive Regularized MAE for ECG (arXiv)](https://arxiv.org/abs/2407.07110)
- [ECG-Byte: tokenizer for generative ECG-language modeling (arXiv)](https://arxiv.org/pdf/2412.14373)
- [S4D-ECG: shallow state-of-the-art cardiac abnormality model (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11239723/)
- [Scaling representation learning from ECG with state-space models (arXiv)](https://arxiv.org/html/2309.15292)

Structural / "hidden" heart disease (the NYT angle)

- [Detecting structural heart disease from ECG using AI — EchoNext (Nature 2025)](https://www.nature.com/articles/s41586-025-09227-0)
- [Can AI detect hidden heart disease? — EchoNext (Columbia/NYP)](https://www.cuimc.columbia.edu/news/can-ai-detect-hidden-heart-disease)
- [Multisite external validation of AI-ECG for low ejection fraction (JACC Advances 2025)](https://www.jacc.org/doi/10.1016/j.jacadv.2025.102537)
- [PRESENT-SHD: ensemble DL for structural heart disease from ECG images (JACC 2025)](https://www.jacc.org/doi/abs/10.1016/j.jacc.2025.01.030)
- [Mayo: AI-guided early detection of heart disease in routine practice](https://newsnetwork.mayoclinic.org/discussion/trial-demonstrates-early-ai-guided-detection-of-heart-disease-in-routine-practice/)
- [Postdevelopment validation of AI-ECG for cardiac amyloidosis (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025724/)

Ischemia / MI

- [Multimodal DL for acute MI from 12-lead ECG, cross-hospital validation (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12853118/)
- [ECG-SMART-NET: occlusion MI detection (arXiv)](https://arxiv.org/pdf/2405.09567)
- [DL for MI detection using ECG images — systematic review (Mathematics 2025)](https://doi.org/10.3390/math14040613)
- [Synthetic-data pipeline for paper-ECG interpretation (arXiv 2025)](https://arxiv.org/html/2507.21968v1)

Risk / prognosis

- [ECG-surv: time-to-1-year mortality from 12-lead ECG (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11751416/)
- [AI-ECG biological age gap and mortality with serial ECGs (Heart Rhythm 2025)](https://www.heartrhythmjournal.com/article/S1547-5271(25)02432-4/abstract)
- [AI-ECG biological heart age predicts mortality and CV outcomes (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10133724/)

Single-lead / wearable

- [Accuracy of single-lead Apple Watch ECG for AFib (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9795256/)
- [Comparative evaluation of consumer wearables for AFib detection (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11737281/)
