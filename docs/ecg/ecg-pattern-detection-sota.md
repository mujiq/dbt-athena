# ECG Pattern Detection — Detailed Technical Specification (SOTA + Open-Source Model Catalog)

> Status: **specification only — no code to be written yet** (per requester instruction, 2026-06-23).
> Scope agreed earlier: deliverable = research + implementation spec; input modality = any/multiple
> (raw 12-lead, single-lead/wearable, ECG images); targets = rhythm/arrhythmia, structural ("hidden")
> disease, ischemia/MI, risk/prognosis; target stack = TensorFlow / Keras.
>
> This revision (v2) adds, per instruction, a catalog of **all public open-source ECG models** with an
> explicit evaluation of **reported accuracy** and **production / clinical-use maturity** for each.

---

## 1. Objectives and non-goals

**Objectives**

- Specify, in implementation-ready detail, a modality-flexible system that reads ECGs to detect
  abnormal patterns across four clinical target classes.
- Catalog every credible public open-source ECG model/codebase, with accuracy and production status,
  so we can decide what to adopt, fine-tune, or use only as a reference baseline.
- Define data, components, evaluation, and a phased roadmap — without writing code yet.

**Non-goals (for now)**

- No code, notebooks, or model weights produced in this phase.
- Not a regulatory submission. Any clinical deployment requires prospective validation and
  FDA/CE review; this spec treats all OSS models as **research/decision-support, not diagnostic devices**.

---

## 2. Background — the "hidden heart damage" angle

The NYT article (2026-06-22, paywalled/blocked here) covers AI that reads ordinary ECGs to surface
heart damage clinicians cannot see. The flagship example is **EchoNext** (Columbia / NewYork-Presbyterian,
*Nature*, July 2025): trained on 1.2M ECG–echo pairs, it flags **structural heart disease** for
echocardiography, scoring AUROC ~0.85 internally and beating cardiologists in a head-to-head. This sits
in the **structural** and **prognostic** branches of the taxonomy below, alongside Mayo's low-EF work
(now commercialized and FDA-cleared as Anumana/Eko ELEFT).

---

## 3. Taxonomy (three orthogonal axes)

### 3.1 Input modality

| Modality | Signal | Primary use | Reference datasets |
| --- | --- | --- | --- |
| Raw 12-lead | 10 s, 250–500 Hz, 12 ch | Clinical diagnosis | PTB-XL, MIMIC-IV-ECG, Chapman-Shaoxing, CODE |
| Single / reduced lead | 1–3 leads, ambulatory | Screening, wearables | iRhythm/Zio-style, Apple Watch, CinC-2021 reduced-lead |
| ECG image / paper | photo or PDF strip | Legacy, low-resource | PTB-XL→image, ECG-Image-Kit synthetic, CinC-2024 |

### 3.2 Model architecture family

- **1-D CNN / ResNet** — clinical workhorse (Ribeiro, Mayo, Strodthoff baselines).
- **CNN + RNN (LSTM/GRU)** — local morphology + rhythm context.
- **Transformers / attention** — long-range + inter-lead dependencies; hybrid CNN+Transformer.
- **State-space models (S4 / S4D / Mamba)** — long-sequence, strong out-of-distribution, low compute.
- **Vision transformers / 2-D CNN** — for image/paper ECGs.
- **Self-supervised foundation models** — masked-reconstruction, contrastive, or hybrid pretraining.

### 3.3 Clinical target

Rhythm/arrhythmia · structural ("hidden") disease · ischemia/MI · risk/prognosis (forecasting).

---

## 4. Open-source model catalog (accuracy + production-use evaluation)

**Reading guide.** "Reported accuracy" = numbers from the source paper/repo (test set differs per model;
not directly comparable). "Production use" rates real-world maturity on this scale:

- **P0 Research-only** — code/weights for reproduction; no clinical validation.
- **P1 Externally validated** — multi-site / OOD validation in literature, but not a product.
- **P2 Deployed (non-regulated)** — used in real clinical workflows without device clearance.
- **P3 Regulated product** — an FDA/CE-cleared product derives from this lineage (often a closed fork).

> **License caution:** verify each repo's LICENSE before any non-research use. Known restrictive ones
> are flagged. Most PhysioNet datasets require a **credentialed Data Use Agreement** and many are
> non-commercial. Treat all unflagged licenses as "verify before production."

### 4.0 Master comparison

| Model | Type | Arch | Pretrain/train data | Reported accuracy | Prod | Repo |
| --- | --- | --- | --- | --- | --- | --- |
| automatic-ecg-diagnosis (Ribeiro) | Supervised | 1-D ResNet | CODE (~2M ECGs) | F1 > 0.80 on 6 classes; > resident cardiologists | P2 | antonior92/automatic-ecg-diagnosis |
| ecg_ptbxl_benchmarking (Strodthoff) | Supervised benchmark | xresnet1d101, inception1d, etc. | PTB-XL | macro-AUROC ≈0.92–0.93 ("all" task) | P0 | helme/ecg_ptbxl_benchmarking |
| awni/ecg (Hannun/Rajpurkar) | Supervised | 34-layer 1-D CNN | 91k single-lead | F1 ≈0.84 > avg cardiologist 0.78 | P3* | awni/ecg |
| ECGFounder | Foundation (supervised) | deep CNN | 10.7M ECGs, 150 labels | AUROC > 0.95 for 80 dx; single-lead capable | P1 | PKUDigitalHealth/ECGFounder |
| HuBERT-ECG | Foundation (SSL) | HuBERT/transformer | 9.1M ECGs, 164 cond. | AUROC 0.84–0.99; 2-yr mortality 0.934 | P1 | Edoardo-BS/HuBERT-ECG |
| ECG-FM | Foundation (SSL) | wav2vec2.0 (90.9M) | MIMIC-IV-ECG + CinC-2021 (~1.5M) | strong on CinC-2021/MIMIC tasks | P0/P1 | bowang-lab/ECG-FM |
| ST-MEM | SSL pretext | masked ECG MAE (ViT) | 12-lead | > SSL baselines on arrhythmia (FT + linear) | P0 | vuno/ST-MEM |
| HeartBEiT | Foundation (SSL) | vision transformer (BEiT) | 8.5M ECG images | > CNNs, esp. at low sample size | P1 | akhilvaid/HeartBEiT |
| CLOCS | SSL pretext | contrastive (CMSC/CMLC) | ECG | > SimCLR/BYOL; strong at 25% labels | P0 | danikiyasseh/CLOCS |
| MERL | Multimodal/zero-shot | ECG–text contrastive | ECG + reports | zero-shot AUC ≈0.75 (6 sets) | P0 | cheliu-computation/MERL-ICML2024 |
| IntroECG / EchoNext | Task (structural) | 1-D CNN | 1.2M ECG–echo pairs | AUROC 0.85 internal; 0.78–0.80 external | P1→P2 | PierreElias/IntroECG |
| ISIBrno-AIMT (CinC-2021) | Supervised | ResNet + multi-head attn ensemble | CinC-2021 multi-source | challenge score 0.58 (all lead sets) | P0 | (challenge code; DeepPSP/cinc2021 mirror) |
| ECG-Image-Kit | Tooling | synthetic image generator | n/a (generator) | n/a (data tool) | P0 | alphanumericslab/ecg-image-kit |
| ECG-Digitiser | Task (digitization) | Hough + DL | CinC-2024 | CinC-2024 winner | P0 | felixkrones/ECG-Digitiser |
| NeuroKit2 | Tooling | classical DSP | n/a | n/a (preprocessing) | P2 | neuropsychology/NeuroKit |

\* P3 = a regulated product exists in the same lineage/space (iRhythm), not from this exact repo.

### 4.1 Supervised baselines

**antonior92/automatic-ecg-diagnosis — Ribeiro et al., *Nature Communications* 2020 (TensorFlow/Keras).**
1-D residual CNN classifying 6 ECG abnormality superclasses (1st/2nd/3rd-degree AV block, RBBB, LBBB,
sinus brady/tachy, AF). Trained on the large Brazilian **CODE** cohort. *Accuracy:* F1 and specificity
that matched or exceeded resident cardiologists on a 827-ECG annotated test set. *Production:* **P2** —
the lineage underpins real telehealth screening at scale in Brazil (Telehealth Network of Minas Gerais).
*Fit:* ideal reference baseline — already Keras, our exact target stack.

**helme/ecg_ptbxl_benchmarking — Strodthoff et al., *IEEE JBHI* 2021 (PyTorch/fastai).**
The canonical PTB-XL leaderboard: `xresnet1d101`, `inception1d`, `resnet1d_wang`, LSTM, FCN, plus
wavelet+ML baselines. *Accuracy:* top 1-D models cluster around **macro-AUROC ≈0.92–0.93** on the
"all" superdiagnostic task (exact value varies by subtask; the repo is the source of truth). *Production:*
**P0** — research benchmark. *Fit:* defines the metric/split protocol we should replicate; port the
winning architectures to Keras as our Phase-1/2 baselines.

**awni/ecg — Hannun & Rajpurkar et al., *Nature Medicine* 2019 (Keras).**
34-layer 1-D CNN mapping single-lead ambulatory ECG to 12 rhythm classes; 91k recordings.
*Accuracy:* sequence-level **F1 ≈0.837**, exceeding the averaged individual cardiologist (0.780);
ROC-AUC ~0.97. *Production:* **P3\*** — single-lead ambulatory rhythm detection is a commercialized
space (iRhythm Zio); this repo itself is research. *Fit:* reference for the single-lead/wearable branch.

### 4.2 Foundation & self-supervised models

**ECGFounder (PKUDigitalHealth) — *NEJM AI* 2025. Weights on HuggingFace.**
Supervised foundation model on **10.7M** ECGs (Harvard–Emory), 150 diagnostic labels; extends to
reduced/single-lead. *Accuracy:* **AUROC > 0.95 for 80 diagnoses**, expert-level on internal validation,
external eval across domains. *Production:* **P1** (external multi-domain validation). *Fit:* strongest
out-of-the-box 12-lead + single-lead backbone candidate; fine-tune per target. Verify license for any
non-research use.

**HuBERT-ECG (Edoardo-BS) — medRxiv 2024/25. License: CC BY-NC 4.0 (non-commercial).**
Self-supervised (HuBERT-style masked prediction) on **9.1M** ECGs, 164 conditions. *Accuracy:* AUROC
**0.84–0.99** across tasks; supportive dx 0.76–0.97; prognosis 0.74–0.91; single-lead 0.88–0.92;
**2-year mortality AUROC 0.934.** *Production:* **P1.** *Fit:* excellent breadth incl. prognosis, but
**CC BY-NC blocks commercial deployment** — research/benchmarking only unless relicensed.

**ECG-FM (bowang-lab) — *JAMIA Open* 2025; built on `Jwoo5/fairseq-signals`.**
wav2vec-2.0 transformer (**90.9M** params), hybrid SSL (W2V + CMSC + RLM) on MIMIC-IV-ECG + PhysioNet-2021
(~1.5M ECGs). *Accuracy:* strong on CinC-2021/MIMIC downstream benchmarks (open, reproducible). *Production:*
**P0/P1.** *Fit:* fully open weights+code+tutorials; best-documented OSS foundation model — primary
candidate for our SSL backbone if we standardize on the fairseq-signals toolchain.

**ST-MEM (vuno / bakqui) — ICLR 2024.**
Spatio-temporal masked ECG modeling (MAE with lead embeddings, separation tokens); reconstructs masked
12-lead patches. *Accuracy:* outperforms SSL baselines on arrhythmia in both fine-tuning and linear
probing; adaptable to arbitrary lead subsets. *Production:* **P0.** *Fit:* clean reference design for our
own masked-autoencoder pretext (Phase 3).

**HeartBEiT (akhilvaid) — *npj Digital Medicine* 2023.**
Vision transformer via masked **image** modeling on **8.5M** ECG images. *Accuracy:* beats CNNs,
especially at low sample sizes (matches CNNs with ~1/10 the labeled data); better waveform-region
explainability. *Production:* **P1.** *Fit:* anchor model for the image/paper-ECG branch and for
low-label structural tasks.

**CLOCS (danikiyasseh) — ICML 2021.**
Contrastive learning across space/time/patients (CMSC, CMLC, CMSMLC). *Accuracy:* beats SimCLR/BYOL on
linear-probe and fine-tune; strong with only 25% labels. *Production:* **P0.** *Fit:* reference for the
contrastive half of our hybrid SSL objective.

**CREMA — Contrastive Regularized MAE (Signal Transformer), arXiv 2024.**
Hybrid generative+contrastive SSL; reports robustness across clinical domains, beating supervised and
prior SSL baselines. *Production:* **P0.** *Fit:* design reference for combining MAE + contrastive
(same recipe family as ECG-FM/CREMA we plan to implement). Confirm code/weights availability.

### 4.3 Multimodal / zero-shot

**MERL (cheliu-computation) — ICML 2024.**
Multimodal ECG–text contrastive pretraining; at test time uses LLM-built clinical prompts (CKEPE) for
**zero-shot** classification. *Accuracy:* zero-shot **AUC ≈0.752** averaged over 6 datasets — beating
linear-probed eSSL methods that use 10% labels. *Production:* **P0.** *Fit:* enables label-free screening
for new conditions; relevant if we want open-set / promptable diagnosis.

### 4.4 Image input & digitization

**ECG-Image-Kit (alphanumericslab) — *Physiological Measurement* 2024.**
Generates realistic synthetic paper-ECG images (wrinkles, handwriting, perspective, noise) with
ground-truth signals; powered the CinC-2024 challenge. *Production:* **P0 (tool).** *Fit:* our synthetic
data engine to train robust image/digitization models without scarce labeled photos.

**ECG-Digitiser (felixkrones) — PhysioNet/CinC 2024 winner.**
Hough-transform + deep learning to reconstruct signals from printouts. *Accuracy:* top of CinC-2024.
*Production:* **P0.** *Fit:* reference for the "image → signal → existing signal model" pipeline.

**PRESENT-SHD — *JACC* 2025 (ensemble CNN + XGBoost on ECG images for structural heart disease).**
*Accuracy:* composite structural-disease screen across LVEF<40%, valvular disease, severe LVH.
*Production:* **P1.** *Fit:* design reference for image-based structural screening; confirm code release.

### 4.5 Task-specific: structural & prognosis

**IntroECG / EchoNext (PierreElias) — *Nature* 2025.**
Library for 12-lead deep learning behind EchoNext (structural heart disease). *Accuracy:* AUROC **0.852**
internal; **0.78–0.80** external (Cedars-Sinai, UCSF, Montreal). The **EchoNext-Mini** dataset (100k ECGs
with echo-derived labels) is public. *Production:* **P1→P2** (prospective clinical evaluation underway).
*Fit:* the closest open reference to the NYT "hidden damage" target; primary template for our structural head.

**ECG-surv / AI-ECG biological age** — open methods/papers (repos vary).
Time-to-1-year-mortality (ECG-surv) and "biological heart age" gap predicting mortality/MACE.
*Production:* **P0/P1.** *Fit:* design references for our survival and ECG-age regression heads (Phase 4).

### 4.6 Challenge solutions (multi-source, robust)

**ISIBrno-AIMT — PhysioNet/CinC 2021 winner.**
ResNet + multi-head attention, 3-model majority-vote ensemble, custom challenge-score + sparsity losses,
evolutionary per-class thresholds; handles 12/6/4/3/2-lead inputs. *Accuracy:* challenge score **0.58**
across all lead configs. *Production:* **P0.** *Fit:* reference for reduced-lead robustness and
multi-label thresholding. `DeepPSP/cinc2021` is a clean open reimplementation of the challenge pipeline.

### 4.7 Tooling / preprocessing

**NeuroKit2 (neuropsychology/NeuroKit)** — classical DSP: filtering, R-peak detection, delineation,
quality indices, HRV. *Production:* **P2** (widely used). *Fit:* preprocessing + feature baseline and
sanity checks; not a model.

### 4.8 Commercial / FDA-cleared lineage (context — NOT open source)

Included only to ground the "production use" axis; **do not treat as adoptable OSS**.

- **Anumana ECG-AI LEF / Eko ELEFT** — first **FDA 510(k)-cleared** AI-ECG for low ejection fraction
  (Oct 2023), from Mayo/Nference research. Validated on 16k patients: **sensitivity 84.5%, specificity
  83.6%.** This is the productized version of the structural/low-EF lineage. **P3.**
- **iRhythm Zio / AliveCor KardiaAI** — commercial single-lead arrhythmia detection (same space as the
  Hannun/Stanford model). **P3.**

---

## 5. Production-readiness summary

- **Adoptable now, permissive enough for our work (verify license):** ECG-FM (best-documented OSS
  foundation), ECGFounder (highest reported breadth/accuracy, single-lead), IntroECG/EchoNext (structural),
  Ribeiro automatic-ecg-diagnosis (Keras baseline), ecg_ptbxl_benchmarking (protocol + baselines).
- **Use as design references / pretext implementations:** ST-MEM, CLOCS, CREMA, HeartBEiT (image),
  ISIBrno/DeepPSP (reduced-lead, thresholding), ECG-Image-Kit + ECG-Digitiser (image pipeline),
  ECG-surv / AI-ECG-age (prognosis heads).
- **Research/benchmark only due to license:** HuBERT-ECG (**CC BY-NC 4.0** — no commercial use), and any
  model whose weights ride on PhysioNet **credentialed** data agreements (data-redistribution limits).
- **No OSS model is FDA-cleared.** Only the closed commercial forks (Anumana/Eko, iRhythm, AliveCor) are
  regulated products. Any clinical claim from our system needs its own prospective validation + clearance.

---

## 6. Recommended model selection (for the build phase)

1. **Backbone:** start from **ECG-FM** (open weights + tutorials, fairseq-signals) and/or **ECGFounder**
   (highest reported accuracy + single-lead); benchmark both, fine-tune per target.
2. **Keras baseline:** reproduce **Ribeiro 1-D ResNet** (already Keras) and the **PTB-XL xresnet1d101**
   protocol as our internal yardsticks.
3. **SSL (if we pretrain ourselves):** implement **MAE (ST-MEM-style) + contrastive (CLOCS-style)** hybrid,
   matching the ECG-FM/CREMA recipe.
4. **Image branch:** **HeartBEiT** design + **ECG-Image-Kit** synthetic data + **ECG-Digitiser** pipeline.
5. **Structural / prognosis:** **IntroECG/EchoNext** template + **EchoNext-Mini** for prototyping; ECG-age
   and survival heads from the prognosis references.
6. **Reduced-lead robustness:** borrow **ISIBrno/DeepPSP** thresholding + multi-lead training.

Orchestration note: the ECG/ML pipeline **does not depend on dbt**. Use **Dagster** as the open-source
orchestrator — its software-defined **assets** provide dbt-like lineage for the curated cohort/label
tables *and* orchestrate the Python TF/Keras training/eval steps in one tool, with native S3 integration.
For heavy distributed/GPU training, **Flyte** or **Metaflow** are acceptable; **Prefect** is the
lightweight fallback. Data may live in S3 (with AWS Athena as an optional query layer), but lineage and
orchestration come from Dagster, not dbt.

---

## 7. Data specification

- **Primary public datasets:** PTB-XL (multi-label, curated), Chapman-Shaoxing, CPSC-2018, CinC-2020/2021
  (multi-source, reduced-lead), MIT-BIH (beat-level rhythm), **MIMIC-IV-ECG** (scale + outcome linkage for
  SSL & prognosis), **CODE/CODE-test** (Ribeiro), **EchoNext-Mini** (structural, 100k with echo labels).
- **Canonical tensor:** resample to 500 Hz, 10 s, 12 leads → `(5000, 12)`. Single-lead/image branches map
  into the same downstream interface via modality adapters.
- **Splits:** **patient-level** only (no patient across train/val/test). Hold out an external site for OOD.
- **Governance:** PhysioNet credentialed DUA for restricted sets; track per-dataset license and
  commercial-use constraints in a data manifest.

---

## 8. Component specifications (functional — no code)

1. **Modality adapter** — normalizes raw-12-lead, reduced-lead (zero/learned-fill missing leads), and
   image (digitize via ECG-Digitiser-style pipeline, or feed image model) into a shared encoder interface.
2. **Preprocessing** — baseline-wander + powerline filtering, resample to 500 Hz, per-lead normalization,
   R-peak detection (NeuroKit2/Pan-Tompkins), fixed 10 s window (or beat segmentation for rhythm).
3. **Augmentation** — time/amplitude scaling, lead masking/dropout, Gaussian + baseline noise, random crop,
   mixup; required for SSL pretexts.
4. **Encoder (swappable)** — 1-D ResNet (baseline) | CNN+Transformer | S4D | imported foundation backbone.
5. **Task heads** — multi-label rhythm (sigmoid + focal loss, per-class thresholds); binary/few-label
   structural screen (report sensitivity@fixed-specificity); MI (location-aware multi-label); prognosis
   (ECG-age regression + discrete-time/Cox survival).
6. **SSL module** — MAE reconstruction + contrastive objective; pretrain on unlabeled MIMIC-IV-ECG;
   fine-tune per head.
7. **Evaluation harness** — macro/per-class AUROC + AUPRC, F1, sensitivity@specificity, calibration (ECE),
   bootstrap CIs; **separate internal vs external/OOD reporting**.
8. **Explainability** — Grad-CAM/saliency over the waveform, attention maps, SHAP; map attributions back to
   P-QRS-T morphology for clinical review.
9. **Serving** — SavedModel export; batch inference over S3-curated cohorts (Athena optional as a query
   layer); experiment tracking (MLflow / W&B); model + data + license manifest per run.
10. **Orchestration** — **Dagster** software-defined assets cover the whole DAG: raw ECG ingestion →
    curated cohort/label tables → preprocessing → training → evaluation → registered model, with
    asset-level lineage (the dbt-replacement). Flyte/Metaflow for distributed GPU training; Prefect as a
    lightweight fallback. **No dbt dependency.**

---

## 9. Phased roadmap

| Phase | Deliverable | Datasets | Exit criterion |
| --- | --- | --- | --- |
| 0 | Dagster asset pipeline + EDA; data/license manifest | PTB-XL, CODE | Patient-level loaders verified |
| 1 | Keras baselines (Ribeiro ResNet + PTB-XL protocol) | PTB-XL, CODE | Macro-AUROC ≈ published; reproducible |
| 2 | Transformer & S4D variants + explainability | PTB-XL, CPSC, Chapman | Match/beat baseline; OOD report |
| 3 | Adopt foundation backbone (ECG-FM/ECGFounder) + own SSL | MIMIC-IV-ECG | Fine-tune > from-scratch |
| 4 | Structural + prognosis heads (EchoNext template) | EchoNext-Mini, outcomes | Low-EF/SHD screen + ECG-age/survival |
| 5 | Single-lead + image/digitization branches | CinC reduced-lead, synthetic images | Modality parity within tolerance |

---

## 10. Risks and guardrails

- **Distribution shift** across sites/devices is the top failure mode — OOD eval from Phase 1; never trust
  single-site numbers.
- **Label leakage** via patient overlap — enforce patient-level splits.
- **License/regulatory** — HuBERT-ECG is non-commercial; PhysioNet sets are credentialed; no OSS model is
  FDA-cleared. Track per-asset license; gate commercial use.
- **Data access** for structural/prognosis — paired ECG–echo and outcome-linked cohorts are mostly
  institutional/IRB-gated (EchoNext-Mini partially mitigates).
- **Clinical claims** — research/decision-support only until prospective validation + clearance.

---

## 11. References (public sources)

Open-source models & code

- [antonior92/automatic-ecg-diagnosis (Ribeiro et al., Nat Commun 2020)](https://github.com/antonior92/automatic-ecg-diagnosis)
- [helme/ecg_ptbxl_benchmarking (Strodthoff et al., IEEE JBHI 2021)](https://github.com/helme/ecg_ptbxl_benchmarking)
- [awni/ecg (Hannun & Rajpurkar et al., Nat Med 2019)](https://github.com/awni/ecg)
- [PKUDigitalHealth/ECGFounder (NEJM AI 2025)](https://github.com/PKUDigitalHealth/ECGFounder) · [weights](https://huggingface.co/PKUDigitalHealth/ECGFounder)
- [Edoardo-BS/HuBERT-ECG (CC BY-NC 4.0)](https://github.com/Edoardo-BS/HuBERT-ECG) · [paper](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v2.full)
- [bowang-lab/ECG-FM (JAMIA Open 2025)](https://github.com/bowang-lab/ECG-FM/) · [fairseq-signals](https://github.com/Jwoo5/fairseq-signals)
- [vuno/ST-MEM (ICLR 2024)](https://github.com/vuno/ST-MEM)
- [akhilvaid/HeartBEiT (npj Digit Med 2023)](https://github.com/akhilvaid/HeartBEiT) · [paper](https://www.nature.com/articles/s41746-023-00840-9)
- [danikiyasseh/CLOCS (ICML 2021)](https://github.com/danikiyasseh/CLOCS)
- [cheliu-computation/MERL-ICML2024](https://github.com/cheliu-computation/MERL-ICML2024) · [paper](https://arxiv.org/abs/2403.06659)
- [PierreElias/IntroECG (EchoNext, Nature 2025)](https://github.com/PierreElias/IntroECG) · [paper](https://www.nature.com/articles/s41586-025-09227-0) · [EchoNext-Mini](https://physionet.org/content/echonext/1.1.1/)
- [DeepPSP/cinc2021 (PhysioNet/CinC 2021)](https://github.com/DeepPSP/cinc2021)
- [felixkrones/ECG-Digitiser (CinC 2024 winner)](https://github.com/felixkrones/ECG-Digitiser)
- [ECG-Image-Kit (Physiol Meas 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11135178/)
- [CREMA: Contrastive Regularized MAE (arXiv)](https://arxiv.org/abs/2407.07110)
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit)

Structural / "hidden" disease, prognosis, MI

- [EchoNext — structural heart disease from ECG (Nature 2025)](https://www.nature.com/articles/s41586-025-09227-0)
- [Multisite validation of AI-ECG for low EF (JACC Advances 2025)](https://www.jacc.org/doi/10.1016/j.jacadv.2025.102537)
- [PRESENT-SHD ensemble on ECG images (JACC 2025)](https://www.jacc.org/doi/abs/10.1016/j.jacc.2025.01.030)
- [ECG-surv: time-to-1-year mortality (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11751416/)
- [AI-ECG biological heart age predicts mortality/MACE (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10133724/)
- [Multimodal DL for acute MI, cross-hospital validation (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12853118/)
- [ECG-SMART-NET: occlusion MI (arXiv)](https://arxiv.org/pdf/2405.09567)

Commercial / FDA context (not OSS)

- [Anumana ECG-AI LEF — FDA 510(k) clearance](https://www.businesswire.com/news/home/20231002851453/en/Anumana-Receives-U.S.-FDA-510k-Clearance-for-ECG-AI-Algorithm-to-Detect-Low-Ejection-Fraction)
- [FDA clears first AI to aid heart failure detection (Eko)](https://www.prnewswire.com/news-releases/fda-clears-first-ai-to-aid-heart-failure-detection-during-routine-check-ups-302103718.html)

Datasets & benchmarks

- [PTB-XL deep learning benchmark (arXiv)](https://arxiv.org/pdf/2004.13701)
- [PhysioNet/CinC Challenge 2021](https://physionet.org/content/challenge-2021/1.0.1/)
- [PhysioNet/CinC Challenge 2024 (image digitization)](https://moody-challenge.physionet.org/2024/)
- [CODE-test annotated 12-lead ECG dataset](https://zenodo.org/records/3765780)
