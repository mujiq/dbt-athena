# Replication Spec: An ECG Biomarker for Sudden Cardiac Death Discovered with Deep Learning

Status: Draft v1.0
Owner: TBD
Last updated: 2026-06-28

## 0. About this document

This is a single, detailed engineering and research specification for **replicating the full discovery pipeline**
described in:

> Obermeyer, Z., Schubert, A., Ross, J. et al. *An ECG biomarker for sudden cardiac death discovered with deep
> learning.* Nature (2026). Article: `s41586-026-10674-6`.

The "capability" being replicated is not a single model but a **method**: train a deep neural network to predict
sudden cardiac death (SCD) directly from the raw electrocardiogram (ECG), then **reverse-engineer the network into a
simple, hand-computable, interpretable biomarker** and validate that biomarker across independent populations.

Scope decisions for this spec (agreed up front):

- **Scope:** full discovery pipeline (deep predictive model -> interpretability/distillation -> biomarker validation).
- **Data:** openly available / research-accessible ECG datasets (no proprietary Swedish registry access assumed).
- **Stack:** Python + PyTorch, multi-GPU cluster.
- **Deliverable:** this document.

> Important framing. The original study's distinctive asset is a **population-complete registry** (every ECG in a
> Swedish region linked to death certificates with cause-of-death coding for SCD). No public dataset reproduces that
> exactly. This spec therefore replicates the **method and its scientific logic** on public data, and is explicit
> about where public data forces a substitution (most importantly, the SCD outcome label). Treat any numeric target
> from the paper as a *reference point*, not a guaranteed reproduction.

---

## 1. Summary of the capability to be replicated

### 1.1 What the original paper did

1. **Data linkage.** Linked all ECGs collected in a Swedish region to national death certificates, giving each ECG a
   downstream mortality outcome, including cause-of-death coding sufficient to identify **sudden cardiac death**.
2. **Deep model.** Trained a deep neural network on the raw 12-lead ECG waveform to predict SCD risk.
3. **Risk stratification result.** The model isolated a high-risk group (~2.2% of the sample) with a ~7.0% annual SCD
   rate, exceeding the standard clinical criterion of reduced left-ventricular ejection fraction (LVEF) group (~1.9%
   of the sample; ~4.6% annual rate). Critically, **~86.1% of the model's high-risk patients were not flagged by
   LVEF** — i.e., the model finds risk that current practice misses.
4. **Interpretability / biomarker discovery.** The learned signal was distilled into a simple geometric feature: the
   **smoothness of the terminal QRS region** — quantified as the **mean absolute first and second differences of
   voltage from the R peak to the end of the QRS**, specifically in **lead aVL**. Greater smoothness of the terminal
   R-wave region robustly predicted SCD.
5. **External validation.** Validated in a US health system (where it predicts lethal ventricular arrhythmias) and a
   Taiwanese hospital registry (where it predicts future arrhythmic cardiac arrests) — i.e., the biomarker
   generalizes across populations and across related arrhythmic endpoints.
6. **Clinical-utility signal.** Among high-risk patients who received an implantable cardioverter-defibrillator (ICD),
   observed mortality was ~54.4% lower than expected, consistent with a real, actionable, treatable risk.

### 1.2 Why this is hard and worth replicating

- The headline scientific claim is **discovery of a previously unappreciated, mechanistically plausible, interpretable
  biomarker** from a black-box model — not merely a high AUROC.
- Replication value lies in reproducing the **pipeline that turns a deep model into a clinician-checkable rule**, and
  showing it transfers across datasets and endpoints.

### 1.3 Replication targets (what "success" looks like)

| # | Target | Definition | Pass criterion (public-data realistic) |
|---|--------|-----------|-----------------------------------------|
| T1 | Predictive model | Deep model predicts the chosen arrhythmic/mortality endpoint from raw ECG | Held-out time-dependent C-index materially above demographic + LVEF baseline |
| T2 | Risk concentration | Top model-defined risk group concentrates events | Top ~2-3% risk bucket has event rate multiple times the cohort base rate |
| T3 | Complementarity to LVEF | Model flags risk LVEF misses | Large fraction of model-high-risk are LVEF-normal (qualitatively reproduce the "missed by LVEF" finding) |
| T4 | Biomarker rediscovery | Distilled aVL terminal-QRS smoothness feature is predictive | Single hand-computed feature is significantly associated with the endpoint, same direction (smoother -> higher risk) |
| T5 | Biomarker recovers model | Simple feature(s) recover much of the deep model's signal | Surrogate model on a handful of interpretable features retains a large share of deep-model discrimination |
| T6 | External transfer | Biomarker + model transfer to a second dataset/endpoint | Effect direction preserved and discrimination above baseline on a fully held-out dataset |

---

## 2. Outcome definition (the crux for public data)

The original SCD label comes from **death certificates with cause-of-death coding**. Public ECG corpora rarely carry
adjudicated SCD. The spec defines a **tiered outcome strategy**; pick the highest tier each dataset supports and record
the choice per dataset.

| Tier | Endpoint | How obtained | Closeness to true SCD | Datasets that support it |
|------|----------|--------------|------------------------|--------------------------|
| A | Cause-coded sudden cardiac / arrhythmic death | Death registry ICD-10 (e.g., I46.x cardiac arrest, I49.0, I47.2, R96 sudden death) | Highest | UK Biobank (death registry linkage) |
| B | Ventricular arrhythmia / cardiac arrest event | Hospital diagnosis/procedure codes (VT/VF, resuscitated arrest, ICD shock) | High (matches the US/Taiwan validation endpoints in the paper) | MIMIC-IV (ICU + hospital), UK Biobank HES |
| C | Cardiovascular death | Death registry, CV chapter ICD-10 (I00-I99) | Medium | UK Biobank |
| D | All-cause mortality within horizon | Any death record / discharge disposition | Low (use only as a coarse proxy / sanity check) | MIMIC-IV, Code-15/CODE (mortality follow-up) |

Rules:

- **Primary endpoint** for the replication should be **Tier B (ventricular arrhythmia / cardiac arrest)** where
  available, because it directly matches the endpoints used in the paper's own external-validation cohorts and is the
  most defensible public proxy for the *mechanism* (arrhythmic death).
- Use **Tier A** as the primary endpoint only in a dataset with true death-registry cause coding (UK Biobank).
- Treat **Tier D (all-cause mortality)** as a secondary/robustness endpoint, never the headline, because it dilutes the
  arrhythmic signal the biomarker is supposed to capture.
- Always model as **time-to-event with censoring**, not naive fixed-window binary, so that variable follow-up across
  public datasets is handled correctly (see Section 5.3).

Document, per dataset: exact code lists, look-back window for exclusions, follow-up start (ECG date) and end
(event/death/censoring date), and competing-risk handling (non-arrhythmic death competes with arrhythmic death).

---

## 3. Data

### 3.1 Recommended public / research-accessible datasets

| Dataset | Approx size | Leads / format | Outcome linkage | Role in this spec | Access |
|---------|-------------|----------------|-----------------|-------------------|--------|
| MIMIC-IV-ECG | ~800k 12-lead, 10s, 500 Hz | 12-lead WFDB | Link to MIMIC-IV hospital/ICU: deaths, VT/VF, arrest codes | **Primary discovery cohort** (Tier B + D) | PhysioNet credentialed |
| UK Biobank ECG | ~80k 12-lead (imaging visit) + exercise ECG | 12-lead | Death registry (cause-coded) + HES hospital records | **Primary for Tier A**, external validation | UKB application |
| CODE / Code-15% | ~2.3M (full) / 345k (open 15%) | 12-lead, 7-10s, 300-600 Hz | Mortality follow-up (full CODE); 15% subset has limited labels | **Scale pretraining / external validation** | Open (15%) / DUA (full) |
| PTB-XL | ~21.8k 12-lead, 10s, 500/100 Hz | 12-lead WFDB | Diagnostic statements; no long-term mortality | **Delineation/QC dev, label-free pretraining** | Open (PhysioNet) |
| Sami-Trop / other CODE-derived | ~1.6k+ | 12-lead | Chagas mortality follow-up | Robustness / transfer | Open |

Notes:

- **MIMIC-IV-ECG + MIMIC-IV** is the recommended **primary discovery cohort** because it is fully credentialed-open,
  large, and richly linked to in-hospital arrhythmic events and death disposition.
- **UK Biobank** is the recommended **primary external-validation cohort** because it offers death-registry
  cause-of-death coding (Tier A) closest to the original SCD definition, plus an LVEF-like measure (CMR-derived
  ejection fraction in the imaging sub-cohort) enabling the LVEF-comparison analyses (T3).
- **CODE** provides scale for self-supervised pretraining and a third-population external check.
- No single public dataset matches Sweden's population completeness; the **multi-dataset** design is how we approximate
  the original's internal + external structure (discovery on MIMIC-IV, external transfer to UKB and CODE).

### 3.2 Canonical internal data schema

All datasets are harmonized into one internal representation before modeling.

`waveforms/` (one file per ECG):

- `signal`: float32 array, shape `[12, T]`, ordered leads `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`.
- Stored at a canonical **sampling rate of 500 Hz** and canonical **duration of 10 s** (`T = 5000`); shorter/longer
  records are resampled/padded/cropped (Section 4).
- Units: **millivolts**, calibration-corrected; per-lead.

`records.parquet` (one row per ECG):

| Column | Type | Description |
|--------|------|-------------|
| `ecg_id` | str | Globally unique ECG id (`{dataset}:{native_id}`) |
| `patient_id` | str | Globally unique subject id (`{dataset}:{native_subject}`) |
| `dataset` | enum | Source dataset |
| `acquired_at` | timestamp | ECG acquisition datetime (follow-up t0) |
| `age`, `sex` | num/enum | Demographics at acquisition |
| `sampling_rate_native`, `duration_native_s` | num | Provenance for QC |
| `lvef` | float / null | Ejection fraction if available (echo/CMR), with `lvef_source`, `lvef_offset_days` |
| `qc_pass` | bool | Passed signal-quality gate (Section 4.4) |
| `split` | enum | `train` / `val` / `test` / `external` (patient-grouped, temporal) |

`outcomes.parquet` (one row per ECG, time-to-event):

| Column | Type | Description |
|--------|------|-------------|
| `ecg_id` | str | FK |
| `endpoint_tier` | enum | A / B / C / D actually used for this row |
| `event` | int | 1 if endpoint event observed, 0 if censored |
| `time_days` | float | Days from `acquired_at` to event or censoring |
| `competing_death` | int | 1 if non-endpoint death occurred (for competing-risk models) |
| `endpoint_codes` | list | The specific ICD/procedure codes that triggered the event |

### 3.3 Cohort construction

- **Unit of analysis:** the ECG, but **all splits are grouped by `patient_id`** so a patient never appears in more than
  one split (prevents leakage). Most analyses also report a **one-ECG-per-patient** sensitivity cohort (first
  qualifying ECG).
- **Inclusion:** standard resting 12-lead ECG; age >= 18; passes QC; non-missing acquisition time; at least a minimum
  follow-up window available (e.g., censoring date known).
- **Exclusion:** paced rhythm (pacemaker spikes) and pre-existing ICD at ECG time (these alter QRS morphology and
  confound the terminal-QRS biomarker) — flag with a `paced`/`prior_device` indicator and exclude from primary,
  include in a sensitivity analysis. Exclude uninterpretable/lead-reversed records failing QC.
- **Index handling:** if an arrhythmic event already occurred before the ECG, treat per analysis plan (typically the
  ECG nearest before first event for a "prediction" framing; document precisely).

### 3.4 Splits and validation structure

- **Discovery cohort (MIMIC-IV):** patient-grouped split into train / val / test (e.g., 70/10/20). Add a **temporal
  holdout** (most recent N months by `acquired_at`) to test distribution shift over time.
- **External cohorts (UKB, CODE):** never used for training or model selection; only for T4-T6 confirmation.
- **Pretraining (optional, label-free):** PTB-XL + Code-15% + CODE waveforms for self-supervised pretraining; ensure no
  patient overlap with any evaluation set.

---

## 4. Signal preprocessing

Implement as a deterministic, versioned pipeline (config-hashed) so every experiment records exactly which transform
produced its inputs.

1. **Lead ordering & polarity:** enforce canonical 12-lead order; detect and correct known lead reversals via QC
   heuristics (e.g., inverted lead I/aVR checks).
2. **Resampling:** polyphase resample to **500 Hz**.
3. **Duration normalization:** center-crop or symmetric zero-pad to **10 s (5000 samples)**. Record padded mask.
4. **Filtering:**
   - Baseline-wander removal: high-pass at ~0.5 Hz (zero-phase, forward-backward) **or** median-baseline subtraction.
   - Powerline: notch at 50 Hz (Europe/UKB) and/or 60 Hz (US/MIMIC) per dataset; record which.
   - Anti-alias low-pass at ~150 Hz before any downsampling.
   - **Crucial caveat for the biomarker:** the terminal-QRS *smoothness* feature is sensitive to low-pass filtering and
     resampling (both smooth the signal). The biomarker extraction in Phase 2 must run on a **filtering profile that is
     fixed and reported**, and the spec mandates a **filter-sensitivity analysis** (Section 6.6). Prefer a *minimal,
     consistent* filter for the biomarker (e.g., baseline removal + powerline notch only, no aggressive low-pass) so
     "smoothness" reflects physiology, not signal processing.
5. **Normalization (for the neural net only):** per-lead robust scaling (median/IQR) computed on train; store scaler.
   The **biomarker computations use physical mV units, not the net's normalized units.**
6. **QC gate (`qc_pass`):** reject records with flatline leads, extreme amplitude, high-frequency noise above
   threshold, excessive saturation/clipping, or failed R-peak detection. Log rejection reasons.

All preprocessing parameters live in a single `preprocess.yaml`; its hash is stored on every produced artifact.

---

## 5. Phase 1 — Deep predictive model

### 5.1 Input representation

- Primary: raw normalized waveform tensor `[12, 5000]`.
- Optional auxiliary tabular inputs (age, sex) concatenated at the head — but the **headline model must be able to run
  waveform-only**, because the scientific claim is that signal *morphology* carries the risk. Provide both
  `waveform-only` and `waveform+demographics` variants and compare.

### 5.2 Architecture

Default and alternatives (multi-GPU cluster assumed, so larger encoders are in scope):

- **Default encoder: 1D residual CNN** (ResNet-style, ~8-16 residual blocks) with squeeze-and-excitation, designed for
  12-lead ECG. Proven, stable, strong baseline; comparable to published ECG deep nets. Recommended starting point.
- **Alternative encoders (ablation):**
  - 1D CNN + Transformer hybrid (CNN stem -> transformer encoder over temporal tokens) for longer-range dependencies.
  - ECG **foundation-model** encoder pretrained self-supervised (masked-waveform or contrastive) on the pretraining
    pool, then fine-tuned — exploits the multi-GPU budget and the large unlabeled corpora.
- **Head:** small MLP producing the survival/risk output (Section 5.3).
- **Design constraint for interpretability:** keep the input as **raw per-lead time series** (no cross-lead mixing
  before the first conv) so that gradient/attribution maps in Phase 2 localize cleanly to **specific leads and
  time-within-beat** (needed to rediscover "lead aVL, terminal QRS").

### 5.3 Learning objective

- Frame as **time-to-event survival** with right-censoring and **competing risks** (non-arrhythmic death competes with
  the arrhythmic endpoint). Recommended approaches, in order:
  1. **Discrete-time survival** (model hazard over a set of time bins; negative log-likelihood with censoring). Simple,
     stable, gives a full risk curve; pairs naturally with a neural net.
  2. **DeepSurv-style Cox** partial-likelihood head (proportional hazards) as a comparator.
  3. **Cause-specific / Fine-Gray** competing-risk formulation for the primary analysis of arrhythmic events.
- Also train a **fixed-horizon binary** head (e.g., 1-year, 5-year event) as an interpretable secondary output and to
  ease the high-risk-bucket analysis (T2/T3).
- **Class imbalance:** SCD/arrhythmic events are rare. Use the survival likelihood (handles imbalance naturally) plus
  optional risk-set weighting; avoid naive oversampling that distorts calibration.

### 5.4 Training procedure (multi-GPU)

- **Framework:** PyTorch + PyTorch Lightning (or plain DDP). `torchrun` for distributed launch.
- **Parallelism:** Distributed Data Parallel across GPUs; mixed precision (bf16/fp16) with grad scaling. Optional
  gradient accumulation for large effective batch. FSDP only if a foundation-model encoder is large enough to need it.
- **Batch / optimizer:** AdamW; cosine decay with warmup; effective batch sized to the cluster. Weight decay tuned.
- **Regularization:** dropout in head, stochastic depth in residual blocks, signal augmentation (below), early stopping
  on validation time-dependent C-index.
- **Augmentation (train only, physiologically safe):** small time shifts, amplitude scaling, lead-wise dropout, Gaussian
  noise, baseline-wander injection, random crop within the 10 s. **Do not** use augmentations that destroy
  terminal-QRS morphology if the same checkpoint will be used for biomarker work; keep an "augmentation-light" variant.
- **Reproducibility:** fixed seeds, deterministic dataloading where feasible, full config + git SHA + data hash logged.

### 5.5 Calibration

- Post-hoc calibrate risk outputs (isotonic or temperature scaling on validation).
- Report calibration curves and Integrated Brier Score across the follow-up horizon.

### 5.6 Phase 1 evaluation

Discrimination and clinical-stratification metrics on the held-out test and temporal-holdout sets:

- **Time-dependent AUROC / C-index** (Harrell and Uno), at clinically meaningful horizons (1, 3, 5 years).
- **AUPRC** at fixed horizons (rare-event sensitive).
- **Calibration:** calibration plots, ICI, Integrated Brier Score.
- **Risk concentration (T2):** define the top **~2.2%** model-risk bucket (mirroring the paper) and report its
  annualized event rate vs the cohort base rate and vs an LVEF-defined group.
- **LVEF comparison (T3):** in the subset with LVEF available, construct the clinical comparator group (reduced LVEF,
  e.g., <= 35%), compute its size and annualized event rate, and compute the **overlap**: fraction of model-high-risk
  patients who are *not* in the reduced-LVEF group (target: qualitatively reproduce the ~86% "missed by LVEF").
- **Decision-curve analysis** and **net reclassification** vs the LVEF baseline.
- **Subgroup fairness:** metrics stratified by sex, age band, and dataset/site; report gaps.

Baselines to beat: (a) age+sex only, (b) LVEF only, (c) age+sex+LVEF, (d) classical ECG features (QRS duration, QTc,
etc.). The deep model must beat these to justify Phase 2.

---

## 6. Phase 2 — Biomarker distillation (the core scientific replication)

Goal: convert the trained black-box into the **interpretable, hand-computable biomarker** — ideally rediscovering
**aVL terminal-QRS smoothness** without hard-coding it, then validating it stands on its own.

This phase runs as a funnel: localize -> hypothesize features -> distill -> confirm independence.

### 6.1 Step A — Attribution / localization

Apply multiple complementary attribution methods to the trained model over test ECGs and **aggregate**:

- Gradient-based: Integrated Gradients, SmoothGrad, Saliency.
- Activation-based: 1D Grad-CAM over conv feature maps.
- Perturbation-based: lead-occlusion (zero each lead, measure risk-score change) and **segment-occlusion** (mask P,
  QRS, ST-T windows using the delineation from Section 6.2) to find *which lead* and *which part of the beat* drives the
  prediction.

Aggregate attributions into a **lead x beat-phase importance map** across many ECGs. **Success signal:** importance
concentrates on **lead aVL** and the **terminal portion of the QRS**, matching the paper. If it concentrates elsewhere,
that is a finding too — record it and proceed with the data-driven location rather than forcing aVL.

### 6.2 Step B — Beat delineation

- Run a validated QRS detector + wave delineator (e.g., a NeuroKit-style pipeline and/or a learned delineator) to mark,
  per beat per lead: P onset/offset, **QRS onset, R peak, QRS offset (J point)**, T onset/offset.
- Derive a **representative/median beat** per lead (align on R peak, average) to suppress noise before computing
  morphology features.
- Validate delineation quality on PTB-XL (which has good signal quality) and on a manually checked sample.

### 6.3 Step C — Candidate interpretable features

Compute a **library** of hand-defined morphology features per lead from the median beat, explicitly including the
paper's biomarker and plausible neighbors so the distillation can select among them:

- **Terminal-QRS smoothness (the target biomarker):** over the window from **R peak to QRS offset**,
  - `mad1 = mean(|first difference of voltage|)` (mean absolute first difference),
  - `mad2 = mean(|second difference of voltage|)` (mean absolute second difference).
  Lower values = smoother terminal R-wave region. Compute for **all 12 leads** (so aVL emerges by selection, not
  assumption). Report sampling-rate-normalized versions.
- Related/confounder features for fair competition: QRS duration, R-wave amplitude, terminal QRS slope, QRS area,
  fragmentation/notching indices, QT/QTc, ST-segment features, T-wave morphology, frontal-plane axis.
- Provide each feature's exact mathematical definition and a reference implementation.

### 6.4 Step D — Surrogate / distillation modeling

- Fit **interpretable surrogate models** (sparse logistic / Cox with L1, shallow gradient-boosted trees, generalized
  additive models) that predict (i) the **deep model's risk score** (knowledge distillation) and (ii) the **actual
  endpoint**, using only the candidate-feature library.
- Use **sparse selection** (L1, stability selection) and report which features survive. **Target result (T4/T5):**
  `mad1`/`mad2` in **aVL** are selected and carry large, stable weight, and a tiny feature set recovers a large share
  of the deep model's discrimination.
- Quantify "how much of the black box is explained": compare surrogate C-index to deep-model C-index (fidelity), and
  report the share of deep-model risk variance explained by the smoothness feature alone.

### 6.5 Step E — Standalone biomarker validation

- Treat the single distilled biomarker (aVL terminal-QRS smoothness) as a **univariable predictor**: estimate hazard
  ratio per SD (adjusted for age/sex, and additionally for LVEF) in the test set.
- Confirm **direction** (greater smoothness -> higher risk) matches the paper.
- Report incremental value over LVEF and over classical ECG features (likelihood-ratio test, delta C-index, NRI).

### 6.6 Step F — Robustness of the biomarker

- **Filter-sensitivity analysis:** recompute the biomarker under several preprocessing profiles (minimal vs aggressive
  low-pass, different resampling) and show the predictive association is not an artifact of smoothing.
- **Sampling-rate sensitivity:** show stability across 250/500/1000 Hz where native data allows.
- **Lead-specificity check:** confirm aVL (and the frontal-plane neighbors aVR/aVF/I) carry the signal more than
  precordial leads, as a physiological plausibility check.
- **Beat-selection sensitivity:** median beat vs single-beat vs trimmed-mean.

Deliverable of Phase 2: a one-paragraph, copy-pasteable definition of the biomarker plus a <50-line reference function
that takes a 12-lead ECG and returns the score — the "hand-computable" artifact that is the point of the paper.

---

## 7. Phase 3 — External validation and transfer

- Freeze the deep model and the biomarker function from the discovery cohort.
- **Apply unchanged** to each external cohort (UKB, CODE), mapping each cohort to its best-supported endpoint tier
  (UKB -> Tier A cause-coded sudden/arrhythmic death; CODE -> available mortality). This mirrors the paper's design of
  validating against *related but non-identical* endpoints in different populations.
- Report per cohort: time-dependent C-index, biomarker hazard ratio and direction, calibration (expect drift; report
  recalibration-in-the-large), and the LVEF-complementarity analysis where LVEF/EF is available (UKB CMR).
- **Transfer success (T6):** biomarker effect direction preserved and discrimination above the demographic baseline in
  at least one fully external cohort.
- Domain-shift handling: report results both raw and after simple recalibration; do **not** refit the biomarker.

---

## 8. Phase 4 — Clinical-utility analysis (stretch / explicitly caveated)

The paper's ICD mortality-benefit finding (high-risk + ICD -> ~54.4% lower mortality than expected) requires treatment
data and careful causal design. On public data this is **out of scope as a primary deliverable** and only attempted if
a dataset records device therapy (some MIMIC-IV / UKB HES records do).

If attempted:

- Use a **target-trial emulation** framing (eligibility, treatment assignment = ICD, outcome = mortality), with
  propensity/overlap weighting and sensitivity to unmeasured confounding (E-value).
- Heavily caveat: observational ICD benefit is confounded by indication; this can only be a **directional, hypothesis-
  consistent** check, not a causal claim. State this prominently.

---

## 9. Statistical analysis plan (pre-registered)

- Pre-register endpoints, primary cohort, primary model, and the six replication targets (T1-T6) **before** looking at
  test results. Keep a frozen analysis-plan file under version control.
- **Primary analysis:** time-dependent C-index of the deep model and HR-per-SD of the distilled biomarker for the
  primary endpoint on the discovery test set.
- **Confidence intervals:** bootstrap (patient-clustered) for all discrimination metrics and HRs.
- **Multiplicity:** control for multiple comparisons across the feature library in Phase 2 (FDR).
- **Missing data:** report LVEF availability; LVEF analyses are restricted to the LVEF-available subset and flagged as
  such (selection bias acknowledged).
- **Competing risks:** primary arrhythmic analyses use cause-specific or Fine-Gray models; all-cause mortality is
  secondary.
- **Negative controls:** confirm the biomarker is *not* equally predictive of an implausible endpoint (e.g., trauma
  death), as a specificity check.

---

## 10. Reproducibility, tooling, and MLOps

- **Repo layout (proposed):**
  - `data/` — dataset adapters (one module per source -> canonical schema), code lists, QC.
  - `preprocess/` — deterministic signal pipeline, `preprocess.yaml`.
  - `models/` — encoders, survival heads, Lightning modules.
  - `train/` — DDP training entrypoints, configs (Hydra), schedulers.
  - `interpret/` — attribution, delineation, feature library, surrogate distillation.
  - `eval/` — metrics, calibration, decision curves, subgroup, bootstrap.
  - `biomarker/` — the frozen <50-line reference function + tests.
  - `analysis/` — frozen statistical-analysis-plan notebooks/scripts.
- **Config:** Hydra; every run logs config + git SHA + data-hash + preprocess-hash.
- **Experiment tracking:** MLflow or Weights & Biases; log metrics, calibration plots, attribution maps, feature
  selections.
- **Data versioning:** DVC or lakeFS for the canonical waveform/parquet artifacts.
- **Environment:** pinned `requirements.txt`/`conda` lock + Dockerfile; CUDA/cuDNN versions recorded.
- **Testing:** unit tests on the biomarker function (synthetic beats with known smoothness), delineation accuracy
  tests, schema-validation tests on each dataset adapter, and an end-to-end smoke test on a tiny fixture.
- **Determinism:** seed control; document residual nondeterminism from cuDNN.

---

## 11. Compute plan (multi-GPU cluster)

- **Pretraining (optional foundation encoder):** the largest job; self-supervised on the full unlabeled pool. Multi-node
  DDP/FSDP; bf16. Budget the bulk of GPU-hours here if this path is chosen.
- **Supervised training:** single-node multi-GPU DDP is typically sufficient for the 1D-CNN default; sweep
  hyperparameters in parallel across GPUs/nodes (one trial per GPU).
- **Interpretation & distillation:** attribution over the test set is embarrassingly parallel (shard ECGs across GPUs);
  surrogate fitting is CPU-light.
- **Evaluation/bootstrap:** CPU-parallel.
- Provide a Slurm (or equivalent) submission template per stage with explicit GPU/memory/time requests.

---

## 12. Milestones and deliverables

| Phase | Milestone | Key deliverables | Exit criteria |
|-------|-----------|------------------|---------------|
| M0 | Data foundation | Dataset adapters, canonical schema, QC report, code lists, frozen splits | All datasets harmonized; QC + leakage checks pass |
| M1 | Baselines | LVEF / demographic / classical-ECG baselines + metrics harness | Baselines reproducible with CIs |
| M2 | Deep model | Trained survival model, calibration, T1-T3 results | Beats baselines on test + temporal holdout |
| M3 | Biomarker distillation | Attribution maps, feature library, surrogate, frozen biomarker function | T4-T5 met; aVL terminal-QRS smoothness (or data-driven equivalent) characterized |
| M4 | External validation | UKB + CODE results, transfer report | T6 met on >=1 external cohort |
| M5 | (Stretch) clinical utility | Target-trial emulation (if data permits) | Directional, fully caveated result |
| M6 | Write-up | Replication report, reproducibility package, model+biomarker cards | All targets reported (met or not), pipeline reproducible end-to-end |

---

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No true SCD label in public data | Endpoint differs from paper | Tiered endpoint strategy (Section 2); use Tier B arrhythmic events matching the paper's own validation endpoints; UKB Tier A for cause-coded death |
| Population-completeness gap vs Swedish registry | Selection bias (sicker MIMIC ICU population) | Report cohort characteristics; multi-dataset external validation; one-ECG-per-patient sensitivity cohort |
| Smoothness is a filtering artifact | False biomarker | Mandatory filter/sampling-rate sensitivity analyses (6.6); compute biomarker in physical units on a minimal, fixed filter |
| Attribution doesn't localize to aVL/terminal QRS | Can't rediscover biomarker cleanly | Multi-method attribution + segment occlusion; if location differs, report the data-driven location honestly |
| LVEF sparsely available | Weak T3 comparison | Restrict LVEF analyses to LVEF-available subset; use UKB CMR EF; flag selection bias |
| Rare events / low power | Unstable estimates | Survival framing, patient-clustered bootstrap, pool datasets for power, pre-register |
| Pacemaker/ICD-altered QRS | Confounds biomarker | Exclude paced/prior-device from primary; sensitivity inclusion |
| Label/leakage from device or arrest already present at ECG time | Optimistic results | Strict index-event handling; exclude events at/just-before ECG; temporal holdout |
| Overclaiming clinical benefit | Ethical/scientific | Phase 4 explicitly stretch + caveated; no causal claims from observational ICD data |

---

## 14. Ethics, governance, and limitations

- **Data access & DUAs:** PhysioNet credentialing (MIMIC, PTB-XL), UK Biobank application, CODE data-use agreements.
  Respect each dataset's license and re-identification prohibitions. Do not attempt linkage across datasets at the
  individual level.
- **No clinical deployment** from this replication; outputs are research artifacts. Any clinical use would require
  prospective validation and regulatory clearance.
- **Bias & equity:** report subgroup performance; ECG-mortality models can encode site/demographic confounders.
- **Honest reporting:** publish targets that fail, not only those that pass; the value is methodological replication,
  including documenting where public data cannot reproduce the original.
- **Key limitations to state up front:** endpoint substitution (arrhythmic events vs cause-coded SCD), cohort
  selection (ICU-heavy MIMIC), LVEF availability, and absence of a population-complete registry.

---

## 15. Open questions for implementation kickoff

- Confirm primary endpoint per dataset (recommend MIMIC Tier B as discovery primary; UKB Tier A as external primary).
- Confirm whether the optional self-supervised foundation-encoder path is in budget, or stay with the 1D-CNN default.
- Confirm access timelines for UK Biobank / full CODE (gate M4).
- Confirm whether Phase 4 (clinical utility) is attempted at all given public-data constraints.

---

## 16. References / sources

- Obermeyer, Z., Schubert, A., Ross, J. et al. *An ECG biomarker for sudden cardiac death discovered with deep
  learning.* Nature (2026). `s41586-026-10674-6`.
  [Nature article](https://www.nature.com/articles/s41586-026-10674-6)
- News & Views: *A hidden predictor of sudden cardiac death uncovered by deep learning.* Nature (2026).
  [link](https://www.nature.com/articles/d41586-026-01806-z)
- Press: [EurekAlert](https://www.eurekalert.org/news-releases/1133419),
  [MedicalXpress](https://medicalxpress.com/news/2026-06-ai-sudden-cardiac-death-thousands.html),
  [Bioengineer.org](https://bioengineer.org/deep-learning-reveals-ecg-sudden-death-marker/)

> Note on sources: the Nature article and several press pages are paywalled or block automated fetching; the
> capability summary in Sections 1-2 is reconstructed from the abstract and press coverage. Before implementation,
> the team should read the full paper and Methods to confirm exact cohort sizes, the network architecture, the precise
> biomarker definition (window boundaries and units), and the validation-endpoint code lists, then reconcile any
> differences against this spec.
