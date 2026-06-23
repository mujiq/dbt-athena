# CLAUDE.md — Project Memory

## ECG pattern-detection work (branch: claude/ecg-pattern-detection-sota-nl2hew)

Spec lives at `docs/ecg/ecg-pattern-detection-sota.md`. Target ML stack: **TensorFlow / Keras**.

### Orchestration decision (do not revisit without the user)

- **Do NOT depend on dbt** for the ECG/ML work. The ECG pipeline must not require dbt.
- **Use Dagster** as the primary open-source orchestrator. Rationale: software-defined **assets** give
  dbt-like data lineage for cohort/label tables, *and* it natively orchestrates the Python TF/Keras
  training/eval pipeline in one tool, with first-class AWS (S3) integrations.
  - Heavy distributed/GPU training at scale: **Flyte** or **Metaflow** are acceptable alternatives.
  - Lightweight Pythonic flows: **Prefect** is the fallback.
- Data may still live in S3, but orchestration and lineage come from Dagster, not dbt.

### Deployment target: OpenShift (on-prem / self-managed, in-cluster GPUs, RHOAI available)

- **Do NOT use AWS Athena** on-prem. Use **Trino** (open-source, the engine behind Athena) over
  S3-compatible storage. On-prem S3 = **OpenShift Data Foundation (NooBaa / Ceph RGW)**.
- Use **RHOAI** for GPU (NVIDIA GPU Operator), model serving (**KServe**/Triton), and workbenches;
  distributed TF training via the Kubeflow **Training Operator** (TFJob).
- **Dagster runs via Helm** with restricted-SCC hardening (rootless, arbitrary UID). It is NOT a
  Red Hat-supported RHOAI component — the supported pipeline engine is **Data Science Pipelines**
  (Kubeflow/Tekton). Default is to keep Dagster; switching to DSP needs the user's call.
- Plan for **restricted/air-gapped egress**: mirror PhysioNet datasets and HuggingFace model weights
  into an internal registry/model store; never bake license-restricted assets (HuBERT-ECG CC BY-NC,
  PhysioNet credentialed data) into images.
- Full assessment in `docs/ecg/ecg-pattern-detection-sota.md` §11.

### Working agreement

- **Specs only — no code** until the user explicitly says to start coding.
- Cite public open-source models with reported accuracy and production/clinical-maturity (P0–P3).
- Verify per-repo LICENSE before assuming any model is usable for production (e.g. HuBERT-ECG is CC BY-NC).
