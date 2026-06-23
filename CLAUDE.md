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
- Data may still live in S3 (and AWS Athena is fine as an optional query layer), but orchestration and
  lineage come from Dagster, not dbt.

### Working agreement

- **Specs only — no code** until the user explicitly says to start coding.
- Cite public open-source models with reported accuracy and production/clinical-maturity (P0–P3).
- Verify per-repo LICENSE before assuming any model is usable for production (e.g. HuBERT-ECG is CC BY-NC).
