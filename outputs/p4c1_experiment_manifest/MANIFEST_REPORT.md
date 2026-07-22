# P4-C1 Experiment Manifest Report

This report freezes sample identity, split assignment, input availability, and leakage audit only.
No model training, parameter fitting, threshold selection, or authenticity performance metric was run.

## Reproducibility

- manifest schema: `p4c1_experiment_manifest_v1`
- protocol SHA-256: `004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12`
- P4-C0 config SHA-256: `8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304`
- P4-C1 config SHA-256: `ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086`
- source label manifest SHA-256: `4d811340ed2ba69c80759a0ef025931e51dfc7dd73fa0a9f1ea456855fd2e633`
- manifest SHA-256: `3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`
- structural dataset ID: `dataset_1ae499fd7cdd9d8501dc`

## Split Freeze

| split | samples | usable | excluded | videos | groups |
|---|---:|---:|---:|---:|---:|
| train | 0 | 0 | 0 | 0 | 0 |
| validation | 59 | 50 | 9 | 6 | 6 |
| test | 0 | 0 | 0 | 0 | 0 |
| official_conflict | 0 | 0 | 0 | 0 | 0 |

## Data Availability

| modality | available | missing | required |
|---|---:|---:|---|
| video | 59 | 0 | true |
| frames | 59 | 0 | true |
| objects | 50 | 9 | true |
| depth | 59 | 0 | true |
| camera | 59 | 0 | true |
| pose | 0 | 59 | false |
| tracks | 44 | 15 | false |
| semantic3d | 45 | 14 | false |
| camera_identity | 0 | 59 | false |

## Exclusions

| reason | samples |
|---|---:|
| required_objects_unavailable | 9 |

## Leakage Audit

- unique source videos: 6
- unique source groups: 6
- error findings: 0
- warning findings: 6
- unresolved original/derivative identities: 6
- cross-split leakage detected: false

Unresolved source identity is retained as a provenance warning. It is not silently treated as proof of independence.
Overlapping clips inherit the source-video split and are not counted as independent source videos.
