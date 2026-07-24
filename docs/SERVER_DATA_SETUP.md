# Server Data Setup

Formal datasets have not been selected or registered. Dataset records must
follow `configs/data_registry/dataset_registry_schema_v1.yaml` and include
source lineage, license, citation, official split, acquisition method, and
checksums before formal construction is considered.

## Six-Video Geometry Smoke

The current six videos are registered in
`configs/data_registry/local_six_video_smoke_v1.yaml` as
`geometry_validation_smoke`. They are not eligible for training, model or
threshold selection, or final evaluation. Their redistribution permission and
original source identity are unverified, so the video files must not be pushed
to Git.

Place them manually under `${DATA_ROOT}/tests_videos/` using the registered
relative paths, then verify every SHA-256 before smoke testing. A missing video
is missing smoke evidence, not a normal anomaly score of zero.

## Formal Data

Do not download or build formal data until P4-C2 reports
`ready_for_formal_batch_build=true`, storage checks pass, and source lineage is
verified. Provider failures must remain quality/missingness records and must not
be converted into authenticity evidence.

## P4-C3B Local-Only Artifacts

Direct reproduction of the M2 smoke additionally needs:

- `outputs/p4c3b_metric_provider_smoke/`;
- `outputs/structural_enhancement_dataset/p4b5_six_video_full_observation/`.

These directories are intentionally excluded from Git. Transfer them with an
authorized file channel and validate their sentinel SHA-256 values, or rebuild
them from registered videos, weights, and frozen configs. See
`docs/SERVER_HANDOFF_P4C3B_M2.md`.

Missing arrays must be reported as `blocked_by_input`; they are not an M2
residual of zero and are not evidence that a provider or method is defective.
