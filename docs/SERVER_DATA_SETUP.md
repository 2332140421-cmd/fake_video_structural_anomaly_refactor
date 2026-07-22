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
