# P4-C2 Formal Data Build Readiness

This stage audits registry, lineage, task eligibility, storage, and missingness only.
It did not download data, run model inference, train, fit a distribution, select a threshold, or read test performance.

## Decision

- `ready_for_formal_batch_build=false`
- registered dataset entries: 1
- registered formal datasets: 0
- registered smoke datasets: 1
- verified formal datasets: 0
- verified lineage records: 0
- unverified lineage records: 6

## Formal Split Plan

- train: 0
- validation: 0
- test: 0

The six local videos remain `geometry_validation_smoke`. Their prior validation-only split is preserved by P4-C1, but it is not promoted into this formal split plan because original-source identity is unverified.

## Task Eligibility States

- eligible: 14
- ineligible: 24
- unknown: 0
- not_applicable: 5
- provider_failed: 17

Every task row separately records declared-role eligibility and formal-experiment eligibility. Provider failure is never treated as anomaly evidence.

## Storage

- audited path: `/mnt/e`
- audited available bytes: 55817076736
- safety margin bytes: 16106127360
- full build required with safety bytes: 319072879645
- full build fits snapshot: false
- batches fitting without archive/reclamation: 1/10

## Blocking Conditions

- `formal_occlusion_and_reappearance_coverage_unavailable`
- `formal_reprojection_D2_coverage_unavailable`
- `formal_reprojection_D3_coverage_unavailable`
- `formal_spatial_localization_annotations_unavailable`
- `formal_temporal_localization_annotations_unavailable`
- `formal_test_split_empty`
- `formal_train_split_empty`
- `formal_validation_split_empty`
- `insufficient_storage_for_planned_formal_build`
- `no_formal_split_assignments`
- `no_verified_formal_dataset_registered`
- `no_verified_original_source_lineage`

## Conditions For The Next Stage

- register an official, licensed formal dataset without using residual results
- verify original_source_id and every real-to-derived relationship
- freeze official or group-aware train/validation/test assignments
- provide task annotations and provider coverage required by the intended experiment
- provide storage for full temporary peak plus safety margin
- rerun P4-C2 validation until all blockers are cleared

## Reproducibility

- P4-C0 protocol SHA-256: `004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12`
- P4-C1 manifest SHA-256: `3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`
- P4-C2 config SHA-256: `fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89`
- P4-C2 readiness manifest SHA-256: `2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981`
