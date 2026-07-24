# Semantic3D Project Instructions

Read this file before changing the repository. Then read:

1. `docs/SERVER_HANDOFF_P4C3B_M2.md` for the transferred M1/M2 inputs
2. `configs/handoff/p4c3b_m2_server_handoff_v1.yaml`
3. `configs/p4c3b_view_scale_history_v1.yaml`
4. `configs/p4c3b_pose_d2_smoke_v1.yaml`
5. the latest entries in `docs/DEV_LOG.md`

Actual source, configurations, artifacts, and tests take precedence over prose.
Do not claim a feature is implemented or executed from a plan or log alone.

## Current Verified Stage

P4-C3B-M4 is complete locally on top of the transferred M1/M2 artifacts:

- UniDepthV2 provides model-predicted metric depth and model-predicted
  intrinsics. These are not sensor ground truth or calibrated camera metadata.
- M2 reconstructs only a single-frame, camera-coordinate metric visible-surface
  2.5D structure.
- New M2 geometry uses `coordinate_frame=camera_frame_metric`.
- `world_frame_reconstruction_complete=false`.
- Generic internal points are `geometric_track_point` candidates, not semantic
  keypoints and not yet verified across frames.
- M3 models camera FOV/intrinsics provenance, explicit object viewpoint state,
  per-dimension observability, metric single-object execution, and same-track
  size histories.
- No reliable category pose provider was available in the persisted smoke:
  all 11 object viewpoints remained `unknown` and no canonical physical
  dimension produced a valid unary metric residual.
- Camera-axis visible extents are separately named diagnostic measurements.
  They produced 12 valid same-track temporal residuals for one couch track;
  they are not canonical width, height, or length.
- M4 estimates adjacent-frame camera pose using formal-mask foreground
  exclusion, background LK tracks, per-frame model-predicted intrinsics, and
  source model-predicted metric depth PnP.
- The accepted convention is
  `X_target_camera = T_target_from_source @ X_source_camera`.
- `fake_1` supplies two multi-evidence `verified_static` frame pairs;
  `real_1` supplies two `estimated_valid` short-baseline camera-motion pairs.
  The persisted `fake_2` control is `blocked_by_intrinsics`, not assigned an
  identity pose or a high residual.
- D2 real smoke generated valid point, visible-boundary, depth, and object
  reprojection diagnostics. Out-of-frame, occluded, depth-conflicting, and
  missing-correspondence points remain invalid with NaN.
- Adjacent metric camera frames can be composed into a short
  `clip_local_aligned` reference gauge. This is not a long-term world map.
- No D3 temporal residual or method-effectiveness conclusion exists.

The six local videos are geometry-validation smoke data only. They are not
eligible for training, model selection, threshold selection, or final
evaluation.

## Non-Negotiable Semantics

- Missing evidence is NaN plus `valid=false`; never encode missing evidence as
  a normal residual of zero.
- Keep `not_applicable`, `blocked_by_input`, `provider_failed`,
  `interface_only`, `executed_invalid`, and `executed_valid` distinct.
- Provider, pose, tracking, or modality failure is quality/missingness
  evidence, not forged-video evidence.
- Preserve depth representation, unit, definition, coordinate frame,
  intrinsics source, pose source, confidence, uncertainty, and provenance.
- Model-predicted metric depth must never be described as measured metric truth.
- Visible instance masks must never be described as amodal masks.
- Do not call camera-frame points world-frame points.
- Do not use authenticity labels to select evidence clips or tune geometry.
- Do not train, fit distributions, choose thresholds, or report authenticity
  performance unless a later user-approved stage explicitly authorizes it.

## Frozen Inputs

Do not edit or overwrite:

- `configs/scale_priors_strict_v1.yaml`
- `configs/scale_priors_strict_v2.yaml`
- P4-C0/C1/C2 semantic configurations or frozen artifacts
- historical strict-v2 results
- existing D1 evidence
- `outputs/p4c3a_method_completion/d2_synthetic_validation.json`
- prior P4-C3A-MD2 six-video read-only results

Verify frozen SHA-256 values with:

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
```

## Working Rules

- Use the repository `.venv/bin/python`; do not assume Conda or system Python.
- Inspect existing modules before adding abstractions. Reuse shared
  observations, geometry, validity, formal-mask, and method-completion APIs.
- Before coding, report existing implementation locations, reused modules,
  files to change, risks, and operations that will not be performed.
- Run focused tests and then `.venv/bin/python -m pytest -q -rs`.
- Append to `docs/DEV_LOG.md`; never overwrite earlier entries.
- Do not run `git add`, commit, tag, push, reset, clean, or destructive data
  operations unless the user explicitly requests that exact action.
- Do not download models or formal datasets silently.

## Server Continuation Gate

After cloning on a new server, first run the handoff verifier and report its
readiness fields. Do not begin a new method stage merely because source tests
pass. Missing weights, videos, M1 arrays, or P4-B.5 artifacts must remain
explicit blockers until transferred or reproducibly rebuilt.
