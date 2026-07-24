# Semantic3D Project Instructions

Read this file before changing the repository. Then read:

1. `docs/SERVER_HANDOFF_P4C3B_M2.md`
2. `configs/handoff/p4c3b_m2_server_handoff_v1.yaml`
3. the latest entries in `docs/DEV_LOG.md`

Actual source, configurations, artifacts, and tests take precedence over prose.
Do not claim a feature is implemented or executed from a plan or log alone.

## Current Verified Stage

P4-C3B-M2 is complete locally:

- UniDepthV2 provides model-predicted metric depth and model-predicted
  intrinsics. These are not sensor ground truth or calibrated camera metadata.
- M2 reconstructs only a single-frame, camera-coordinate metric visible-surface
  2.5D structure.
- New M2 geometry uses `coordinate_frame=camera_frame_metric`.
- `world_frame_reconstruction_complete=false`.
- Generic internal points are `geometric_track_point` candidates, not semantic
  keypoints and not yet verified across frames.
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
