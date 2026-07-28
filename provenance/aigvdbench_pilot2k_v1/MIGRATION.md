# AIGVDBench Pilot-2K Full Residual Migration

This directory records the lightweight, reviewable provenance for the completed
AIGVDBench Paired Pilot-2K residual extraction. The formal run used source commit
`739650c00470f9ed127cdcaa69460abc9d493f5e`; it is not a smoke run.

## What Git stores

Git stores the frozen candidate, local-media, and final-residual manifests; the
formal media and residual audit tables; package and hardware metadata; model
asset checksums; fixed run parameters; and this migration procedure. The copied
CSV/JSON files are audit records only. They do not replace the media or
per-sample residual payloads.

The formal summary contains exactly 2,000 samples, all with
`analysis_ok=True`. `residual_failures.csv` has no data rows. The local-media
manifest also contains exactly 2,000 distinct, complete video records.

## What cloud storage must retain

Retain these directories outside Git:

- `/root/autodl-tmp/datasets/aigvdbench_pilot_v1/videos_v2`
- `/root/autodl-tmp/outputs/paper_core_r5b3/aigvdbench_pilot2k_residuals_v1`
- `/root/autodl-tmp/checkpoints`

The formal residual tree includes per-sample arrays, JSON results, sampling
provenance, and visual artifacts. None of those payloads are copied into Git.
The 2,000 videos, model weights, virtual environment, and caches are likewise
excluded from Git.

Do not migrate `.venv` or `/root/autodl-tmp/caches`. Recreate the environment
from a clean Python 3.12.3 virtual environment and
`environment/pip_freeze.txt`; allow caches to be rebuilt.

## New-server layout

Restore the following paths:

```text
/root/autodl-tmp/projects/fake_video_structural_anomaly_refactor
/root/autodl-tmp/projects/third_party/UniDepth
/root/autodl-tmp/datasets/aigvdbench_pilot_v1
/root/autodl-tmp/checkpoints
/root/autodl-tmp/caches
/root/autodl-tmp/outputs/paper_core_r5b3/aigvdbench_pilot2k_residuals_v1
```

Checkout UniDepth at
`8d8cfe4c7ee15297099983607febf0d4f32eb3d6` and require a clean worktree.
Source `server_env_template.sh` only after reviewing the restored paths.

## Integrity checks

Verify the copied lightweight records from this directory:

```bash
cd provenance/aigvdbench_pilot2k_v1
sha256sum -c checksums.sha256
```

Verify every restored video against `source_video_sha256` in
`manifests/local_media_manifest.csv`. Verify every residual against
`residual_sha256` in
`manifests/aigvdbench_open_sora_paired_2k_residual_v1.csv`. Verify model files
by filename, byte size, and SHA-256 using `environment/model_weights.csv`.

The frozen identities are:

- candidate manifest:
  `688c8d7a1995bef1ca23ba42d1072811a7a1be7cae5690cb49773fb598b5dfe9`
- local media manifest:
  `d376d30b1e4fdce7c3e06b2d456f264cc4669bb8ab1f82bb9d21765b019a2d88`
- final residual manifest:
  `93221458f9675b53849a2b66a0574347fe47d26b3420c2fba8e69f72f93bdda8`
- Full-2K source config:
  `c76ce5268887a3224e7ebd13fc1e5a44325f265148f84770774bf0bf7ff0bf80`

## Frozen Full-2K residual command

After restoring the media, model assets, pinned UniDepth source, project
environment, and path template, the frozen resumable command is:

```bash
source provenance/aigvdbench_pilot2k_v1/server_env_template.sh

PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/run_extract_residuals.sh" \
  --manifest "${DATA_ROOT}/aigvdbench_pilot_v1/media_inventory/full2k/local_media_manifest.csv" \
  --config "${PROJECT_ROOT}/configs/default.yaml" \
  --output "${OUTPUT_ROOT}/paper_core_r5b3/aigvdbench_pilot2k_residuals_v1" \
  --sampling-mode uniform_full_video \
  --sampled-frames 32 \
  --clip-length 8 \
  --clip-count 4 \
  --device cuda \
  --resume \
  --log-every 50 \
  --workers 1
```

The command is label-blind during extraction. Do not regenerate the completed
formal output merely to validate migration; first run checksum and row-count
checks against the restored assets.
