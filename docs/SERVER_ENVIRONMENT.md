# Server Environment

The server obtains source code with `git clone` or `git pull`. Do not copy the
local `.venv`, caches, outputs, model binaries, or videos into Git.

## Required Variables

Set these paths before running the bootstrap script:

```bash
export PROJECT_ROOT="$PWD"
export DATA_ROOT="/srv/semantic3d/data"
export DOWNLOAD_ROOT="/srv/semantic3d/downloads"
export CACHE_ROOT="/srv/semantic3d/cache"
export MODEL_ROOT="/srv/semantic3d/models"
export OUTPUT_ROOT="/srv/semantic3d/outputs"
export LOG_ROOT="/srv/semantic3d/logs"
```

The temporary root is `${CACHE_ROOT}/tmp`. Location-specific values are runtime
metadata and are excluded from canonical semantic hashes.

## Dependencies

- Python: 3.11 through 3.13.
- System packages: `ffmpeg` and `ffprobe`.
- Base/test Python packages: `requirements-lock.txt`.
- Inference packages: `requirements-inference-lock.txt`.
- PyTorch/CUDA: install separately for the server's driver and CUDA capability.
  The local Torch 2.12.1 CUDA 13.0 build is a verification record, not a server
  requirement.

The bootstrap script intentionally stops when PyTorch is absent unless
`P4_TORCH_INSTALL_COMMAND` is explicitly supplied. Review that command for the
server before execution.

## Bootstrap

```bash
bash scripts/bootstrap_p4_server.sh
```

The script does not download formal data, train a model, modify Git history, or
run a formal batch.

## P4-C3B Metric Provider

P4-C3B-M1/M2 adds an external UniDepthV2 provider. Its source and weights are
not committed to Git. Follow `docs/SERVER_HANDOFF_P4C3B_M2.md` to install the
pinned source revision and stage the registered weight/config files. The base
bootstrap verifies source compatibility but does not silently clone UniDepth or
download its weights.

After checkout, run:

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
```

Source readiness does not imply that model files, smoke videos, M1 arrays, or
P4-B.5 formal observations are available.

Historical archive inputs are optional and machine-specific. Set
`SEMANTIC3D_ARCHIVE_ROOT` or `SEMANTIC3D_STRICT_V2_RESULT` explicitly when a
server has those files; repository defaults no longer assume a WSL `/mnt/e`
mount.
