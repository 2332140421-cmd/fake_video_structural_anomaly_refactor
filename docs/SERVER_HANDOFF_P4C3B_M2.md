# Server Handoff: P4-C3B-M2

This document transfers project state, not experimental data or model binaries.
The receiving agent must verify actual source and local files before continuing.

## 1. Honest Current State

P4-C3B-M1 completed a real, offline UniDepthV2 smoke on six frames. The output
is model-predicted metric depth and model-predicted intrinsics, not sensor truth
or calibrated camera metadata.

P4-C3B-M2 completed:

- single-frame metric visible-surface scene points;
- formal-mask object surface point clouds;
- robust per-axis object extents;
- visible-mask metric boundary points;
- internal `geometric_track_point` candidates;
- single-frame metric structure graphs.

All real M2 points use `camera_frame_metric`. The result is a
“single-frame camera-coordinate metric visible-surface 2.5D structure”.
World-frame reconstruction, semantic keypoints, cross-frame D3, training,
threshold selection, and authenticity effectiveness are not complete.

Local M2 smoke result:

- 6/6 frames valid;
- 11/11 object point clouds valid;
- 352 valid boundary points;
- 220 geometric track-point candidates;
- 11 valid single-frame graphs and 1158 valid edges;
- `world_frame_reconstruction_complete=false`;
- `method_effectiveness_established=false`.

## 2. What Git Contains

Git should contain source, configurations, tests, documentation, small frozen
protocol artifacts, model registries, and hashes. It must not contain:

- `.venv/`;
- `checkpoints/`;
- `data/tests_videos/`;
- generated `outputs/`;
- external UniDepth source checkout;
- credentials or local cache directories.

After committing, record the pushed commit:

```bash
git rev-parse HEAD
git status --short
git log -1 --oneline
```

The server must check out that exact commit. Do not write the pre-commit hash
into the handoff configuration.

## 3. Local-Only Files That Need Separate Transfer or Rebuild

| Content | Approximate size | Required for |
|---|---:|---|
| `checkpoints/` | 150MB | YOLO and UniDepth inference |
| `data/tests_videos/` | 16MB | six-video geometry smoke |
| `outputs/p4c3b_metric_provider_smoke/` | 137MB | direct M2 rerun |
| `outputs/structural_enhancement_dataset/p4b5_six_video_full_observation/` | 2.3GB | formal masks/tracks for direct M2 rerun |
| `outputs/p4c3b_metric_scene3d/` | 2.3MB | historical M2 audit comparison |

Do not push these through Git. Transfer with `rsync`/SCP or rebuild them. Verify
the registry and sentinel SHA-256 values in
`configs/handoff/p4c3b_m2_server_handoff_v1.yaml`.

The six videos have unverified redistribution permission. Use a private,
authorized transfer channel.

## 4. Server Checkout Sequence

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd fake_video_structural_anomaly
git checkout <PINNED_COMMIT>
git status --short
```

Set server paths as described in `docs/SERVER_ENVIRONMENT.md`, then create the
environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
# Install a torch/torchvision build matched to the server NVIDIA driver/CUDA.
.venv/bin/python -m pip install -r requirements-inference-lock.txt
.venv/bin/python -m pip install -e . --no-deps
```

UniDepth is an external pinned provider and is not vendored:

```bash
git clone https://github.com/lpiccinelli-eth/UniDepth \
  "${CACHE_ROOT}/providers/UniDepth"
git -C "${CACHE_ROOT}/providers/UniDepth" checkout \
  8d8cfe4c7ee15297099983607febf0d4f32eb3d6
.venv/bin/python -m pip install -e "${CACHE_ROOT}/providers/UniDepth"
```

Review server-compatible Torch/CUDA dependencies before that install. Do not
copy the WSL `.venv`.

Place model files under `${MODEL_ROOT}` and verify SHA-256. The existing smoke
configs reference project-relative `checkpoints/`; use explicit verified
symlinks or copy the registered files there after validation. Never commit the
links or binaries.

## 5. Readiness Verification

Source and frozen semantics only:

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
```

After weights and videos are staged:

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py \
  --model-root "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --require-models \
  --require-videos
```

After M1 and P4-B.5 artifacts are transferred:

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py \
  --model-root "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --require-models \
  --require-videos \
  --require-m2-inputs
```

Then run:

```bash
.venv/bin/python -m pytest \
  tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py -q -rs
.venv/bin/python -m pytest -q -rs
```

Do not rerun M1/M2 until the corresponding readiness field is true.

## 6. Prompt for the First Server Conversation

Paste the following as the first instruction to the server-side coding agent:

```text
项目位于当前仓库根目录。先不要修改代码，也不要开始新阶段。

请依次阅读并遵守：
1. AGENTS.md
2. docs/SERVER_HANDOFF_P4C3B_M2.md
3. configs/handoff/p4c3b_m2_server_handoff_v1.yaml
4. docs/DEV_LOG.md 最后两个阶段记录

然后以实际源码、配置和本机文件为准执行只读验收：

.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
.venv/bin/python -m pytest tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py -q -rs

请汇报：
- 当前 branch、commit、git status；
- frozen hash 是否一致；
- source_checkout_ready；
- model_files_ready；
- smoke_videos_ready；
- m1_artifacts_ready；
- p4b5_artifacts_ready；
- ready_to_reproduce_m1；
- ready_to_reproduce_m2；
- 全部 missing/blocking reasons；
- 专项测试、skip 和 warning。

缺失模型、视频或 outputs 时只标记 blocked_by_input，不得把它解释为算法失败，
不得自动下载、训练、选阈值、评价真假性能或开始下一阶段。等待我给出下一条阶段提示词。
```

## 7. Recommended Next Topic

The next likely topic is view observability and temporal size materialization.
It is only a recommendation, not authorization. A new user-approved stage must
define its exact scope before code changes begin.
