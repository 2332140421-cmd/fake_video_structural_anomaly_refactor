# 3D Spatiotemporal Structural Anomaly Detection for Forged Video

## Research goal

This project studies forged-video evidence that violates metric 3D scene structure over time. It is not an RGB appearance classifier: frozen vision providers produce shared observations, and the paper method measures geometric, object-semantic, motion, reprojection, relation, occlusion, and reappearance consistency.

Engineering risk scores are not calibrated probabilities. Missing evidence remains explicitly unavailable and is never replaced with zero.

## Method flow

```text
video → continuous clips → shared 2D/3D observations
      → metric backprojection and camera-motion prediction
      → metric object semantics + D1 motion + D2 reprojection + D3 relations/events
      → missing-aware point/track/object/frame/clip fusion
      → full-video timeline, suspicious clips, objects, tracks, and residual heatmap
```

All active stages exchange in-memory dataclasses. The main path does not serialize intermediate M1–M6 CSV files and does not call historical smoke, audit, handoff, or M6→A2 bridge code.

## Project structure

- `data/`: video decoding, clip splitting, manifest reading, and unified observations.
- `models/`: provider adapters, canonical geometry, final residual branches, fusion, and a small temporal head.
- `inference/`: the single `ForgeryAnalysisPipeline`, result writers, and two-command CLI.
- `experiments/`: minimal 3–5 epoch training, evaluation, and paper ablations.
- `utils/`: configuration, atomic I/O, logging, and visualization.
- `configs/default.yaml`: runtime configuration.
- `configs/scale_priors.yaml`: metric object-size priors.

Historical `src/semantic3d/`, scripts, and stage tests remain temporarily for validation and Git history, but they are outside the new active path.

## Metric object-semantic residual

For each sufficiently visible instance mask, metric Z-depth and intrinsics are used to backproject its visible 2.5D surface. A robust metric extent is compared with a category/dimension interval in `configs/scale_priors.yaml`. The branch records its observable dimension, confidence, depth coverage, and reason when unavailable. Same-track metric size stability is a separate temporal residual.

The old object-pair scale/depth ratio is not enabled by default.

## D1, D2, and D3 residuals

- D1 reuses the verified dynamic reprojection, 3D track continuity, direction consistency, and relative-velocity producers.
- D2 reuses the verified point/boundary/depth reprojection calculation under explicit short-baseline pose.
- D3 reuses pose-aligned object-relation graphs and conservative occlusion/reappearance contracts.

Provider failure, missing pose, absent correspondence, severe occlusion, and insufficient history remain unavailable rather than anomaly evidence.

## Missing-aware fusion

`models/fusion.py` is the only active fusion implementation. It averages only valid residuals with confidence weighting, reports branch coverage and confidence separately, preserves per-object/per-track contributions, joins clips into a complete timeline, and merges adjacent suspicious intervals. Its threshold is an engineering configuration, not a fitted authenticity threshold.

## Environment installation

Use Python 3.11–3.13 and the project virtual environment. No provider downloads occur automatically.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set external locations before execution:

```bash
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/local/model/files
export CHECKPOINT_ROOT=/path/to/checkpoints
export HF_HOME=/path/to/huggingface/cache
export TORCH_HOME=/path/to/torch/cache
```

Local YOLO, YOLO-seg, and UniDepth assets must match the paths and registered UniDepth SHA-256 in `configs/default.yaml`.

## Single-video inference

```bash
python -m inference.cli analyze \
  --video /path/to/video.mp4 \
  --config configs/default.yaml \
  --output /path/to/output
```

The command uses `ForgeryAnalysisPipeline`; it neither downloads media/models nor writes intermediate evidence CSVs.

## Small residual-head training

The manifest columns are `video_path,label,split` plus optional temporal/spatial annotation paths. Splits must be `train`, `validation`, or `test`.

```bash
python -m inference.cli train \
  --manifest /path/to/manifest.csv \
  --config configs/default.yaml \
  --epochs 3
```

Only a LayerNorm→GRU→MLP residual fusion head is trained with `BCEWithLogitsLoss`. YOLO, UniDepth, pose, tracking, geometry, and residual producers stay frozen.

## Output files

Inference writes:

- `result.json`
- `timeline.csv` and `timeline.png`
- `clip_scores.csv`
- `object_scores.csv`
- `track_scores.csv`
- `suspicious_clips.json`
- `abnormal_tracks.png`
- `structural_residual_heatmap.png`

The track plot shows actual and predicted locations plus their deviation. The heatmap is produced from residual spatial support, not a fixed-color mask overlay.

## Current limitations

- The new path has synthetic end-to-end validation; no paper performance claim is made.
- A local short video additionally requires verified local provider assets and provider-compatible frame content.
- Cross-frame object/point tracking and formal occlusion/reappearance measurements must be supplied by a `TrackProvider` or shared-observation producer; absent evidence stays unavailable.
- UniDepth produces model-predicted metric depth, not sensor ground truth.
- World-frame geometry is not claimed; the active semantic surface is camera-frame metric 2.5D.
- Historical modules have not yet been deleted and remain available only for comparison and validation.
