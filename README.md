# Fake Video Structural Anomaly Prototype

This is a lightweight research prototype for object-aware 3D structural residuals.
The current stage implements object-level scale-depth consistency residuals:

- Ratio-space residual `R_sd`
- Log-space residual `R_sd_log`
- A small fusion interface reserved for future residual terms:
  `R_flow`, `R_track`, `R_depth_cons`, `R_occ`, and `R_corr`

No model download or large pretrained dependency is required.

## Run tests

```bash
cd /home/chenyh/fake_video_structural_anomaly
./scripts/run_tests.sh
```

The project-local Python environment is stored at:

```bash
/home/chenyh/fake_video_structural_anomaly/.venv/bin/python
```

Use this interpreter in VS Code or run:

```bash
cd /home/chenyh/fake_video_structural_anomaly
conda activate /home/chenyh/fake_video_structural_anomaly/.venv
pytest tests
```

## Run demo

```bash
cd /home/chenyh/fake_video_structural_anomaly
python examples/demo_scale_depth.py
```
