# Fake Video Structural Anomaly Prototype

This repository is a research prototype for object-aware 3D structural residuals.
It contains frozen 2D/2.5D baselines, shared 3D observation contracts, static and
dynamic structural evidence, and P4 experiment protocol tooling.

- Ratio-space residual `R_sd`
- Log-space residual `R_sd_log`
- A small fusion interface reserved for future residual terms:
  `R_flow`, `R_track`, `R_depth_cons`, `R_occ`, and `R_corr`

Model weights and videos are intentionally excluded from Git. Their registries
record acquisition instructions and SHA-256 values for reproducible setup.

## Run tests

```bash
cd fake_video_structural_anomaly
./scripts/run_tests.sh
```

Create and use a project-local environment rather than copying one from another
machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Use this interpreter in VS Code or run:

```bash
.venv/bin/python -m pytest -q
```

## Run demo

```bash
.venv/bin/python examples/demo_scale_depth.py
```

Server setup is documented in `docs/SERVER_ENVIRONMENT.md` and
`docs/SERVER_DATA_SETUP.md`.
