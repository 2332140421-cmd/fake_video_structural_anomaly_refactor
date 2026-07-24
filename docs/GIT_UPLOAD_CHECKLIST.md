# Git Upload Checklist

## Commit

Commit source, tests, configs, and documentation, including the M1/M2 files
listed in `configs/handoff/p4c3b_m2_server_handoff_v1.yaml`.

Do not commit:

- `.venv/`, caches, `.codex/`;
- `checkpoints/` or external UniDepth source;
- videos under `data/tests_videos/`;
- generated M1/M2/P4-B.5 outputs;
- secrets or credentials.

## Before Commit

```bash
.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
.venv/bin/python -m pytest \
  tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py -q -rs
.venv/bin/python -m pytest -q -rs
git status --short
git diff --check
git diff --stat
```

Review every file shown by `git status`. Existing protocol CSV changes and
server-script changes belong to earlier uncommitted stages; include them only
after reviewing that they are intended.

## Suggested Commit Summary

```text
Complete P4-C3B metric depth and single-frame metric Scene3D smoke

- add offline UniDepthV2 metric-depth provider with provenance and quality audit
- add camera-frame metric visible-surface, object, boundary and internal points
- add robust object extents and single-frame metric structure graphs
- preserve NaN validity semantics and frozen protocol/prior hashes
- add M1/M2 smoke configs, tests, server handoff manifest and continuation guide
```

## After Push

```bash
git rev-parse HEAD
git status --short
git log -1 --oneline
git remote -v
```

Save the commit SHA and check out that exact SHA on the server.
