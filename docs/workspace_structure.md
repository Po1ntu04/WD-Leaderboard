# Workspace Structure and Development Contract

This repository should be edited as one canonical workspace. For local work, prefer
`/home/yu/projects/NLP_latest` and avoid running the older `/home/yu/projects/NLP`
copy, because that copy may still contain the historical P0/P1 tab dashboard.

## Canonical map

```text
app/                  Thin CLI wrappers for daily use.
platform/app/         Canonical implementation: scoring, sessions, dashboard, exports.
platform/results/     Committed demo/classroom artifacts consumed by the dashboard.
algorithms/common/    Shared segmentation IO/scoring helpers used by platform/app.
test_assets/          Small reproducible fixtures and evaluation package inputs.
tests/                Regression tests for scoring/export behavior.
docs/                 Submission protocol and workspace/developer notes.
scripts/              Developer utilities that do not belong to the runtime app.
```

## Daily commands

Use the root `Makefile` so commands always run from the correct workspace:

```bash
make workspace-check
make dashboard
make test
make export-figures
```

Equivalent direct commands remain available:

```bash
python app/leaderboard.py --results-dir platform/results --host 0.0.0.0 --port 8050
python app/export_figures.py --results-dir platform/results --out-dir platform/results/figures
python -m pytest tests -q
```

## Editing rules

- Edit `platform/app/dashboard.py` for frontend behavior.
- Edit `platform/app/assets/style.css` for frontend styling.
- Edit `platform/app/export_figures.py` for PPT/static figure export.
- Keep `app/*.py` as thin compatibility wrappers only.
- Do not reintroduce `my_platform/` or competing implementation directories.
- Do not change scoring semantics or long-table schemas from dashboard-only work.
- Generated PPT figures belong in `platform/results/figures/` and are ignored by git.

## Results policy

`platform/results/` contains a committed demo handoff snapshot so the dashboard can
open immediately after clone. Large regenerated outputs should only be committed
when they are intentionally part of a new demo handoff.

For temporary experiments, write to a separate results directory, for example:

```bash
python app/session.py --prediction-dir test_assets/dashboard_sample/submissions \
  --raw test_assets/dashboard_sample/raw.txt \
  --gold test_assets/dashboard_sample/gold.txt \
  --manifest test_assets/dashboard_sample/manifest.csv \
  --results-dir /tmp/wd-dashboard-sample
```

## Avoiding the old-dashboard trap

Before launching the dashboard, verify the workspace:

```bash
make workspace-check
```

Expected output should say `Dashboard: new grouped tabs`. If it says old/unknown,
you are probably in the wrong checkout.
