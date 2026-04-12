# Platform Prototype

This directory contains the local classroom platform prototype for the workspace.

## Current Scope

- Chinese-first local scoring flow
- `pred.txt` submission scoring mode
- `exe + bat` executable package scoring mode
- leaderboard CSV generation
- static HTML leaderboard page generation
- session overview, filters, detail pages, and export links

## Default Evaluation Package

Default teacher-side package:

- `test_assets/platform_eval_v2_draft/raw.txt`
- `test_assets/platform_eval_v2_draft/gold.txt`
- `test_assets/platform_eval_v2_draft/gold_manifest.csv`

This package uses only subsets with valid gold segmentation.
In particular, `NLPCC-Weibo` now uses `dev` rather than `test`.

## Current Commands

Score one prediction file:

```powershell
python platform\app\score_submission.py --submission path\to\pred.txt --name your_name
```

Recommended simplified entry:

```powershell
python app\score.py --submission path\to\pred.txt --name your_name
```

Score one executable submission package:

```powershell
python platform\app\score_executable_submission.py --submission-dir path\to\submission_dir --name your_name
```

Note:

- executable-package scoring depends on `run.bat`
- it is intended for a Windows teacher machine
- in Linux/WSL environments, the platform will now report a clear environment error instead of a vague subprocess failure

Run one whole local session:

```powershell
python platform\app\run_session.py --prediction-dir path\to\predictions --executable-dir path\to\submissions
```

Recommended simplified entry:

```powershell
python app\session.py --prediction-dir path\to\predictions --executable-dir path\to\submissions
```

Prediction-file batch import notes:

- the session runner now scans `*.txt` files in the prediction directory
- recommended naming rule is `{学号}_{姓名}_pred.txt`
- `raw.txt` and `gold.txt` are ignored automatically if they appear in the folder

Build the static leaderboard page:

```powershell
python platform\app\build_demo_page.py
```

Run the interactive leaderboard dashboard:

```powershell
python platform\app\dashboard.py
```

Recommended simplified entry:

```powershell
python app\leaderboard.py
```

Generate the current end-to-end demo:

```powershell
python platform\app\generate_demo_results.py
```

## `pred.txt` Format

Canonical format:

```text
词1 / 词2 / 词3
```

Backward-compatible legacy format:

```text
词1 词2 词3
```

Validation rules:

- line count must match `raw.txt`
- concatenated tokens must exactly restore each raw line
- optional runtime metadata, if present, must be written in the final line as `# runtime_seconds: 数值`
- executable mode must generate `output/pred.txt`

## Notes

- `pred.txt` mode is the lower-risk backup mode.
- `exe + bat` mode is the primary official mode when runtime comparison is required.
- Both modes write into the same scoring and leaderboard pipeline.
- The demo package is still visible and engineering-oriented; the final classroom competition package should later be hidden and versioned separately.

