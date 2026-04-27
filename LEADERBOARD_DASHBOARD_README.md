# Chinese Word Segmentation Leaderboard and Analytics Dashboard

This repository includes a local, offline leaderboard and visual analytics dashboard for Chinese Word Segmentation submissions.

## Inputs

Required files:

```text
raw.txt              # one raw sentence per line
gold.txt             # one gold segmentation per line
submissions/*.txt    # one pred.txt-style file per participant/method
```

Optional metadata:

```text
manifest.csv
```

Recommended manifest columns:

- `line_no`
- `dataset` or `source`
- `difficulty_bucket` or `difficulty`
- `sentence_type`
- `gold_status`: `confirmed`, `suspicious`, or `excluded`
- `selection_tags`
- `raw_text`

If `gold_status=excluded`, the sentence remains visible in exported review tables but is excluded from ranking metrics.
If `gold_status=suspicious`, the sentence is scored but visibly marked for review.

## Segmentation format

The parser accepts both canonical slash-separated and whitespace-separated segmentations:

```text
我 / 爱 / 北京
我 爱 北京
```

Internally, tokens are normalized before validation and scoring.

## Metrics

All metrics are computed in the scoring/export layer before dashboard rendering.
The dashboard only reads exported tables.

### Validation

A prediction file is valid only when:

1. line count equals `raw.txt` line count;
2. for every line, `''.join(pred_tokens) == raw_text`.

Invalid submissions are exported with failure status and zero ranking score.

### Word-level metrics

Word precision/recall/F1 use character spans, not token string sets.
This correctly handles repeated words.

Example: for raw text `重复重复`, gold spans and prediction spans are compared by `(start, end)` offsets.

### Boundary-level metrics

Boundary metrics compare internal word-boundary character positions.

- predicted boundary not in gold = over-segmentation;
- gold boundary not predicted = under-segmentation.

### Sentence metrics

- exact-match sentence rate;
- sentence average word/boundary F1 across submissions;
- discrimination index: top-vs-bottom score spread for each sentence.

### Subset metrics

If metadata is available, scores are grouped by:

- `source`
- `difficulty`
- `sentence_type`

## Standard exports

Running a session writes:

```text
leaderboard.json
sentence_table.csv
sentence_table.json
submission_table.csv
submission_table.json
sentence_score_table.csv
sentence_score_table.parquet
boundary_table.csv
boundary_table.parquet
span_error_table.csv
span_error_table.parquet
```

Parquet export requires `pyarrow`. If unavailable, CSV remains the canonical fallback.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the bundled small sample:

```bash
python my_platform/app/run_session.py \
  --prediction-dir test_assets/dashboard_sample/submissions \
  --raw test_assets/dashboard_sample/raw.txt \
  --gold test_assets/dashboard_sample/gold.txt \
  --manifest test_assets/dashboard_sample/manifest.csv \
  --results-dir my_platform/results/dashboard_sample
```

Open the dashboard:

```bash
python my_platform/app/dashboard.py --results-dir my_platform/results/dashboard_sample
```

Simplified entry points also exist:

```bash
python app/session.py --prediction-dir submissions/
python app/leaderboard.py
```

## Dashboard pages

P0:

1. Overview Dashboard
2. Leaderboard Table
3. Subset Score Heatmap
4. Boundary Diff Viewer
5. Sentence Difficulty Scatter Plot

P1:

6. Student / Method Profile
7. Error Type Bar Chart
8. Rank Delta View
9. Gold Review Console

P2:

10. Word Cloud
11. Sankey Chart
12. Student Clustering
13. Network Graph

The visual design is intentionally clean and academic: white / blue-gray background, dense readable tables, no 3D charts, and every chart maps back to an exported table.

## Tests

```bash
python -m pytest tests/test_leaderboard_analytics_exports.py -q
python -m pytest tests -q
```

The dashboard sample verifies:

- perfect prediction gives F1 = 1;
- over-segmentation is detected;
- under-segmentation is detected;
- repeated words are scored by spans;
- invalid reconstruction is flagged.
