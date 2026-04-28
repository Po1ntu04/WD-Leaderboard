# Chinese Word Segmentation Leaderboard and Analytics Dashboard

This repository includes a local, offline leaderboard and visual analytics dashboard for Chinese Word Segmentation submissions.

Canonical implementation directory: `platform/`. The `app/` scripts are thin compatibility wrappers that dispatch to `platform/app/*`.

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

### Validation and tolerant row-level scoring

The scoring layer uses policy B:

1. File-level fatal issues still fail the whole submission, including missing file, non-UTF-8 encoding, illegal control characters, invalid runtime metadata, oversized files, and unrecoverable severe format problems.
2. Row-level issues do not fail the whole submission:
   - `reconstruction_mismatch`: the predicted tokens cannot reconstruct the corresponding raw sentence;
   - `missing_line`: the prediction line is missing;
   - extra prediction lines are ignored and reported as warnings.
3. `reconstruction_mismatch` and `missing_line` rows remain in the denominator and are scored as 0. Their exported `pred_valid=0`, `is_evaluable=1`, predicted spans/boundaries are empty, and word/boundary F1 are 0.
4. `gold_status=excluded` rows remain visible for review but use `is_evaluable=0`, so they are excluded from ranking denominators.

The leaderboard includes `tolerant_issue_count`; reports include `validation_warnings` / `eval_warnings`.

### Word-level metrics

Word precision/recall/F1 use character spans, not token string sets.
This correctly handles repeated words.

Example: for raw text `重复重复`, gold spans and prediction spans are compared by `(start, end)` offsets.

### Boundary-level metrics

Boundary metrics compare internal word-boundary character positions.

- predicted boundary not in gold = over-segmentation;
- gold boundary not predicted = under-segmentation.

`boundary_table` stores only non-TN boundary positions:

- `boundary_case=TP`: gold and prediction both have the boundary;
- `boundary_case=FP`: prediction has an extra boundary;
- `boundary_case=FN`: gold has a missing boundary in the prediction.

True negatives are not stored because character positions without a gold or predicted boundary dominate the table and do not aid boundary-error inspection.

`span_error_table` groups continuous local boundary mismatches into reviewable error fragments with `raw_span`, `gold_span_tokens`, `pred_span_tokens`, `start_char`, `end_char`, `error_type`, and `severity`.

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
sentence_score_table.csv.gz
sentence_score_table.parquet
boundary_table.csv
boundary_table.csv.gz
boundary_table.parquet
span_error_table.csv
span_error_table.csv.gz
span_error_table.parquet
long_tables_manifest.json
```

Parquet export requires `pyarrow`. If unavailable, CSV remains the canonical fallback. The long tables are also written as `.csv.gz` because GitHub may not preview very large CSV files; `long_tables_manifest.json` records row counts, columns, and artifact paths for handoff validation.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the bundled small sample:

```bash
python platform/app/run_session.py \
  --prediction-dir test_assets/dashboard_sample/submissions \
  --raw test_assets/dashboard_sample/raw.txt \
  --gold test_assets/dashboard_sample/gold.txt \
  --manifest test_assets/dashboard_sample/manifest.csv \
  --results-dir platform/results/dashboard_sample
```

Open the dashboard:

```bash
python platform/app/dashboard.py --results-dir platform/results/dashboard_sample
```

Simplified entry points also exist:

```bash
python app/session.py --prediction-dir submissions/
python app/leaderboard.py
```

## Dashboard navigation

The dashboard uses five grouped tabs instead of separate peer tabs for every visualization:

1. **Overview** — title, data version, KPI cards, leaderboard preview, top-15 bar chart, metric heatmap, and source summary.
2. **Leaderboard** — official table, subset heatmap, metric-scope rank comparison, and student/method profile.
3. **Diagnostics** — character-level boundary diff viewer, error-type chart, sentence difficulty map, and collapsed developer/raw artifact views.
4. **Gold Review** — review console for `confirmed` / `suspicious` / `excluded` gold rows, with low-F1 and high-discrimination review reasons.
5. **Experimental** — exploratory Top-N token/error bars, error-flow Sankey, metric-space scatter, and collapsed similarity network. These are exploratory only and are not used for official ranking.

The visual design is intentionally clean and academic: `#f8fafc` background, white cards, blue-gray accents, dense readable tables, no 3D charts, and every chart maps back to an exported table.

## PPT-ready figure export

Export static 16:9 PNG figures for slides:

```bash
python app/export_figures.py \
  --results-dir platform/results \
  --out-dir platform/results/figures \
  --font-path /path/to/ChineseFont.ttf
```

The `--font-path` argument is recommended for Chinese word clouds; font files are not committed to this repository.

The exporter writes:

- `01_leaderboard_top12.png`
- `02_metric_space_scatter.png`
- `03_subset_heatmap_top20.png`
- `04_error_type_distribution.png`
- `05_sentence_difficulty_map.png`
- `06_dataset_source_distribution.png`
- `07_boundary_diff_case.png`
- `08_gold_review_low_f1_cases.png`
- `wordcloud_dataset_gold.png`
- `wordcloud_common_error_spans.png`

Static Plotly PNG export uses `kaleido`; word clouds use `wordcloud`. Both are listed in `requirements.txt`.

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
- invalid reconstruction is tolerated at row level and counted as 0, not excluded.
