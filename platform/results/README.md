# Demo Results

This directory stores generated demo leaderboard artifacts for the canonical `platform/` implementation.

Expected generated files:

- `leaderboard.csv`
- `index.html`
- `reports/*.report.json`
- `sentence_score_table.csv` and `sentence_score_table.csv.gz`
- `boundary_table.csv` and `boundary_table.csv.gz`
- `span_error_table.csv` and `span_error_table.csv.gz`
- `long_tables_manifest.json`

Regenerate the demo artifacts from the classroom prediction directory:

```bash
python app/session.py --prediction-dir "submit/2026春-分词大赛(word)/predictions"
```

The three long analytics tables are intentionally committed both as plain CSV and compressed CSV.GZ handoff artifacts. GitHub may refuse to preview large CSV files such as `boundary_table.csv`; in that case download the file, use the `.csv.gz` artifact, or regenerate locally with the command above. The CSV/GZ/parquet files are produced by the same scoring run as `leaderboard.csv`, `submission_table.csv`, and `reports/*.report.json`.

The exported `boundary_table` intentionally stores only TP/FP/FN boundary positions. TN character positions are omitted because they are numerous and not useful for error review.
