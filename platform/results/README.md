# Demo Results

This directory stores generated demo leaderboard artifacts for the canonical `platform/` implementation.

Expected generated files:

- `leaderboard.csv`
- `index.html`
- `reports/*.report.json`

Regenerate the demo artifacts from the classroom prediction directory:

```bash
python app/session.py --prediction-dir "submit/2026春-分词大赛(word)/predictions"
```

The exported `boundary_table` intentionally stores only TP/FP/FN boundary positions. TN character positions are omitted because they are numerous and not useful for error review.
