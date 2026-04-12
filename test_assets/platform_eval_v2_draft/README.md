# Classroom Evaluation Package v2 Draft

This package is the current classroom evaluation draft under active review.

## Composition

- NLPCC-Weibo: 138
- EvaHan-2022: 46
- TCM-Ancient-Books: 23
- samechar: 23

Total: 230 lines

## Principles

- reduce obviously easy samples where practical
- keep samechar as the full 23-line specialized bucket
- prefer classroom-discussable, relatively difficult samples
- keep classroom evaluation separate from internal training

## Files

- `raw.txt`
- `gold.txt`
- `gold_manifest.csv`
- `raw_manifest.csv`
- `package_meta.json`
- `REVIEW.md`
- `evahan_simplification_review.csv`

## Notes

- `gold.txt` has passed self-check scoring against the same package.
- samechar is taken from the user-confirmed CSV as the authoritative source.
- EvaHan lines use conservative simplified-Chinese replacement only where the replacement was judged obvious enough for application-level use.
