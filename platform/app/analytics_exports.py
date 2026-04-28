from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from algorithms.common.io import SEGMENT_DELIMITER_PADDED, read_raw_file, read_segmented_file


SUBSET_TYPES = ("source", "difficulty", "sentence_type")
GOLD_STATUS_CONFIRMED = "confirmed"
GOLD_STATUS_SUSPICIOUS = "suspicious"
GOLD_STATUS_EXCLUDED = "excluded"


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def prf(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = safe_div(correct, predicted)
    recall = safe_div(correct, gold)
    return {
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1(precision, recall)),
    }


def token_spans(tokens: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = 0
    for index, token in enumerate(tokens):
        end = start + len(token)
        spans.append({"index": index, "start": start, "end": end, "text": token})
        start = end
    return spans


def span_key(span: dict[str, Any]) -> tuple[int, int]:
    return int(span["start"]), int(span["end"])


def boundary_positions(tokens: list[str], raw_len: int) -> set[int]:
    positions: set[int] = set()
    offset = 0
    for token in tokens[:-1]:
        offset += len(token)
        if 0 < offset < raw_len:
            positions.add(offset)
    return positions


def boundary_context(raw_text: str, boundary: int, window: int = 8) -> dict[str, Any]:
    return {
        "left_char": raw_text[boundary - 1] if 0 < boundary <= len(raw_text) else "",
        "right_char": raw_text[boundary] if 0 <= boundary < len(raw_text) else "",
        "left_context": raw_text[max(0, boundary - window) : boundary],
        "right_context": raw_text[boundary : min(len(raw_text), boundary + window)],
    }


def tokens_in_region(tokens: list[str], start_char: int, end_char: int) -> list[str]:
    out: list[str] = []
    for span in token_spans(tokens):
        if int(span["start"]) >= start_char and int(span["end"]) <= end_char:
            out.append(str(span["text"]))
    return out


def classify_boundary_error(fp_boundaries: set[int], fn_boundaries: set[int]) -> str:
    if fp_boundaries and not fn_boundaries:
        return "over_seg"
    if fn_boundaries and not fp_boundaries:
        return "under_seg"
    if fp_boundaries and fn_boundaries and len(fp_boundaries) == len(fn_boundaries):
        return "boundary_shift"
    return "mixed"


def local_span_error_rows(
    *,
    submission_name: str,
    sentence_id: int,
    raw_text: str,
    gold_tokens: list[str],
    pred_tokens: list[str],
    subsets: dict[str, Any],
    gold_boundaries: set[int],
    pred_boundaries: set[int],
) -> list[dict[str, Any]]:
    mismatch_boundaries = sorted(gold_boundaries ^ pred_boundaries)
    if not mismatch_boundaries:
        return []

    common_anchors = sorted({0, len(raw_text)} | (gold_boundaries & pred_boundaries))
    rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(mismatch_boundaries):
        first = mismatch_boundaries[cursor]
        start_char = max(anchor for anchor in common_anchors if anchor < first)
        end_char = min(anchor for anchor in common_anchors if anchor > first)
        region_boundaries = [first]
        cursor += 1
        while cursor < len(mismatch_boundaries) and mismatch_boundaries[cursor] < end_char:
            region_boundaries.append(mismatch_boundaries[cursor])
            cursor += 1

        region_set = set(region_boundaries)
        fp_boundaries = region_set & (pred_boundaries - gold_boundaries)
        fn_boundaries = region_set & (gold_boundaries - pred_boundaries)
        rows.append(
            {
                "submission_name": submission_name,
                "sentence_id": sentence_id,
                **subsets,
                "raw_span": raw_text[start_char:end_char],
                "gold_span_tokens": " / ".join(tokens_in_region(gold_tokens, start_char, end_char)),
                "pred_span_tokens": " / ".join(tokens_in_region(pred_tokens, start_char, end_char)),
                "start_char": start_char,
                "end_char": end_char,
                "error_type": classify_boundary_error(fp_boundaries, fn_boundaries),
                "severity": len(region_boundaries),
                "false_positive_boundary_count": len(fp_boundaries),
                "false_negative_boundary_count": len(fn_boundaries),
            }
        )
    return rows


def infer_difficulty(row: dict[str, Any]) -> str:
    explicit = str(row.get("difficulty") or row.get("difficulty_bucket") or "").strip()
    if explicit:
        return explicit
    dataset = str(row.get("dataset") or row.get("source") or "")
    if dataset == "samechar":
        return "specialized"
    try:
        score = float(row.get("difficulty_score", 0) or 0)
    except Exception:
        score = 0.0
    if score >= 55:
        return "high"
    if score > 0:
        return "medium"
    return "unknown"


def infer_sentence_type(row: dict[str, Any]) -> str:
    explicit = str(row.get("sentence_type") or "").strip()
    if explicit:
        return explicit
    dataset = str(row.get("dataset") or row.get("source") or "")
    tags = str(row.get("selection_tags") or "").lower()
    if dataset == "samechar" or "samechar" in tags:
        return "samechar"
    if dataset == "TCM-Ancient-Books" or "tcm" in tags or "medical" in tags:
        return "tcm"
    if "classical" in tags or "ancient" in tags:
        return "classical"
    if "mixed_script" in tags:
        return "mixed_script"
    if "long_sentence" in tags:
        return "long_sentence"
    if "sentence_level" in tags:
        return "sentence_level"
    return "standard"


def infer_gold_status(row: dict[str, Any]) -> str:
    explicit = str(row.get("gold_status") or "").strip().lower()
    if explicit in {GOLD_STATUS_CONFIRMED, GOLD_STATUS_SUSPICIOUS, GOLD_STATUS_EXCLUDED}:
        return explicit
    review_flags = str(row.get("review_flags") or "").lower()
    if any(flag in review_flags for flag in ("exclude", "excluded", "drop", "invalid_gold")):
        return GOLD_STATUS_EXCLUDED
    if any(flag in review_flags for flag in ("suspicious", "review", "uncertain", "疑")):
        return GOLD_STATUS_SUSPICIOUS
    return GOLD_STATUS_CONFIRMED


def read_manifest_frame(manifest_path: str | Path | None, raw_rows: list[str]) -> pd.DataFrame:
    if manifest_path and Path(manifest_path).exists():
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig").reset_index(drop=True)
    else:
        manifest = pd.DataFrame({"line_no": list(range(1, len(raw_rows) + 1)), "raw_text": raw_rows})

    if len(manifest) < len(raw_rows):
        missing = pd.DataFrame(
            {
                "line_no": list(range(len(manifest) + 1, len(raw_rows) + 1)),
                "raw_text": raw_rows[len(manifest) :],
            }
        )
        manifest = pd.concat([manifest, missing], ignore_index=True)
    return manifest.iloc[: len(raw_rows)].copy()


def build_sentence_table(
    raw_rows: list[str],
    gold_rows: list[list[str]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = manifest.to_dict(orient="records")
    for index, raw_text in enumerate(raw_rows):
        manifest_row = records[index] if index < len(records) else {}
        row = {
            "sentence_id": index + 1,
            "line_no": int(manifest_row.get("line_no") or index + 1),
            "sample_id": str(manifest_row.get("sample_id") or f"sent_{index + 1:04d}"),
            "source": str(manifest_row.get("source") or manifest_row.get("dataset") or "unknown"),
            "difficulty": infer_difficulty(manifest_row),
            "sentence_type": infer_sentence_type(manifest_row),
            "difficulty_score": manifest_row.get("difficulty_score", ""),
            "gold_status": infer_gold_status(manifest_row),
            "selection_tags": str(manifest_row.get("selection_tags") or ""),
            "raw_text": raw_text,
            "gold": SEGMENT_DELIMITER_PADDED.join(gold_rows[index]) if index < len(gold_rows) else "",
            "char_len": len(raw_text),
            "gold_word_count": len(gold_rows[index]) if index < len(gold_rows) else 0,
        }
        for optional in ("split", "split_role", "balanced_split", "review_flags", "notes"):
            if optional in manifest_row:
                row[optional] = manifest_row.get(optional, "")
        rows.append(row)
    return pd.DataFrame(rows)


def validate_prediction_rows(raw_rows: list[str], pred_rows: list[list[str]]) -> tuple[list[str], dict[int, str]]:
    errors: list[str] = []
    sentence_errors: dict[int, str] = {}
    if len(raw_rows) != len(pred_rows):
        message = f"行数不匹配：raw={len(raw_rows)} pred={len(pred_rows)}"
        errors.append(message)
        for sentence_id in range(1, len(raw_rows) + 1):
            sentence_errors[sentence_id] = "line_count_mismatch"
        return errors, sentence_errors

    for index, (raw_text, pred_tokens) in enumerate(zip(raw_rows, pred_rows), start=1):
        if "".join(pred_tokens) != raw_text:
            message = f"第 {index} 行分词结果无法还原原句。"
            errors.append(message)
            sentence_errors[index] = "reconstruction_mismatch"
            if len(errors) >= 20:
                errors.append("更多重构错误已省略。")
                break
    return errors, sentence_errors


def _sentence_subset(sentence_table: pd.DataFrame, sentence_id: int) -> dict[str, Any]:
    if sentence_table.empty or sentence_id - 1 >= len(sentence_table):
        return {"source": "unknown", "difficulty": "unknown", "sentence_type": "unknown"}
    row = sentence_table.iloc[sentence_id - 1]
    return {key: row.get(key, "unknown") for key in SUBSET_TYPES}


def _sentence_gold_status(sentence_table: pd.DataFrame, sentence_id: int) -> str:
    if sentence_table.empty or sentence_id - 1 >= len(sentence_table):
        return GOLD_STATUS_CONFIRMED
    return str(sentence_table.iloc[sentence_id - 1].get("gold_status", GOLD_STATUS_CONFIRMED) or GOLD_STATUS_CONFIRMED)


def score_sentence(
    *,
    submission_name: str,
    sentence_id: int,
    raw_text: str,
    gold_tokens: list[str],
    pred_tokens: list[str],
    subsets: dict[str, Any],
    validation_status: str,
    gold_status: str = GOLD_STATUS_CONFIRMED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pred_valid = validation_status == "ok"
    is_evaluable = gold_status != GOLD_STATUS_EXCLUDED
    gold_spans = token_spans(gold_tokens)
    pred_spans = token_spans(pred_tokens) if pred_valid else []
    gold_span_keys = {span_key(span): span for span in gold_spans}
    pred_span_keys = {span_key(span): span for span in pred_spans}
    correct_word_spans = set(gold_span_keys) & set(pred_span_keys)

    gold_boundaries = boundary_positions(gold_tokens, len(raw_text))
    pred_boundaries = boundary_positions(pred_tokens, len(raw_text)) if pred_valid else set()
    correct_boundaries = gold_boundaries & pred_boundaries
    over_segmentation = pred_boundaries - gold_boundaries
    under_segmentation = gold_boundaries - pred_boundaries

    word_metrics = prf(len(correct_word_spans), len(pred_span_keys), len(gold_span_keys))
    boundary_metrics = prf(len(correct_boundaries), len(pred_boundaries), len(gold_boundaries))
    exact_match = int(pred_valid and gold_tokens == pred_tokens)

    score_row: dict[str, Any] = {
        "submission_name": submission_name,
        "sentence_id": sentence_id,
        **subsets,
        "validation_status": validation_status,
        "gold_status": gold_status,
        "pred_valid": int(pred_valid),
        "is_evaluable": int(is_evaluable),
        "is_scored": int(is_evaluable),
        "word_precision": word_metrics["precision"],
        "word_recall": word_metrics["recall"],
        "word_f1": word_metrics["f1"],
        "boundary_precision": boundary_metrics["precision"],
        "boundary_recall": boundary_metrics["recall"],
        "boundary_f1": boundary_metrics["f1"],
        "exact_match": exact_match,
        "gold_word_count": len(gold_span_keys),
        "pred_word_count": len(pred_span_keys),
        "correct_word_count": len(correct_word_spans),
        "gold_boundary_count": len(gold_boundaries),
        "pred_boundary_count": len(pred_boundaries),
        "correct_boundary_count": len(correct_boundaries),
        "over_segmentation_count": len(over_segmentation),
        "under_segmentation_count": len(under_segmentation),
    }

    boundary_rows: list[dict[str, Any]] = []
    if not is_evaluable:
        return score_row, boundary_rows, []

    for boundary in sorted(gold_boundaries | pred_boundaries):
        gold_boundary = boundary in gold_boundaries
        pred_boundary = boundary in pred_boundaries
        if gold_boundary and pred_boundary:
            boundary_case = "TP"
            boundary_type = "true_positive"
        elif pred_boundary:
            boundary_case = "FP"
            boundary_type = "over_segmentation"
        else:
            boundary_case = "FN"
            boundary_type = "under_segmentation"
        boundary_rows.append(
            {
                "submission_name": submission_name,
                "sentence_id": sentence_id,
                **subsets,
                "boundary_position": boundary,
                **boundary_context(raw_text, boundary),
                "gold_boundary": int(gold_boundary),
                "pred_boundary": int(pred_boundary),
                "boundary_case": boundary_case,
                "boundary_type": boundary_type,
            }
        )

    span_error_rows = local_span_error_rows(
        submission_name=submission_name,
        sentence_id=sentence_id,
        raw_text=raw_text,
        gold_tokens=gold_tokens,
        pred_tokens=pred_tokens if pred_valid else [],
        subsets=subsets,
        gold_boundaries=gold_boundaries,
        pred_boundaries=pred_boundaries,
    )

    return score_row, boundary_rows, span_error_rows


def aggregate_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if int(row.get("is_evaluable", row.get("is_scored", 1)) or 0) == 1]
    total_sentences = len(rows)
    gold_words = sum(int(row.get("gold_word_count", 0) or 0) for row in rows)
    pred_words = sum(int(row.get("pred_word_count", 0) or 0) for row in rows)
    correct_words = sum(int(row.get("correct_word_count", 0) or 0) for row in rows)
    gold_boundaries = sum(int(row.get("gold_boundary_count", 0) or 0) for row in rows)
    pred_boundaries = sum(int(row.get("pred_boundary_count", 0) or 0) for row in rows)
    correct_boundaries = sum(int(row.get("correct_boundary_count", 0) or 0) for row in rows)
    exact_matches = sum(int(row.get("exact_match", 0) or 0) for row in rows)
    word = prf(correct_words, pred_words, gold_words)
    boundary = prf(correct_boundaries, pred_boundaries, gold_boundaries)
    return {
        "precision": word["precision"],
        "recall": word["recall"],
        "f1": word["f1"],
        "word_precision": word["precision"],
        "word_recall": word["recall"],
        "word_f1": word["f1"],
        "boundary_precision": boundary["precision"],
        "boundary_recall": boundary["recall"],
        "boundary_f1": boundary["f1"],
        "exact_match_sentence_rate": _round(safe_div(exact_matches, total_sentences)),
        "exact_match_sentences": exact_matches,
        "total_sentences": total_sentences,
        "gold_words": gold_words,
        "pred_words": pred_words,
        "correct_words": correct_words,
        "gold_boundaries": gold_boundaries,
        "pred_boundaries": pred_boundaries,
        "correct_boundaries": correct_boundaries,
        "over_segmentation_count": sum(int(row.get("over_segmentation_count", 0) or 0) for row in rows),
        "under_segmentation_count": sum(int(row.get("under_segmentation_count", 0) or 0) for row in rows),
    }


def subset_scores(sentence_score_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {subset_type: {} for subset_type in SUBSET_TYPES}
    for subset_type in SUBSET_TYPES:
        values = sorted({str(row.get(subset_type, "unknown")) for row in sentence_score_rows})
        for value in values:
            rows = [row for row in sentence_score_rows if str(row.get(subset_type, "unknown")) == value]
            out[subset_type][value] = aggregate_score_rows(rows)
    return out


def sentence_level_statistics(sentence_score_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(sentence_score_rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "sentence_id",
                "participant_count",
                "sentence_avg_word_f1",
                "sentence_avg_boundary_f1",
                "sentence_exact_match_rate",
                "discrimination_index",
            ]
        )
    scored = frame[frame.get("is_evaluable", frame.get("is_scored", 1)).astype(int) == 1].copy()
    if scored.empty:
        return pd.DataFrame(columns=["sentence_id", "participant_count", "sentence_avg_word_f1", "sentence_avg_boundary_f1", "sentence_exact_match_rate", "discrimination_index"])
    for column in ["word_f1", "boundary_f1", "exact_match"]:
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)

    def discrimination(values: pd.Series) -> float:
        values = values.sort_values().reset_index(drop=True)
        if values.empty:
            return 0.0
        bucket_size = max(1, int(round(len(values) * 0.27)))
        lower = values.head(bucket_size).mean()
        upper = values.tail(bucket_size).mean()
        return _round(float(upper - lower))

    grouped = scored.groupby("sentence_id")
    stats = grouped.agg(
        participant_count=("submission_name", "nunique"),
        sentence_avg_word_f1=("word_f1", "mean"),
        sentence_avg_boundary_f1=("boundary_f1", "mean"),
        sentence_exact_match_rate=("exact_match", "mean"),
    ).reset_index()
    stats["discrimination_index"] = grouped["word_f1"].apply(discrimination).reset_index(drop=True)
    for column in ["sentence_avg_word_f1", "sentence_avg_boundary_f1", "sentence_exact_match_rate", "discrimination_index"]:
        stats[column] = stats[column].map(_round)
    return stats


def flatten_subset_columns(subsets: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for subset_type, subset_payload in subsets.items():
        for subset_name, metrics in subset_payload.items():
            safe_name = str(subset_name).replace(" ", "_").replace("/", "_")
            prefix = f"{subset_type}:{safe_name}"
            for metric in ("word_f1", "boundary_f1", "exact_match_sentence_rate"):
                row[f"{prefix}:{metric}"] = metrics.get(metric, 0.0)
    return row


def evaluate_submission(
    *,
    raw_rows: list[str],
    gold_rows: list[list[str]],
    pred_rows: list[list[str]],
    sentence_table: pd.DataFrame,
    submission_name: str,
    status: str,
    validation_errors: list[str],
    runtime_seconds: float | None = None,
    submission_group: str = "",
    mode: str = "",
    submission_path: str = "",
    timestamp: str = "",
    validation_statuses: dict[int, str] | None = None,
) -> dict[str, Any]:
    _, sentence_errors = validate_prediction_rows(raw_rows, pred_rows)
    sentence_errors.update(validation_statuses or {})
    sentence_score_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    span_error_rows: list[dict[str, Any]] = []
    scoring_enabled = status == "成功" and not validation_errors

    for index, raw_text in enumerate(raw_rows):
        sentence_id = index + 1
        gold_status = _sentence_gold_status(sentence_table, sentence_id)
        pred_tokens = pred_rows[index] if scoring_enabled and index < len(pred_rows) else []
        validation_status = sentence_errors.get(sentence_id, "ok") if scoring_enabled else sentence_errors.get(sentence_id, "submission_invalid")
        score_row, boundary, span_errors = score_sentence(
            submission_name=submission_name,
            sentence_id=sentence_id,
            raw_text=raw_text,
            gold_tokens=gold_rows[index] if index < len(gold_rows) else [],
            pred_tokens=pred_tokens,
            subsets=_sentence_subset(sentence_table, sentence_id),
            validation_status=validation_status,
            gold_status=gold_status,
        )
        sentence_score_rows.append(score_row)
        boundary_rows.extend(boundary)
        span_error_rows.extend(span_errors)

    overall = aggregate_score_rows(sentence_score_rows) if scoring_enabled else aggregate_score_rows([])
    if not scoring_enabled:
        overall["total_sentences"] = len(raw_rows)
    subsets = subset_scores(sentence_score_rows) if scoring_enabled else {subset_type: {} for subset_type in SUBSET_TYPES}
    submission_row = {
        "submission_name": submission_name,
        "submission_group": submission_group,
        "mode": mode,
        "status": status,
        "timestamp": timestamp,
        "runtime_seconds": _round(float(runtime_seconds or 0.0)),
        "message": validation_errors[0] if validation_errors else "",
        **overall,
        **flatten_subset_columns(subsets),
        "submission_path": submission_path,
    }
    return {
        "overall": overall,
        "subset_scores": subsets,
        "submission_row": submission_row,
        "sentence_score_rows": sentence_score_rows,
        "boundary_rows": boundary_rows,
        "span_error_rows": span_error_rows,
    }


def _write_json_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_table_pair(frame: pd.DataFrame, csv_path: Path, json_path: Path | None = None) -> None:
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if json_path:
        _write_json_records(json_path, frame.to_dict(orient="records"))


def _write_long_table(frame: pd.DataFrame, csv_path: Path) -> dict[str, Any]:
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    gzip_path = csv_path.with_suffix(csv_path.suffix + ".gz")
    frame.to_csv(gzip_path, index=False, encoding="utf-8-sig", compression="gzip")
    return {
        "csv": str(csv_path.resolve()),
        "csv_gzip": str(gzip_path.resolve()),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
    }


def _write_parquet(frame: pd.DataFrame, parquet_path: Path) -> None:
    try:
        frame.to_parquet(parquet_path, index=False)
        unavailable = parquet_path.with_suffix(parquet_path.suffix + ".unavailable.txt")
        if unavailable.exists():
            unavailable.unlink()
    except Exception as exc:
        parquet_path.with_suffix(parquet_path.suffix + ".unavailable.txt").write_text(
            "Parquet export requires an installed parquet engine such as pyarrow.\n"
            f"CSV export is available. Original error: {exc}\n",
            encoding="utf-8",
        )


def export_standard_tables(
    *,
    results_dir: str | Path,
    raw_path: str | Path,
    gold_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    raw_rows = read_raw_file(raw_path)
    gold_rows = read_segmented_file(gold_path)
    manifest = read_manifest_frame(manifest_path, raw_rows)
    sentence_table = build_sentence_table(raw_rows, gold_rows, manifest)

    reports_dir = results / "reports"
    reports = []
    if reports_dir.exists():
        for report_path in sorted(reports_dir.glob("*.report.json")):
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    submission_rows: list[dict[str, Any]] = []
    sentence_score_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    span_error_rows: list[dict[str, Any]] = []
    for report in reports:
        if report.get("submission_row"):
            submission_rows.append(report["submission_row"])
        elif report.get("overall"):
            submission_rows.append(
                {
                    "submission_name": report.get("submission_name", ""),
                    "status": report.get("status", ""),
                    **report.get("overall", {}),
                }
            )
        sentence_score_rows.extend(report.get("sentence_score_rows", []))
        boundary_rows.extend(report.get("boundary_rows", []))
        span_error_rows.extend(report.get("span_error_rows", []))

    submission_table = pd.DataFrame(submission_rows)
    if not submission_table.empty:
        submission_table["_status_order"] = submission_table["status"].map(lambda value: 0 if value == "成功" else 1)
        submission_table = (
            submission_table.sort_values(
                ["_status_order", "word_f1", "boundary_f1", "runtime_seconds", "timestamp"],
                ascending=[True, False, False, True, True],
                na_position="last",
            )
            .drop(columns=["_status_order"])
            .reset_index(drop=True)
        )
        submission_table.insert(0, "rank", range(1, len(submission_table) + 1))
    else:
        submission_table = pd.DataFrame(columns=["rank", "submission_name", "status", "word_f1", "boundary_f1"])

    sentence_score_table = pd.DataFrame(sentence_score_rows)
    boundary_table = pd.DataFrame(boundary_rows)
    span_error_table = pd.DataFrame(span_error_rows)
    sentence_stats = sentence_level_statistics(sentence_score_rows)
    if not sentence_stats.empty:
        sentence_table = sentence_table.merge(sentence_stats, on="sentence_id", how="left")
    for column in ["participant_count", "sentence_avg_word_f1", "sentence_avg_boundary_f1", "sentence_exact_match_rate", "discrimination_index"]:
        if column not in sentence_table.columns:
            sentence_table[column] = 0
        sentence_table[column] = sentence_table[column].fillna(0)

    _write_table_pair(sentence_table, results / "sentence_table.csv", results / "sentence_table.json")
    _write_table_pair(submission_table, results / "submission_table.csv", results / "submission_table.json")
    _write_json_records(results / "leaderboard.json", submission_table.to_dict(orient="records"))
    long_table_manifest = {
        "note": (
            "Long-form analytics tables are exported as CSV plus CSV.GZ. "
            "GitHub may not preview large CSV files; use the .csv.gz or parquet artifacts "
            "or regenerate locally with python app/session.py --prediction-dir <demo_predictions_dir>."
        ),
        "tables": {
            "sentence_score_table": _write_long_table(sentence_score_table, results / "sentence_score_table.csv"),
            "boundary_table": _write_long_table(boundary_table, results / "boundary_table.csv"),
            "span_error_table": _write_long_table(span_error_table, results / "span_error_table.csv"),
        },
    }
    manifest_text = json.dumps(long_table_manifest, ensure_ascii=False, indent=2)
    (results / "long_tables_manifest.json").write_text(manifest_text, encoding="utf-8")
    _write_parquet(sentence_score_table, results / "sentence_score_table.parquet")
    _write_parquet(boundary_table, results / "boundary_table.parquet")
    _write_parquet(span_error_table, results / "span_error_table.parquet")

    return {
        "leaderboard": str((results / "leaderboard.json").resolve()),
        "sentence_table": str((results / "sentence_table.csv").resolve()),
        "submission_table": str((results / "submission_table.csv").resolve()),
        "sentence_score_table": str((results / "sentence_score_table.csv").resolve()),
        "boundary_table": str((results / "boundary_table.csv").resolve()),
        "span_error_table": str((results / "span_error_table.csv").resolve()),
    }
