from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from algorithms.common.io import SEGMENT_DELIMITER_PADDED


ELLIPSIS_CHARS = {'.', '。', '…', '．'}


@dataclass
class ScoreResult:
    precision: float
    recall: float
    f1: float
    gold_words: int
    pred_words: int
    correct_words: int
    exact_match_sentences: int
    total_sentences: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "gold_words": self.gold_words,
            "pred_words": self.pred_words,
            "correct_words": self.correct_words,
            "exact_match_sentences": self.exact_match_sentences,
            "total_sentences": self.total_sentences,
        }


def _word_spans(words: list[str]) -> set[tuple[int, int]]:
    spans = set()
    start = 0
    for word in words:
        end = start + len(word)
        spans.add((start, end))
        start = end
    return spans


def _normalize_ellipsis_tokens(words: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(words):
        token = words[i]
        if token and all(ch in ELLIPSIS_CHARS for ch in token):
            j = i
            parts: list[str] = []
            while j < len(words) and words[j] and all(ch in ELLIPSIS_CHARS for ch in words[j]):
                parts.append(words[j])
                j += 1
            merged = ''.join(parts)
            if len(merged) >= 3:
                normalized.append(merged)
                i = j
                continue
        normalized.append(token)
        i += 1
    return normalized


def score_predictions(gold_rows: list[list[str]], pred_rows: list[list[str]]) -> ScoreResult:
    if len(gold_rows) != len(pred_rows):
        raise ValueError("gold and prediction line counts do not match")

    gold_words = pred_words = correct_words = exact_match = 0
    for gold, pred in zip(gold_rows, pred_rows):
        gold = _normalize_ellipsis_tokens(gold)
        pred = _normalize_ellipsis_tokens(pred)
        gold_words += len(gold)
        pred_words += len(pred)
        g_spans = _word_spans(gold)
        p_spans = _word_spans(pred)
        correct_words += len(g_spans & p_spans)
        if gold == pred:
            exact_match += 1

    precision = correct_words / pred_words if pred_words else 0.0
    recall = correct_words / gold_words if gold_words else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return ScoreResult(
        precision=precision,
        recall=recall,
        f1=f1,
        gold_words=gold_words,
        pred_words=pred_words,
        correct_words=correct_words,
        exact_match_sentences=exact_match,
        total_sentences=len(gold_rows),
    )


def build_word_vocab(rows: list[list[str]]) -> set[str]:
    vocab: set[str] = set()
    for words in rows:
        for word in words:
            if word:
                vocab.add(word)
    return vocab


def oov_recall(gold_rows: list[list[str]], pred_rows: list[list[str]], train_vocab: set[str]) -> dict[str, float | int]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError("gold and prediction line counts do not match")

    total_oov_words = 0
    correct_oov_words = 0
    total_gold_words = 0
    for gold, pred in zip(gold_rows, pred_rows):
        pred_spans = _word_spans(pred)
        start = 0
        for word in gold:
            end = start + len(word)
            total_gold_words += 1
            if word not in train_vocab:
                total_oov_words += 1
                if (start, end) in pred_spans:
                    correct_oov_words += 1
            start = end

    recall = correct_oov_words / total_oov_words if total_oov_words else 0.0
    rate = total_oov_words / total_gold_words if total_gold_words else 0.0
    return {
        "oov_words": total_oov_words,
        "correct_oov_words": correct_oov_words,
        "oov_recall": round(recall, 4),
        "oov_rate": round(rate, 4),
    }


def collect_wrong_cases(raw_rows: list[str], gold_rows: list[list[str]], pred_rows: list[list[str]], limit: int = 50) -> list[dict]:
    errors = []
    for idx, (raw, gold, pred) in enumerate(zip(raw_rows, gold_rows, pred_rows), start=1):
        if _normalize_ellipsis_tokens(gold) == _normalize_ellipsis_tokens(pred):
            continue
        errors.append(
            {
                "line_no": idx,
                "raw_text": raw,
                "gold": SEGMENT_DELIMITER_PADDED.join(gold),
                "pred": SEGMENT_DELIMITER_PADDED.join(pred),
            }
        )
        if len(errors) >= limit:
            break
    return errors


def bucket_by_dataset(manifest_df, pred_rows: list[list[str]], gold_rows: list[list[str]]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for dataset, sub in manifest_df.groupby("dataset"):
        idxs = [int(i) for i in sub.index.tolist()]
        score = score_predictions([gold_rows[i] for i in idxs], [pred_rows[i] for i in idxs])
        results[dataset] = score.to_dict()
    return results


def bucket_by_difficulty(manifest_df, pred_rows: list[list[str]], gold_rows: list[list[str]]) -> dict[str, dict]:
    bins = {
        "high": (manifest_df["dataset"] != "samechar") & (manifest_df["difficulty_score"] >= 55),
        "medium": (manifest_df["dataset"] != "samechar") & (manifest_df["difficulty_score"] < 55),
        "specialized": manifest_df["dataset"].isin(["samechar"]),
    }
    results: dict[str, dict] = {}
    for name, mask in bins.items():
        idxs = [int(i) for i in manifest_df[mask].index.tolist()]
        if not idxs:
            continue
        score = score_predictions([gold_rows[i] for i in idxs], [pred_rows[i] for i in idxs])
        results[name] = score.to_dict()
    return results


def bucket_oov_by_dataset(
    manifest_df,
    pred_rows: list[list[str]],
    gold_rows: list[list[str]],
    train_vocab: set[str],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for dataset, sub in manifest_df.groupby("dataset"):
        idxs = [int(i) for i in sub.index.tolist()]
        results[dataset] = oov_recall(
            [gold_rows[i] for i in idxs],
            [pred_rows[i] for i in idxs],
            train_vocab,
        )
    return results

