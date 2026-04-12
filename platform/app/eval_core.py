from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from algorithms.common.io import ensure_dir, parse_segmented_line, read_raw_file, read_segmented_file, write_json
from algorithms.common.scorer import bucket_by_dataset, bucket_by_difficulty, collect_wrong_cases, score_predictions


STATUS_WAITING = '等待中'
STATUS_RUNNING = '运行中'
STATUS_SUCCESS = '成功'
STATUS_FORMAT_ERROR = '格式错误'
STATUS_RUNTIME_ERROR = '运行错误'
STATUS_TIMEOUT = '超时'
STATUS_MISSING_OUTPUT = '缺少输出'
STATUS_DUPLICATE = '重复提交'
STATUS_REJECTED = '拒收'

DEFAULT_SUBMISSION_GROUP = '课堂提交'
MAX_SUBMISSION_BYTES = 2 * 1024 * 1024
MAX_PREDICTION_LINE_CHARS = 10000
RUNTIME_PATTERN = re.compile(r'^#\s*runtime_seconds\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$')

LEADERBOARD_COLUMNS = [
    'rank',
    'submission_name',
    'submission_group',
    'mode',
    'status',
    'timestamp',
    'runtime_seconds',
    'precision',
    'recall',
    'f1',
    'wrong_sentence_count',
    'message',
    'NLPCC-Weibo_f1',
    'EvaHan-2022_f1',
    'TCM-Ancient-Books_f1',
    'samechar_f1',
    'high_f1',
    'medium_f1',
    'specialized_f1',
]


def _contains_disallowed_control_chars(text: str) -> bool:
    for ch in text:
        if ch in {'\n', '\r', '\t'}:
            continue
        if ord(ch) < 32:
            return True
    return False


def load_prediction_submission(submission_path: str | Path) -> tuple[list[list[str]], list[str], float | None]:
    path = Path(submission_path)
    errors: list[str] = []
    if not path.exists():
        return [], ['未找到输出文件 pred.txt。'], None

    if path.stat().st_size > MAX_SUBMISSION_BYTES:
        errors.append(f'提交文件过大：超过 {MAX_SUBMISSION_BYTES // 1024} KB。')
        return [], errors, None

    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        return [], ['提交文件必须是 UTF-8 编码文本。'], None

    if _contains_disallowed_control_chars(text):
        errors.append('提交文件包含非法控制字符。')

    lines = text.splitlines()
    last_nonempty_idx = -1
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            last_nonempty_idx = idx
            break

    runtime_seconds: float | None = None
    runtime_line_idx: int | None = None
    if last_nonempty_idx >= 0:
        tail = lines[last_nonempty_idx].strip()
        if tail.startswith('#'):
            match = RUNTIME_PATTERN.match(tail)
            if match:
                runtime_seconds = float(match.group(1))
                runtime_line_idx = last_nonempty_idx
            else:
                errors.append('最后一行元信息格式错误，应为 # runtime_seconds: 数值')

    pred_rows: list[list[str]] = []
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if len(raw_line) > MAX_PREDICTION_LINE_CHARS:
            errors.append(f'第 {line_no} 行过长，请检查是否误写了异常内容。')
            if len(errors) >= 20:
                break
            continue
        if stripped.startswith('#'):
            if runtime_line_idx is not None and line_no - 1 == runtime_line_idx:
                continue
            errors.append('元信息行只能放在最后一行，且格式为 # runtime_seconds: 数值')
            continue
        pred_rows.append(parse_segmented_line(raw_line))

    return pred_rows, errors, runtime_seconds


def validate_prediction_lines(raw_rows: list[str], pred_rows: list[list[str]]) -> list[str]:
    errors: list[str] = []
    if len(raw_rows) != len(pred_rows):
        errors.append(f'行数不匹配：raw={len(raw_rows)} pred={len(pred_rows)}')
        return errors
    for idx, (raw, pred) in enumerate(zip(raw_rows, pred_rows), start=1):
        if ''.join(pred) != raw:
            errors.append(f'第 {idx} 行分词结果无法还原原句。')
            if len(errors) >= 20:
                break
    return errors


def first_message(validation_errors: list[str], eval_warnings: list[str]) -> str:
    if validation_errors:
        return str(validation_errors[0])
    if eval_warnings:
        return str(eval_warnings[0])
    return ''


def build_score_payload(
    raw_rows: list[str],
    gold_rows: list[list[str]],
    pred_rows: list[list[str]],
    manifest: pd.DataFrame,
    *,
    submission_name: str,
    submission_group: str,
    submission_path: str,
    mode: str,
    status: str,
    timestamp: str,
    validation_errors: list[str],
    runtime_seconds: float | None = None,
) -> tuple[dict, dict]:
    eval_warnings: list[str] = []
    by_dataset: dict[str, dict] = {}
    by_difficulty: dict[str, dict] = {}
    wrong_cases: list[dict] = []

    if status == STATUS_SUCCESS:
        overall = score_predictions(gold_rows, pred_rows).to_dict()
        by_dataset = bucket_by_dataset(manifest, pred_rows, gold_rows)
        by_difficulty = bucket_by_difficulty(manifest, pred_rows, gold_rows)
        wrong_cases = collect_wrong_cases(raw_rows, gold_rows, pred_rows)
    else:
        overall = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'gold_words': 0,
            'pred_words': 0,
            'correct_words': 0,
            'exact_match_sentences': 0,
            'total_sentences': len(raw_rows),
        }

    wrong_sentence_count = int(overall.get('total_sentences', 0) - overall.get('exact_match_sentences', 0)) if status == STATUS_SUCCESS else 0
    message = first_message(validation_errors, eval_warnings)

    report = {
        'submission_name': submission_name,
        'submission_group': submission_group,
        'submission_path': submission_path,
        'mode': mode,
        'status': status,
        'timestamp': timestamp,
        'runtime_seconds': round(float(runtime_seconds or 0.0), 6),
        'overall': overall,
        'wrong_sentence_count': wrong_sentence_count,
        'message': message,
        'by_dataset': by_dataset,
        'by_difficulty': by_difficulty,
        'validation_errors': validation_errors,
        'eval_warnings': eval_warnings,
        'wrong_cases': wrong_cases[:20],
    }

    row = {
        'submission_name': submission_name,
        'submission_group': submission_group,
        'mode': mode,
        'status': status,
        'timestamp': timestamp,
        'runtime_seconds': round(float(runtime_seconds or 0.0), 6),
        'precision': overall.get('precision', 0.0),
        'recall': overall.get('recall', 0.0),
        'f1': overall.get('f1', 0.0),
        'wrong_sentence_count': wrong_sentence_count,
        'message': message,
        'NLPCC-Weibo_f1': by_dataset.get('NLPCC-Weibo', {}).get('f1'),
        'EvaHan-2022_f1': by_dataset.get('EvaHan-2022', {}).get('f1'),
        'TCM-Ancient-Books_f1': by_dataset.get('TCM-Ancient-Books', {}).get('f1'),
        'samechar_f1': by_dataset.get('samechar', {}).get('f1'),
        'high_f1': by_difficulty.get('high', {}).get('f1', 0.0),
        'medium_f1': by_difficulty.get('medium', {}).get('f1', 0.0),
        'specialized_f1': by_difficulty.get('specialized', {}).get('f1', 0.0),
    }
    return report, row


def write_report(report: dict, reports_dir: str | Path) -> Path:
    report_dir = ensure_dir(reports_dir)
    safe_name = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in report['submission_name'])
    report_path = report_dir / f'{safe_name}.report.json'
    write_json(report_path, report)
    return report_path


def _empty_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(columns=LEADERBOARD_COLUMNS[1:])


def update_leaderboard(leaderboard_path: str | Path, row: dict) -> pd.DataFrame:
    leaderboard_path = Path(leaderboard_path)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    if leaderboard_path.exists():
        board = pd.read_csv(leaderboard_path)
        if 'rank' in board.columns:
            board = board.drop(columns=['rank'])
    else:
        board = _empty_leaderboard()

    records = board.to_dict(orient='records')
    records.append(row)
    board = pd.DataFrame(records)

    for column in LEADERBOARD_COLUMNS[1:]:
        if column not in board.columns:
            board[column] = None
    board = board[LEADERBOARD_COLUMNS[1:]]

    board['_status_order'] = board['status'].map(lambda value: 0 if value == STATUS_SUCCESS else 1)
    board = board.sort_values(['_status_order', 'f1', 'runtime_seconds', 'timestamp'], ascending=[True, False, True, True], na_position='last').drop(columns=['_status_order']).reset_index(drop=True)
    board.insert(0, 'rank', range(1, len(board) + 1))
    board.to_csv(leaderboard_path, index=False, encoding='utf-8-sig')
    return board


def score_prediction_submission(
    *,
    submission_path: str | Path,
    submission_name: str,
    submission_group: str = DEFAULT_SUBMISSION_GROUP,
    mode: str,
    raw_path: str | Path,
    gold_path: str | Path,
    manifest_path: str | Path,
    leaderboard_path: str | Path,
    reports_dir: str | Path,
    runtime_seconds: float | None = None,
    execution_errors: list[str] | None = None,
    failure_status: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    raw_rows = read_raw_file(raw_path)
    gold_rows = read_segmented_file(gold_path)
    manifest = pd.read_csv(manifest_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    submission_path = Path(submission_path)
    pred_rows, parse_errors, runtime_from_file = load_prediction_submission(submission_path)
    errors = list(execution_errors or [])
    errors.extend(parse_errors)
    effective_runtime = runtime_seconds if runtime_seconds is not None else runtime_from_file
    if submission_path.exists() and not errors:
        errors.extend(validate_prediction_lines(raw_rows, pred_rows))

    status = STATUS_SUCCESS if not errors else (failure_status or STATUS_FORMAT_ERROR)
    report, row = build_score_payload(
        raw_rows,
        gold_rows,
        pred_rows,
        manifest,
        submission_name=submission_name,
        submission_group=submission_group,
        submission_path=str(submission_path.resolve()).replace('\\', '/'),
        mode=mode,
        status=status,
        timestamp=timestamp,
        validation_errors=errors,
        runtime_seconds=effective_runtime,
    )
    write_report(report, reports_dir)
    board = update_leaderboard(leaderboard_path, row)
    return report, board
