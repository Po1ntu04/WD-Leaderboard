from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRED_FILENAME_RE = re.compile(r'^(?P<student_id>[^_]+)_(?P<name>.+)_pred\.txt$', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a teacher-side local session over prediction and executable submissions.')
    parser.add_argument('--session-name', default='demo_session')
    parser.add_argument('--prediction-dir', default='')
    parser.add_argument('--executable-dir', default='')
    parser.add_argument('--raw', default='test_assets/platform_eval_v2_draft/raw.txt')
    parser.add_argument('--gold', default='test_assets/platform_eval_v2_draft/gold.txt')
    parser.add_argument('--manifest', default='test_assets/platform_eval_v2_draft/gold_manifest.csv')
    parser.add_argument('--results-dir', default='my_platform/results')
    parser.add_argument('--timeout-seconds', type=int, default=120)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def clear_results(results_dir: Path) -> None:
    for name in ['leaderboard.csv', 'index.html', 'session_summary.json']:
        path = results_dir / name
        if path.exists():
            path.unlink()
    for child in ['reports', 'details']:
        path = results_dir / child
        if path.exists():
            shutil.rmtree(path)


def prediction_files(prediction_dir: Path) -> list[Path]:
    candidates = sorted(path for path in prediction_dir.glob('*.txt') if path.is_file())
    ignored_names = {'raw.txt', 'gold.txt'}
    return [path for path in candidates if path.name not in ignored_names]


def display_name_from_prediction_file(pred_file: Path) -> str:
    match = PRED_FILENAME_RE.match(pred_file.name)
    if match:
        return f"{match.group('student_id')}_{match.group('name')}"
    return pred_file.stem.replace('.pred', '')


def score_prediction_files(prediction_dir: Path, args: argparse.Namespace, results_dir: Path) -> list[dict]:
    scored: list[dict] = []
    for pred_file in prediction_files(prediction_dir):
        name = display_name_from_prediction_file(pred_file)
        run_cmd([
            sys.executable,
            str(Path(__file__).resolve().parent.parent / 'app' / 'score_submission.py'),
            '--submission', str(pred_file),
            '--name', name,
            '--raw', args.raw,
            '--gold', args.gold,
            '--manifest', args.manifest,
            '--leaderboard', str(results_dir / 'leaderboard.csv'),
            '--reports-dir', str(results_dir / 'reports'),
        ])
        scored.append({'name': name, 'mode': 'prediction_file_only', 'path': str(pred_file)})
    return scored


def score_executable_dirs(executable_dir: Path, args: argparse.Namespace, results_dir: Path) -> list[dict]:
    scored: list[dict] = []
    for submission_dir in sorted(path for path in executable_dir.iterdir() if path.is_dir()):
        name = submission_dir.name
        run_cmd([
            sys.executable,
            str(Path(__file__).resolve().parent.parent / 'app' / 'score_executable_submission.py'),
            '--submission-dir', str(submission_dir),
            '--name', name,
            '--raw', args.raw,
            '--gold', args.gold,
            '--manifest', args.manifest,
            '--leaderboard', str(results_dir / 'leaderboard.csv'),
            '--reports-dir', str(results_dir / 'reports'),
            '--timeout-seconds', str(args.timeout_seconds),
        ])
        scored.append({'name': name, 'mode': 'executable_package', 'path': str(submission_dir)})
    return scored


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    clear_results(results_dir)

    scored_items: list[dict] = []
    if args.prediction_dir:
        prediction_dir = Path(args.prediction_dir).resolve()
        if prediction_dir.exists():
            scored_items.extend(score_prediction_files(prediction_dir, args, results_dir))
    if args.executable_dir:
        executable_dir = Path(args.executable_dir).resolve()
        if executable_dir.exists():
            scored_items.extend(score_executable_dirs(executable_dir, args, results_dir))

    leaderboard = pd.read_csv(results_dir / 'leaderboard.csv')
    session_summary = {
        'session_name': args.session_name,
        'dataset_version': Path(args.raw).resolve().parent.name,
        'scoring_rule_version': 'benchmark_v2',
        'submission_protocol_version': 'submission_v1',
        'submission_count': int(len(leaderboard)),
        'success_count': int((leaderboard['status'] == '成功').sum()),
        'failure_count': int((leaderboard['status'] != '成功').sum()),
        'prediction_mode_count': int((leaderboard['mode'] == 'prediction_file_only').sum()),
        'executable_mode_count': int((leaderboard['mode'] == 'executable_package').sum()),
        'results_dir': str(results_dir.resolve()).replace('\\', '/'),
        'raw_path': str(Path(args.raw).resolve()).replace('\\', '/'),
        'gold_path': str(Path(args.gold).resolve()).replace('\\', '/'),
        'manifest_path': str(Path(args.manifest).resolve()).replace('\\', '/'),
        'scored_items': scored_items,
    }
    session_summary_path = results_dir / 'session_summary.json'
    session_summary_path.write_text(json.dumps(session_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    package_meta = Path(args.raw).resolve().parent / 'package_meta.json'
    build_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / 'app' / 'build_demo_page.py'),
        '--leaderboard', str(results_dir / 'leaderboard.csv'),
        '--reports-dir', str(results_dir / 'reports'),
        '--output', str(results_dir / 'index.html'),
        '--session-summary', str(session_summary_path),
    ]
    if package_meta.exists():
        build_cmd.extend(['--package-meta', str(package_meta)])
    run_cmd(build_cmd)

    print(json.dumps(session_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

