from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eval_core import DEFAULT_SUBMISSION_GROUP, score_prediction_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score a pred.txt submission against hidden gold.')
    parser.add_argument('--submission', required=True, help='Path to pred.txt')
    parser.add_argument('--name', required=True, help='Displayed team or submission name')
    parser.add_argument('--submission-group', default=DEFAULT_SUBMISSION_GROUP, help='Displayed submission group label')
    parser.add_argument('--mode', default='prediction_file_only', choices=['prediction_file_only', 'executable_package'])
    parser.add_argument('--raw', default='test_assets/platform_eval_v2_draft/raw.txt')
    parser.add_argument('--gold', default='test_assets/platform_eval_v2_draft/gold.txt')
    parser.add_argument('--manifest', default='test_assets/platform_eval_v2_draft/gold_manifest.csv')
    parser.add_argument('--leaderboard', default='platform/results/leaderboard.csv')
    parser.add_argument('--reports-dir', default='platform/results/reports')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, board = score_prediction_submission(
        submission_path=args.submission,
        submission_name=args.name,
        submission_group=args.submission_group,
        mode=args.mode,
        raw_path=args.raw,
        gold_path=args.gold,
        manifest_path=args.manifest,
        leaderboard_path=args.leaderboard,
        reports_dir=args.reports_dir,
    )
    print(f'已更新排行榜：{Path(args.leaderboard)}')
    print(f'提交状态：{report["status"]}')
    print(board.to_string())


if __name__ == '__main__':
    main()

