from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from algorithms.common.io import ensure_dir
from eval_core import DEFAULT_SUBMISSION_GROUP, score_prediction_submission


def append_runtime_footer(pred_path: Path, runtime_seconds: float) -> None:
    if not pred_path.exists():
        return
    text = pred_path.read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    footer = f'# runtime_seconds: {runtime_seconds:.6f}'
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().startswith('#'):
        lines[-1] = footer
    else:
        lines.append(footer)
    pred_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a submission package and score its generated pred.txt.')
    parser.add_argument('--submission-dir', required=True, help='Directory containing run.bat and model/code files.')
    parser.add_argument('--name', required=True, help='Displayed team or submission name')
    parser.add_argument('--submission-group', default=DEFAULT_SUBMISSION_GROUP, help='Displayed submission group label')
    parser.add_argument('--raw', default='test_assets/platform_eval_v2_draft/raw.txt')
    parser.add_argument('--gold', default='test_assets/platform_eval_v2_draft/gold.txt')
    parser.add_argument('--manifest', default='test_assets/platform_eval_v2_draft/gold_manifest.csv')
    parser.add_argument('--leaderboard', default='my_platform/results/leaderboard.csv')
    parser.add_argument('--reports-dir', default='my_platform/results/reports')
    parser.add_argument('--runtime-root', default='my_platform/runtime')
    parser.add_argument('--timeout-seconds', type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_dir = Path(args.submission_dir).resolve()
    launcher = submission_dir / 'run.bat'
    if not launcher.exists():
        report, board = score_prediction_submission(
            submission_path=submission_dir / 'output' / 'pred.txt',
            submission_name=args.name,
            submission_group=args.submission_group,
            mode='executable_package',
            raw_path=args.raw,
            gold_path=args.gold,
            manifest_path=args.manifest,
            leaderboard_path=args.leaderboard,
            reports_dir=args.reports_dir,
            execution_errors=['缺少 run.bat。'],
            failure_status='拒收',
        )
        print(f'提交失败：{report["status"]}')
        print(board.to_string())
        return

    safe_name = ''.join(ch if ch.isascii() and (ch.isalnum() or ch in '-_.') else '_' for ch in args.name).strip('_') or 'submission'
    runtime_root = ensure_dir(Path(args.runtime_root).resolve())
    job_dir = runtime_root / safe_name
    if job_dir.exists():
        shutil.rmtree(job_dir)
    input_dir = ensure_dir(job_dir / 'input')
    output_dir = ensure_dir(job_dir / 'output')

    input_raw = Path(args.raw).resolve()
    runtime_raw = (input_dir / 'raw.txt').resolve()
    shutil.copy2(input_raw, runtime_raw)
    runtime_pred = (output_dir / 'pred.txt').resolve()

    execution_errors: list[str] = []
    failure_status: str | None = None
    start = time.perf_counter()
    cmd_exe = shutil.which('cmd')
    if cmd_exe is None:
        runtime_seconds = 0.0
        failure_status = '运行错误'
        execution_errors.append('当前环境缺少 Windows cmd，无法执行 run.bat。请在 Windows 教师机上使用可执行包评分模式。')
    else:
        try:
            completed = subprocess.run(
                [cmd_exe, '/c', str(launcher), str(runtime_raw), str(runtime_pred)],
                cwd=submission_dir,
                timeout=args.timeout_seconds,
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            runtime_seconds = time.perf_counter() - start
            if completed.returncode != 0:
                failure_status = '运行错误'
                execution_errors.append(f'程序退出码非零：{completed.returncode}')
            if completed.stderr.strip():
                execution_errors.append(completed.stderr.strip()[:400])
        except subprocess.TimeoutExpired:
            runtime_seconds = time.perf_counter() - start
            failure_status = '超时'
            execution_errors.append(f'执行超过 {args.timeout_seconds} 秒。')
        except Exception as exc:
            runtime_seconds = time.perf_counter() - start
            failure_status = '运行错误'
            execution_errors.append(str(exc))

    if failure_status is None and not runtime_pred.exists():
        failure_status = '缺少输出'
        execution_errors.append('程序运行结束，但未生成 pred.txt。')
    elif failure_status is None and runtime_pred.exists():
        append_runtime_footer(runtime_pred, runtime_seconds)

    report, board = score_prediction_submission(
        submission_path=runtime_pred,
        submission_name=args.name,
        submission_group=args.submission_group,
        mode='executable_package',
        raw_path=args.raw,
        gold_path=args.gold,
        manifest_path=args.manifest,
        leaderboard_path=args.leaderboard,
        reports_dir=args.reports_dir,
        runtime_seconds=runtime_seconds,
        execution_errors=execution_errors,
        failure_status=failure_status,
    )
    print(f'已更新排行榜：{Path(args.leaderboard)}')
    print(f'提交状态：{report["status"]}')
    print(board.to_string())


if __name__ == '__main__':
    main()

