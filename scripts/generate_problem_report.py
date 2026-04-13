from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'my_platform' / 'results'


def classify(message: str) -> str:
    text = str(message or '')
    if '行数不匹配' in text:
        return '行数问题'
    if '额外输出' in text:
        return '额外输出'
    if '无法还原原句' in text:
        return '原句还原问题'
    if 'runtime_seconds' in text:
        return 'runtime 元信息问题'
    return '其他'


def main() -> None:
    board = pd.read_csv(RESULTS_DIR / 'leaderboard.csv', encoding='utf-8-sig')
    problems = board[board['message'].fillna('') != ''].copy()
    problems['problem_type'] = problems['message'].map(classify)
    out_csv = RESULTS_DIR / 'problem_submissions.csv'
    out_md = RESULTS_DIR / 'problem_submissions.md'
    problems[['submission_name', 'status', 'problem_type', 'message', 'f1', 'runtime_seconds']].to_csv(out_csv, index=False, encoding='utf-8-sig')

    lines = [
        '# 问题提交清单',
        '',
        f'- 总提交数：{len(board)}',
        f'- 有问题提交数：{len(problems)}',
        '',
        '| 提交名 | 状态 | 问题类型 | 说明 | F1 | 时间 |',
        '|---|---|---|---|---:|---:|',
    ]
    for _, row in problems.iterrows():
        lines.append(
            f"| {row['submission_name']} | {row['status']} | {row['problem_type']} | {str(row['message']).replace('|','/')} | {float(row['f1']):.4f} | {float(row['runtime_seconds']):.4f} |"
        )
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out_csv)
    print(out_md)


if __name__ == '__main__':
    main()
