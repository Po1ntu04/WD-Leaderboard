from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'my_platform' / 'results'


def main() -> None:
    board = pd.read_csv(RESULTS_DIR / 'leaderboard.csv', encoding='utf-8-sig')
    board = board.sort_values(['f1', 'timestamp'], ascending=[False, False]).drop_duplicates(subset=['submission_name'], keep='first').reset_index(drop=True)
    problems = pd.read_csv(RESULTS_DIR / 'problem_submissions.csv', encoding='utf-8-sig') if (RESULTS_DIR / 'problem_submissions.csv').exists() else pd.DataFrame()

    success = board[board['status'] == '成功'].copy()
    reference = board[board['submission_group'].astype(str).str.contains('工具|AI|tool|大模型', na=False)].copy()

    lines = [
        '# 课堂评测总结报告',
        '',
        f'- 总提交数：{len(board)}',
        f'- 成功提交：{int((board["status"] == "成功").sum())}',
        f'- 对比基线（工具/AI）：{len(reference)}',
        f'- 带问题提示提交：{len(problems)}',
        '',
        '## 一、榜单概况',
        '',
        '### 1.1 前十名',
        '',
        '| 排名 | 提交名 | F1 | 用时 |',
        '|---|---|---:|---:|',
    ]
    for _, row in success.head(10).iterrows():
        lines.append(f"| {int(row['rank']) if pd.notna(row['rank']) else ''} | {row['submission_name']} | {float(row['f1']):.4f} | {float(row['runtime_seconds']):.4f} |")

    lines += ['', '### 1.2 工具 / AI 对比', '', '| 提交名 | 分组 | F1 | NLPCC | EvaHan | TCM | samechar专项 |', '|---|---|---:|---:|---:|---:|---:|']
    if reference.empty:
        lines.append('| 暂无 | - | - | - | - | - | - |')
    else:
        for _, row in reference.sort_values('f1', ascending=False).iterrows():
            lines.append(
                f"| {row['submission_name']} | {row['submission_group']} | {float(row['f1']):.4f} | {float(row.get('NLPCC-Weibo_f1', 0) or 0):.4f} | {float(row.get('EvaHan-2022_f1', 0) or 0):.4f} | {float(row.get('TCM-Ancient-Books_f1', 0) or 0):.4f} | {float(row.get('samechar_f1', 0) or 0):.4f} |"
            )

    lines += ['', '## 二、问题提交清单', '', '| 提交名 | 问题类型 | 说明 | F1 |', '|---|---|---|---:|']
    if problems.empty:
        lines.append('| 暂无 | - | - | - |')
    else:
        for _, row in problems.iterrows():
            lines.append(f"| {row['submission_name']} | {row['problem_type']} | {str(row['message']).replace('|','/')} | {float(row['f1']):.4f} |")

    lines += [
        '',
        '## 三、补充说明',
        '',
        '- 当前评分逻辑已兼容最后一行 runtime 写法的常见变体，并对少量问题句采用“该句 0 分，其余句正常计分”的策略。',
        '- 前端可通过“加入 AI / 工具对比”开关决定是否将工具或大模型提交纳入排行榜显示。',
        '- `Seed Thinking` 为单行超长输出，系统已自动按句长重排后评测，并保留问题提示。',
        '',
    ]
    out = RESULTS_DIR / 'classroom_summary_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
