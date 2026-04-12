from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

import pandas as pd


SORTS = [
    ('rank', '综合排名'),
    ('f1', '总体 F1'),
    ('runtime', '运行时间'),
    ('latest', '最新提交'),
    ('nlpcc', 'NLPCC-Weibo 排名'),
    ('evahan', 'EvaHan 排名'),
    ('tcm', 'TCM 古籍排名'),
    ('samechar', 'samechar 排名'),
    ('high', '高难层排名'),
    ('specialized', '专项层排名'),
]
STATUS_CLASS = {
    '成功': 'ok',
    '等待中': 'warn',
    '运行中': 'warn',
    '重复提交': 'warn',
    '格式错误': 'bad',
    '运行错误': 'bad',
    '超时': 'bad',
    '缺少输出': 'bad',
    '拒收': 'bad',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build the static classroom leaderboard page.')
    parser.add_argument('--leaderboard', default='my_platform/results/leaderboard.csv')
    parser.add_argument('--reports-dir', default='my_platform/results/reports')
    parser.add_argument('--output', default='my_platform/results/index.html')
    parser.add_argument('--session-summary', default='')
    parser.add_argument('--package-meta', default='')
    return parser.parse_args()


def load_json(path: str | Path | None) -> dict:
    target = Path(path) if path else None
    if not target or not target.exists():
        return {}
    return json.loads(target.read_text(encoding='utf-8'))


def posix(value: object) -> str:
    return str(value or '').replace('\\', '/').strip()


def metric(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return 'N/A'
    return 'N/A' if pd.isna(number) else f'{number:.{digits}f}'


def status_class(status: str) -> str:
    return STATUS_CLASS.get(status, 'muted')


def manifest_rows(summary: dict) -> dict[int, dict]:
    manifest_path = summary.get('manifest_path', '')
    target = Path(str(manifest_path).replace('/', '\\')) if manifest_path else None
    if not target or not target.exists():
        return {}
    rows: dict[int, dict] = {}
    with target.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                rows[int(row.get('line_no', '0'))] = row
            except Exception:
                continue
    return rows


def han_count(text: str) -> int:
    return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')


def tokenish_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum() or ('\u4e00' <= ch <= '\u9fff'))


def worthy_case(case: dict, manifest_row: dict | None) -> bool:
    raw = str(case.get('raw_text', '')).strip()
    gold = str(case.get('gold', '')).strip()
    if not raw or not gold:
        return False
    tokenish = tokenish_count(raw)
    han = han_count(raw)
    ratio = han / tokenish if tokenish else 0.0
    if han < 4 or ratio < 0.55:
        return False
    dataset = str((manifest_row or {}).get('dataset', ''))
    if dataset == 'samechar':
        return True
    tags = str((manifest_row or {}).get('selection_tags', ''))
    if 'social_marker' in tags and 'mixed_script' in tags and ratio < 0.8:
        return False
    return True


def load_reports(reports_dir: Path) -> list[dict]:
    reports: list[dict] = []
    if not reports_dir.exists():
        return reports
    for path in sorted(reports_dir.glob('*.report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        report['_file'] = posix(path.resolve())
        reports.append(report)
    return reports


def collect_common_failures(reports: list[dict], manifest_lookup: dict[int, dict]) -> list[dict]:
    aggregate: dict[str, dict] = {}
    for report in reports:
        if report.get('status') != '成功':
            continue
        submission_name = str(report.get('submission_name', ''))
        for case in report.get('wrong_cases', []):
            line_no = int(case.get('line_no', 0) or 0)
            manifest_row = manifest_lookup.get(line_no)
            if not worthy_case(case, manifest_row):
                continue
            raw = str(case.get('raw_text', '')).strip()
            if not raw:
                continue
            item = aggregate.setdefault(
                raw,
                {
                    'line_no': line_no,
                    'raw_text': raw,
                    'gold': str(case.get('gold', '')).strip(),
                    'dataset': str((manifest_row or {}).get('dataset', '')),
                    'sample_id': str((manifest_row or {}).get('sample_id', '')),
                    'count': 0,
                    'names': set(),
                },
            )
            if submission_name not in item['names']:
                item['names'].add(submission_name)
                item['count'] += 1

    rows = []
    for item in aggregate.values():
        rows.append(
            {
                'line_no': item['line_no'],
                'raw_text': item['raw_text'],
                'gold': item['gold'],
                'dataset': item['dataset'],
                'sample_id': item['sample_id'],
                'count': item['count'],
                'names': sorted(item['names']),
            }
        )
    rows.sort(key=lambda row: (-row['count'], len(row['raw_text']), row['raw_text']))
    return rows[:10]


def render_common_failures(items: list[dict]) -> str:
    if not items:
        return "<p class='empty'>当前没有适合课堂展示的共性错例。</p>"
    rows = []
    for item in items:
        names = '、'.join(item['names'][:5]) + (' 等' if len(item['names']) > 5 else '')
        source = ' / '.join([x for x in [item.get('dataset', ''), item.get('sample_id', '')] if x]) or '-'
        rows.append(
            f"<tr><td class='src'>{escape(source)}</td><td>{escape(item['raw_text'])}</td><td>{escape(item['gold'])}</td><td>{item['count']}</td><td>{escape(names)}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table class='compact-table'><thead><tr>"
        "<th>来源</th><th>原句</th><th>标准切分</th><th>失败提交数</th><th>涉及提交</th>"
        "</tr></thead><tbody>"
        + ''.join(rows)
        + "</tbody></table></div>"
        "<p class='hint'>仅保留适合课堂讨论的中文主体样本。</p>"
    )


def render_package_table(package_meta: dict) -> str:
    datasets = package_meta.get('datasets', {}) or {}
    source_policy = package_meta.get('source_policy', {}) or {}
    if not datasets:
        return "<p class='muted'>暂无数据包说明。</p>"
    rows = []
    for dataset, count in datasets.items():
        rows.append(
            f"<tr><td>{escape(str(dataset))}</td><td>{escape(str(count))}</td><td>{escape(str(source_policy.get(dataset, '')))}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table class='compact-table'><thead><tr>"
        "<th>数据桶</th><th>句数</th><th>选取策略</th>"
        "</tr></thead><tbody>"
        + ''.join(rows)
        + "</tbody></table></div>"
    )


def render_teacher_panel(summary: dict) -> str:
    raw = escape(posix(summary.get('raw_path', 'test_assets/platform_eval_v2_draft/platform_eval_v2_draft/raw.txt')))
    gold = escape(posix(summary.get('gold_path', 'test_assets/platform_eval_v2_draft/platform_eval_v2_draft/gold.txt')))
    manifest = escape(posix(summary.get('manifest_path', 'test_assets/platform_eval_v2_draft/platform_eval_v2_draft/gold_manifest.csv')))
    results = escape(posix(summary.get('results_dir', 'my_platform/results')))
    python_exe = escape(posix(summary.get('benchmark_python', 'python')))
    toolkit_text = '、'.join(summary.get('available_toolkits', []) or []) or '当前未探测到可用 toolkit'
    return f"""
<section class='panel'>
  <div class='panel-head'>
    <h2>教师会话台</h2>
    <div class='sub'>统一固定课堂常用命令、路径和提交模式。</div>
  </div>
  <div class='card-grid'>
    <div class='card'>
      <div class='card-title'>1. 单个 pred.txt 评测</div>
      <pre>python platform/app/score_submission.py --submission path/to/pred.txt --name 学生提交名 --submission-group 课堂提交 --raw {raw} --gold {gold} --manifest {manifest}</pre>
      <p class='hint'>若写运行时间，请放在 <code>pred.txt</code> 最后一行：<code># runtime_seconds: 0.183</code></p>
    </div>
    <div class='card'>
      <div class='card-title'>2. 可执行包评测</div>
      <pre>python platform/app/score_executable_submission.py --submission-dir path/to/submission_dir --name 学生提交名 --submission-group 课堂提交 --raw {raw} --gold {gold} --manifest {manifest}</pre>
    </div>
    <div class='card'>
      <div class='card-title'>3. 批量会话运行</div>
      <pre>python platform/app/run_session.py --session-name class_session --prediction-dir path/to/predictions --executable-dir path/to/submissions --raw {raw} --gold {gold} --manifest {manifest}</pre>
    </div>
    <div class='card'>
      <div class='card-title'>4. 本地展示服务</div>
      <pre>python -m http.server 8765 --directory {results}</pre>
      <p class='hint'>浏览器地址：<code>http://127.0.0.1:8765/index.html</code></p>
    </div>
  </div>
  <div class='mini-grid'>
    <div class='mini-card'><strong>Benchmark Python</strong><br /><code>{python_exe}</code></div>
    <div class='mini-card'><strong>已探测 toolkit</strong><br /><span class='muted'>{escape(toolkit_text)}</span></div>
    <div class='mini-card'><strong>结果目录</strong><br /><code>{results}</code></div>
  </div>
</section>
"""


def render_overview_cards(board: pd.DataFrame) -> str:
    success = board[board['status'] == '成功'].copy()
    if success.empty:
        return "<p class='empty'>当前还没有成功提交。</p>"

    def pick(title: str, column: str, fastest: bool = False) -> str:
        ranked = success.copy()
        ranked[column] = pd.to_numeric(ranked[column], errors='coerce')
        ranked['runtime_seconds'] = pd.to_numeric(ranked['runtime_seconds'], errors='coerce').fillna(10**9)
        ranked = ranked.dropna(subset=[column])
        if ranked.empty:
            return ''
        if fastest:
            row = ranked.sort_values(by=['runtime_seconds', column], ascending=[True, False]).iloc[0]
        else:
            row = ranked.sort_values(by=[column, 'runtime_seconds'], ascending=[False, True]).iloc[0]
        return (
            "<div class='mini-card'>"
            f"<div class='meta'>{escape(title)}</div>"
            f"<div class='name'>{escape(str(row['submission_name']))}</div>"
            f"<div class='score'>F1 {metric(row.get(column))}</div>"
            f"<div class='meta'>时间 {metric(row.get('runtime_seconds'))} s</div>"
            "</div>"
        )

    cards = [
        pick('总榜最佳', 'f1'),
        pick('执行模式最佳', 'f1') if 'mode' in success.columns and not success[success['mode'] == 'executable_package'].empty else '',
        pick('最快成功提交', 'f1', fastest=True),
        pick('NLPCC 最佳', 'NLPCC-Weibo_f1'),
        pick('EvaHan 最佳', 'EvaHan-2022_f1'),
        pick('TCM 古籍最佳', 'TCM-Ancient-Books_f1'),
        pick('samechar 最佳', 'samechar_f1'),
        pick('高难层最佳', 'high_f1'),
        pick('专项层最佳', 'specialized_f1'),
    ]
    return ''.join(card for card in cards if card)


def render_dataset_spotlights(board: pd.DataFrame) -> str:
    success = board[board['status'] == '成功'].copy()
    if success.empty:
        return "<p class='empty'>当前没有可展示的成功提交。</p>"

    items = [
        ('总体最优', 'f1'),
        ('NLPCC', 'NLPCC-Weibo_f1'),
        ('EvaHan', 'EvaHan-2022_f1'),
        ('TCM', 'TCM-Ancient-Books_f1'),
        ('samechar', 'samechar_f1'),
        ('高难层', 'high_f1'),
    ]
    cards = []
    for title, column in items:
        ranked = success.copy()
        ranked[column] = pd.to_numeric(ranked[column], errors='coerce')
        ranked['runtime_seconds'] = pd.to_numeric(ranked['runtime_seconds'], errors='coerce').fillna(10**9)
        ranked = ranked.dropna(subset=[column])
        if ranked.empty:
            continue
        row = ranked.sort_values(by=[column, 'runtime_seconds'], ascending=[False, True]).iloc[0]
        value = float(row[column])
        width = max(6, min(100, round(value * 100)))
        cards.append(
            "<div class='spot-card'>"
            f"<div class='meta'>{escape(title)}</div>"
            f"<div class='name'>{escape(str(row['submission_name']))}</div>"
            f"<div class='spot-score'>{value:.4f}</div>"
            f"<div class='bar'><span style='width:{width}%'></span></div>"
            f"<div class='meta'>时间 {metric(row.get('runtime_seconds'))} s</div>"
            "</div>"
        )
    return ''.join(cards)


def render_failure_table(board: pd.DataFrame) -> str:
    failures = board[board['status'] != '成功'].copy()
    if failures.empty:
        return "<p class='empty'>当前没有失败提交。</p>"
    rows = []
    for _, row in failures.sort_values(by='timestamp', ascending=False).iterrows():
        rows.append(
            f"<tr><td>{escape(str(row.get('submission_name', '')))}</td>"
            f"<td>{escape(str(row.get('submission_group', '')))}</td>"
            f"<td><span class='{status_class(str(row.get('status', '')))}'>{escape(str(row.get('status', '')))}</span></td>"
            f"<td class='reason'>{escape(str(row.get('message', '')))}</td>"
            f"<td>{escape(str(row.get('timestamp', '')))}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table class='compact-table'><thead><tr>"
        "<th>提交名</th><th>分组</th><th>状态</th><th>原因摘要</th><th>时间</th>"
        "</tr></thead><tbody>"
        + ''.join(rows)
        + "</tbody></table></div>"
    )


def write_detail_pages(reports: list[dict], detail_dir: Path) -> dict[str, str]:
    detail_dir.mkdir(parents=True, exist_ok=True)
    for path in detail_dir.glob('*.html'):
        path.unlink()
    links: dict[str, str] = {}
    for idx, report in enumerate(reports, start=1):
        slug = f'submission_{idx:03d}.html'
        links[str(report.get('submission_name', slug))] = f'details/{slug}'
        wrong_rows = ''.join(
            f"<tr><td>{int(case.get('line_no', 0) or 0)}</td><td>{escape(str(case.get('raw_text', '')))}</td>"
            f"<td>{escape(str(case.get('gold', '')))}</td><td>{escape(str(case.get('pred', '')))}</td></tr>"
            for case in (report.get('wrong_cases', []) or [])
        )
        html = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>{escape(str(report.get('submission_name', '提交详情')))}</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f7f3ed; color: #1f1b18; font-family: 'Source Han Sans SC','Noto Sans SC','Microsoft YaHei',sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; }}
    .panel {{ background: rgba(255,252,247,.96); border: 1px solid #ddd0be; border-radius: 18px; box-shadow: 0 12px 28px rgba(88,58,28,.08); padding: 18px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #ddd0be; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ background: #ead8c5; }}
    .table-wrap {{ overflow: auto; max-height: 560px; }}
  </style>
</head>
<body>
  <main>
    <div class='panel'>
      <h1 style='margin:0 0 10px'>{escape(str(report.get('submission_name', '提交详情')))}</h1>
      <p><a href='../index.html'>返回首页</a></p>
      <p>状态：<strong>{escape(str(report.get('status', '')))}</strong>；总体 F1：{metric((report.get('overall') or {}).get('f1'))}；运行时间：{metric(report.get('runtime_seconds'))} s；错句数：{int(report.get('wrong_sentence_count', 0) or 0)}</p>
      <p><code>{escape(posix(report.get('submission_path', '')))}</code></p>
    </div>
    <div class='panel'>
      <h2 style='margin:0'>错例样本</h2>
      <div class='table-wrap'>
        <table>
          <thead><tr><th>行号</th><th>原句</th><th>标准</th><th>预测</th></tr></thead>
          <tbody>{wrong_rows}</tbody>
        </table>
      </div>
    </div>
  </main>
</body>
</html>
"""
        (detail_dir / slug).write_text(html, encoding='utf-8')
    return links


def render_page(board: pd.DataFrame, reports: list[dict], links: dict[str, str], summary: dict, package_meta: dict, manifest_lookup: dict[int, dict]) -> str:
    success = board[board['status'] == '成功'].copy()
    highest_f1 = metric(pd.to_numeric(success['f1'], errors='coerce').max()) if not success.empty and 'f1' in success else 'N/A'
    group_badges = ''.join(
        f"<div class='pill'>{escape(str(name))}：{int(count)}</div>"
        for name, count in (board['submission_group'].fillna('未分组').value_counts().to_dict() if not board.empty and 'submission_group' in board else {}).items()
    )
    sort_buttons = ''.join(f"<button class='pill sort-btn' type='button' data-sort='{key}'>{label}</button>" for key, label in SORTS)
    sort_options = ''.join(f"<option value='{key}'>{label}</option>" for key, label in SORTS)
    common_failures = render_common_failures(collect_common_failures(reports, manifest_lookup))
    package_table = render_package_table(package_meta)
    failure_table = render_failure_table(board)
    overview_cards = render_overview_cards(board)
    spotlight_cards = render_dataset_spotlights(board)

    board_view = board.copy()
    board_view['detail_link'] = board_view['submission_name'].map(links).fillna('')
    board_json = board_view.to_json(orient='records', force_ascii=False)

    return f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>中文分词课堂平台</title>
  <style>
    :root {{
      --bg: #0b1220;
      --paper: rgba(15, 23, 42, .92);
      --paper-soft: rgba(20, 31, 56, .92);
      --ink: #e6eefc;
      --muted: #98a8ca;
      --line: rgba(93, 126, 182, .28);
      --accent: #4da3ff;
      --accent-2: #7dd3fc;
      --soft: rgba(24, 38, 68, .95);
      --ok: #34d399;
      --bad: #fb7185;
      --warn: #fbbf24;
      --shadow: 0 20px 50px rgba(2, 8, 23, .35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: 'Source Han Sans SC','Noto Sans SC','Microsoft YaHei',sans-serif;
      background:
        radial-gradient(circle at top left, rgba(77,163,255,.18), transparent 22%),
        radial-gradient(circle at 82% 0%, rgba(125,211,252,.13), transparent 22%),
        linear-gradient(180deg, #09111f 0%, #0f172a 100%);
    }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 28px 20px 44px; }}
    .hero, .panel {{ background: var(--paper); border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow); backdrop-filter: blur(10px); }}
    .hero {{ padding: 26px; }}
    .hero-top {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; flex-wrap: wrap; }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 1px; }}
    h2 {{ margin: 0; font-size: 22px; }}
    .lead {{ margin: 10px 0 0; color: var(--muted); max-width: 920px; line-height: 1.8; }}
    .pill-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{ background: rgba(20,31,56,.82); border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; font-size: 13px; color: var(--muted); white-space: nowrap; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }}
    .stat {{ background: linear-gradient(180deg, rgba(21,32,58,.96), rgba(14,23,41,.92)); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }}
    .stat-value {{ font-size: 30px; font-weight: 800; color: var(--accent-2); }}
    .stat-label {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .panel {{ margin-top: 18px; padding: 18px; }}
    .panel-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .sub, .hint {{ color: var(--muted); font-size: 13px; line-height: 1.7; }}
    .split-grid {{ display: grid; grid-template-columns: 1.35fr .9fr; gap: 18px; }}
    .double-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 14px; }}
    .card, .mini-card, .spot-card {{ background: linear-gradient(180deg, rgba(20,31,56,.94), rgba(14,23,41,.94)); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }}
    .card-title {{ font-weight: 700; margin-bottom: 8px; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .name {{ margin-top: 8px; font-weight: 700; font-size: 16px; }}
    .score {{ margin-top: 6px; color: var(--accent); font-weight: 700; }}
    .spot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }}
    .spot-score {{ margin-top: 8px; font-size: 24px; font-weight: 800; color: var(--accent-2); }}
    .bar {{ margin-top: 10px; height: 8px; border-radius: 999px; background: rgba(77,163,255,.12); overflow: hidden; }}
    .bar span {{ display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .filter-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 14px; }}
    .field {{ display: flex; flex-direction: column; gap: 6px; }}
    .field label {{ color: var(--muted); font-size: 12px; }}
    .field input, .field select {{ padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(9,17,31,.88); color: var(--ink); }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: Consolas, monospace; font-size: 12px; line-height: 1.6; color: #dbe7ff; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; color: #dbe7ff; }}
    .table-wrap {{ overflow: auto; border-radius: 16px; border: 1px solid var(--line); background: rgba(9,17,31,.68); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 14px; table-layout: fixed; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; line-height: 1.65; overflow-wrap: anywhere; }}
    th {{ background: var(--soft); position: sticky; top: 0; z-index: 1; }}
    tbody tr:hover {{ background: rgba(77,163,255,.08); }}
    tbody tr:nth-child(even) {{ background: rgba(20,31,56,.36); }}
    .compact-table {{ min-width: 780px; }}
    .leaderboard-table {{ min-width: 1360px; }}
    .leaderboard-table td:nth-child(2) a {{ font-weight: 700; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .reason {{ max-width: 320px; color: #dbe7ff; }}
    .src {{ white-space: nowrap; color: var(--muted); }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 1120px) {{
      .split-grid, .double-grid {{ grid-template-columns: 1fr; }}
      .leaderboard-table {{ min-width: 1100px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class='hero'>
      <div class='hero-top'>
        <div>
          <h1>中文分词课堂平台</h1>
          <p class='lead'>此页面用于课堂线下评测与演示：统一展示分词算法的总体 F1、分桶得分、运行时间、提交状态和课堂错例。当前默认课堂评测包已切换到 <code>test_assets/platform_eval_v2_draft</code>，四个主线数据域为 <code>NLPCC-Weibo</code>、<code>EvaHan-2022</code>、<code>TCM-Ancient-Books</code> 与 <code>samechar</code>。</p>
        </div>
        <div class='pill-row'>
          <div class='pill'>会话：{escape(str(summary.get('session_name', '未命名会话')))}</div>
          <div class='pill'>数据版本：{escape(str(summary.get('dataset_version', '未知')))}</div>
          <div class='pill'>标准版本：{escape(str(summary.get('standard_version', '未知')))}</div>
          <div class='pill'>评分规则：{escape(str(summary.get('scoring_rule_version', '未知')))}</div>
          <div class='pill'>提交协议：{escape(str(summary.get('submission_protocol_version', '未知')))}</div>
          <div class='pill'>模式覆盖：预测文件 {int(summary.get('prediction_mode_count', 0) or 0)} | 可执行包 {int(summary.get('executable_mode_count', 0) or 0)}</div>
        </div>
      </div>
      <div class='stats-grid'>
        <div class='stat'><div class='stat-value'>{int(summary.get('submission_count', len(board)) or 0)}</div><div class='stat-label'>提交总数</div></div>
        <div class='stat'><div class='stat-value'>{int(summary.get('success_count', len(success)) or 0)}</div><div class='stat-label'>成功提交</div></div>
        <div class='stat'><div class='stat-value'>{int(summary.get('failure_count', 0) or 0)}</div><div class='stat-label'>失败或异常</div></div>
        <div class='stat'><div class='stat-value'>{highest_f1}</div><div class='stat-label'>当前最高 F1</div></div>
        <div class='stat'><div class='stat-value'>{int(package_meta.get('line_count', 0) or 0)}</div><div class='stat-label'>评测句数</div></div>
      </div>
      <div class='pill-row' style='margin-top:12px'>{group_badges}</div>
    </section>

    {render_teacher_panel(summary)}

    <section class='panel split-grid'>
      <div>
        <div class='panel-head'><h2>表现总览</h2><div class='sub'>列出课堂最常需要即时解读的最佳结果。</div></div>
        <div class='mini-grid'>{overview_cards}</div>
      </div>
      <div>
        <div class='panel-head'><h2>导出与说明</h2></div>
        <div class='pill-row'>
          <a class='pill' href='leaderboard.csv'>下载 leaderboard.csv</a>
          <a class='pill' href='session_summary.json'>查看 session_summary.json</a>
          <a class='pill' href='reports/'>打开 reports 目录</a>
        </div>
        <p class='sub' style='margin-top:12px'>主榜支持按综合排名、总体 F1、运行时间、NLPCC、EvaHan、TCM 古籍、samechar、高难层、专项层等多个口径切换，适合课堂主屏直接展示。</p>
      </div>
    </section>

    <section class='panel'>
      <div class='panel-head'><h2>分榜聚焦</h2><div class='sub'>采用 Dash 风格的信息卡，快速展示各子集和关键指标上的最佳提交。</div></div>
      <div class='spot-grid'>{spotlight_cards}</div>
    </section>

    <section class='panel double-grid'>
      <div>
        <div class='panel-head'><h2>数据包构成</h2><div class='sub'>展示当前评测包中各数据桶的来源与句数。</div></div>
        {package_table}
      </div>
      <div>
        <div class='panel-head'><h2>教学错例统计</h2><div class='sub'>统计多个成功提交共同失败的样本，便于课后讲评与试题筛选。</div></div>
        {common_failures}
      </div>
    </section>

    <section class='panel' id='leaderboardPanel'>
      <div class='panel-head'><h2>总榜与筛选</h2><div class='sub'>支持按分组、模式、状态和多种排序口径切换。</div></div>
      <div class='pill-row' style='margin-top:12px'>{sort_buttons}</div>
      <div class='filter-grid'>
        <div class='field'><label>搜索提交名</label><input id='searchBox' type='text' placeholder='输入提交名或分组名'></div>
        <div class='field'><label>提交分组</label><select id='groupFilter'><option value='all'>全部</option></select></div>
        <div class='field'><label>提交模式</label><select id='modeFilter'><option value='all'>全部</option><option value='prediction_file_only'>预测文件</option><option value='executable_package'>可执行包</option></select></div>
        <div class='field'><label>提交状态</label><select id='statusFilter'><option value='all'>全部</option><option value='成功'>成功</option><option value='格式错误'>格式错误</option><option value='运行错误'>运行错误</option><option value='超时'>超时</option><option value='缺少输出'>缺少输出</option><option value='拒收'>拒收</option></select></div>
        <div class='field'><label>排序方式</label><select id='sortFilter'>{sort_options}</select></div>
      </div>
      <div class='table-wrap'>
        <table class='leaderboard-table'>
          <thead>
            <tr>
              <th style='width:72px'>排名</th>
              <th style='width:220px'>提交名</th>
              <th style='width:120px'>分组</th>
              <th style='width:120px'>模式</th>
              <th style='width:90px'>状态</th>
              <th style='width:90px'>总体 F1</th>
              <th style='width:90px'>NLPCC</th>
              <th style='width:90px'>EvaHan</th>
              <th style='width:90px'>TCM</th>
              <th style='width:90px'>samechar</th>
              <th style='width:90px'>高难层</th>
              <th style='width:90px'>专项层</th>
              <th style='width:90px'>错句数</th>
              <th style='width:96px'>用时(s)</th>
              <th style='width:144px'>提交时间</th>
              <th style='width:240px'>说明</th>
            </tr>
          </thead>
          <tbody id='boardBody'></tbody>
        </table>
      </div>
      <div id='boardMeta' class='sub' style='margin-top:10px'></div>
    </section>

    <section class='panel'>
      <div class='panel-head'><h2>失败提交一览</h2><div class='sub'>集中展示格式错误、运行错误、超时等异常。</div></div>
      {failure_table}
    </section>
  </main>

  <script>
    const BOARD = {board_json};
    const groupFilter = document.getElementById('groupFilter');
    const modeFilter = document.getElementById('modeFilter');
    const statusFilter = document.getElementById('statusFilter');
    const sortFilter = document.getElementById('sortFilter');
    const searchBox = document.getElementById('searchBox');
    const boardBody = document.getElementById('boardBody');
    const boardMeta = document.getElementById('boardMeta');

    function metric(value) {{
      const number = Number(value);
      return Number.isNaN(number) ? 'N/A' : number.toFixed(4);
    }}

    function modeLabel(mode) {{
      return mode === 'executable_package' ? '可执行包' : '预测文件';
    }}

    function statusClass(status) {{
      if (status === '成功') return 'ok';
      if (status === '等待中' || status === '运行中' || status === '重复提交') return 'warn';
      return 'bad';
    }}

    function metricKey(key) {{
      if (key === 'rank' || key === 'f1') return 'f1';
      if (key === 'runtime') return 'runtime_seconds';
      if (key === 'nlpcc') return 'NLPCC-Weibo_f1';
      if (key === 'evahan') return 'EvaHan-2022_f1';
      if (key === 'tcm') return 'TCM-Ancient-Books_f1';
      if (key === 'samechar') return 'samechar_f1';
      if (key === 'high') return 'high_f1';
      if (key === 'specialized') return 'specialized_f1';
      return 'f1';
    }}

    function numericValue(row, column) {{
      const value = Number(row[column]);
      return Number.isNaN(value) ? null : value;
    }}

    function sortedRows(rows, sortKey) {{
      const items = [...rows];
      if (sortKey === 'latest') {{
        items.sort((a, b) => String(b.timestamp || '').localeCompare(String(a.timestamp || ''), 'zh-CN'));
        return items;
      }}
      if (sortKey === 'runtime') {{
        items.sort((a, b) => {{
          const av = numericValue(a, 'runtime_seconds');
          const bv = numericValue(b, 'runtime_seconds');
          const ax = av === null ? Number.POSITIVE_INFINITY : av;
          const bx = bv === null ? Number.POSITIVE_INFINITY : bv;
          if (ax !== bx) return ax - bx;
          return (numericValue(b, 'f1') || 0) - (numericValue(a, 'f1') || 0);
        }});
        return items;
      }}
      if (sortKey === 'rank') {{
        items.sort((a, b) => Number(a.rank || 99999) - Number(b.rank || 99999));
        return items;
      }}
      const column = metricKey(sortKey);
      items.sort((a, b) => {{
        const av = numericValue(a, column);
        const bv = numericValue(b, column);
        const ax = av === null ? Number.NEGATIVE_INFINITY : av;
        const bx = bv === null ? Number.NEGATIVE_INFINITY : bv;
        if (bx !== ax) return bx - ax;
        return (numericValue(a, 'runtime_seconds') || 0) - (numericValue(b, 'runtime_seconds') || 0);
      }});
      return items;
    }}

    [...new Set(BOARD.map(row => row.submission_group || '未分组'))]
      .sort((a, b) => a.localeCompare(b, 'zh-CN'))
      .forEach(group => {{
        const option = document.createElement('option');
        option.value = group;
        option.textContent = group;
        groupFilter.appendChild(option);
      }});

    function renderBoard() {{
      const query = (searchBox.value || '').trim().toLowerCase();
      const group = groupFilter.value;
      const mode = modeFilter.value;
      const status = statusFilter.value;
      const sortKey = sortFilter.value;

      let rows = BOARD.filter(row => {{
        const haystack = `${{row.submission_name || ''}} ${{row.submission_group || ''}} ${{row.message || ''}}`.toLowerCase();
        if (query && !haystack.includes(query)) return false;
        if (group !== 'all' && (row.submission_group || '未分组') !== group) return false;
        if (mode !== 'all' && row.mode !== mode) return false;
        if (status !== 'all' && row.status !== status) return false;
        return true;
      }});

      rows = sortedRows(rows, sortKey);
      boardBody.innerHTML = rows.map((row, index) => {{
        const rank = sortKey === 'rank' ? (row.rank ?? index + 1) : index + 1;
        const name = row.detail_link ? `<a href='${{row.detail_link}}'>${{row.submission_name}}</a>` : `${{row.submission_name}}`;
        return `<tr>
          <td>${{rank}}</td>
          <td>${{name}}</td>
          <td>${{row.submission_group || '未分组'}}</td>
          <td>${{modeLabel(row.mode)}}</td>
          <td><span class='${{statusClass(row.status)}}'>${{row.status}}</span></td>
          <td>${{metric(row.f1)}}</td>
          <td>${{metric(row['NLPCC-Weibo_f1'])}}</td>
          <td>${{metric(row['EvaHan-2022_f1'])}}</td>
          <td>${{metric(row['TCM-Ancient-Books_f1'])}}</td>
          <td>${{metric(row.samechar_f1)}}</td>
          <td>${{metric(row.high_f1)}}</td>
          <td>${{metric(row.specialized_f1)}}</td>
          <td>${{row.wrong_sentence_count ?? 0}}</td>
          <td>${{metric(row.runtime_seconds)}}</td>
          <td>${{row.timestamp || ''}}</td>
          <td class='reason'>${{row.message || ''}}</td>
        </tr>`;
      }}).join('');

      boardMeta.textContent = `当前显示 ${{rows.length}} / ${{BOARD.length}} 条提交，排序方式：${{sortFilter.options[sortFilter.selectedIndex].text}}。`;
    }}

    [groupFilter, modeFilter, statusFilter, sortFilter, searchBox].forEach(element => {{
      element.addEventListener('input', renderBoard);
      element.addEventListener('change', renderBoard);
    }});
    document.querySelectorAll('.sort-btn').forEach(button => {{
      button.addEventListener('click', () => {{
        sortFilter.value = button.dataset.sort;
        renderBoard();
        document.getElementById('leaderboardPanel').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }});
    }});
    renderBoard();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    leaderboard_path = Path(args.leaderboard)
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if leaderboard_path.exists():
        board = pd.read_csv(leaderboard_path, encoding='utf-8-sig')
    else:
        board = pd.DataFrame(
            columns=[
                'rank', 'submission_name', 'submission_group', 'mode', 'status', 'timestamp', 'runtime_seconds',
                'precision', 'recall', 'f1', 'wrong_sentence_count', 'message', 'NLPCC-Weibo_f1', 'EvaHan-2022_f1',
                'TCM-Ancient-Books_f1', 'samechar_f1', 'high_f1', 'medium_f1', 'specialized_f1',
            ]
        )

    summary = load_json(args.session_summary)
    package_meta = load_json(args.package_meta)
    reports = load_reports(reports_dir)
    manifest_lookup = manifest_rows(summary)
    links = write_detail_pages(reports, output_path.parent / 'details')
    output_path.write_text(render_page(board, reports, links, summary, package_meta, manifest_lookup), encoding='utf-8')


if __name__ == '__main__':
    main()
