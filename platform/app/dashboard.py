from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import dash
from dash import Dash, Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISPLAY_NAMES = {
    'NLPCC-Weibo_f1': 'NLPCC',
    'EvaHan-2022_f1': 'EvaHan',
    'TCM-Ancient-Books_f1': 'TCM',
    'samechar_f1': 'samechar',
    'high_f1': '高难层',
    'medium_f1': '中难层',
    'specialized_f1': '专项层',
    'f1': '总体F1',
    'runtime_seconds': '运行时间(s)',
}

METRIC_COLUMNS = [
    'f1',
    'NLPCC-Weibo_f1',
    'EvaHan-2022_f1',
    'TCM-Ancient-Books_f1',
    'samechar_f1',
    'high_f1',
    'medium_f1',
    'specialized_f1',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the interactive classroom leaderboard dashboard.')
    parser.add_argument('--results-dir', default='platform/results')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def load_reports(reports_dir: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    if not reports_dir.exists():
        return reports
    for path in sorted(reports_dir.glob('*.report.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        reports[str(data.get('submission_name', path.stem))] = data
    return reports


def load_manifest(summary: dict) -> pd.DataFrame:
    manifest_path = Path(summary.get('manifest_path', '')) if summary.get('manifest_path') else None
    if manifest_path and manifest_path.exists():
        return pd.read_csv(manifest_path, encoding='utf-8-sig')
    return pd.DataFrame()


def collect_common_failures(reports: dict[str, dict], manifest_df: pd.DataFrame) -> pd.DataFrame:
    manifest_rows = {}
    if not manifest_df.empty and 'line_no' in manifest_df.columns:
        manifest_rows = {int(row['line_no']): row for _, row in manifest_df.iterrows()}

    aggregate: dict[tuple[int, str], dict] = {}
    for submission_name, report in reports.items():
        if report.get('status') != '成功':
            continue
        for case in report.get('wrong_cases', []) or []:
            line_no = int(case.get('line_no', 0) or 0)
            raw = str(case.get('raw_text', '')).strip()
            gold = str(case.get('gold', '')).strip()
            if not raw or not gold:
                continue
            key = (line_no, raw)
            manifest_row = manifest_rows.get(line_no, {})
            item = aggregate.setdefault(
                key,
                {
                    'line_no': line_no,
                    'raw_text': raw,
                    'gold': gold,
                    'dataset': str(manifest_row.get('dataset', '')),
                    'sample_id': str(manifest_row.get('sample_id', '')),
                    'count': 0,
                    'submissions': [],
                },
            )
            item['count'] += 1
            item['submissions'].append(submission_name)
    rows = sorted(aggregate.values(), key=lambda row: (-row['count'], row['line_no']))[:20]
    return pd.DataFrame(rows)


def normalize_board(board: pd.DataFrame) -> pd.DataFrame:
    frame = board.copy()
    for column in ['rank', 'runtime_seconds', 'precision', 'recall', 'f1', 'wrong_sentence_count', *METRIC_COLUMNS]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if 'status' in frame.columns:
        frame['status'] = frame['status'].fillna('')
    if 'submission_group' in frame.columns:
        frame['submission_group'] = frame['submission_group'].fillna('未分组')
    if 'mode' in frame.columns:
        frame['mode_label'] = frame['mode'].map(
            {
                'prediction_file_only': '预测文件',
                'executable_package': '可执行包',
            }
        ).fillna(frame['mode'])
    return frame


def build_kpi_cards(board: pd.DataFrame, summary: dict) -> list[dbc.Col]:
    success = board[board['status'] == '成功'].copy()
    highest = float(success['f1'].max()) if not success.empty else 0.0
    fastest = float(success['runtime_seconds'].min()) if not success.empty else 0.0
    cards = [
        ('提交总数', int(summary.get('submission_count', len(board)) or 0), '次'),
        ('成功提交', int(summary.get('success_count', int((board['status'] == '成功').sum())) or 0), '次'),
        ('最高 F1', f'{highest:.4f}', ''),
        ('最快用时', f'{fastest:.4f}', '秒'),
    ]
    return [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(title, className='text-secondary small mb-2'),
                        html.Div(f'{value}{suffix}', className='display-6 fw-bold'),
                    ]
                ),
                className='shadow-sm border-0 leaderboard-card',
            ),
            md=3,
        )
        for title, value, suffix in cards
    ]


def make_status_figure(board: pd.DataFrame) -> go.Figure:
    counts = board['status'].value_counts().reset_index()
    counts.columns = ['status', 'count']
    fig = px.pie(
        counts,
        names='status',
        values='count',
        hole=0.58,
        color='status',
        color_discrete_map={
            '成功': '#34d399',
            '格式错误': '#fb7185',
            '运行错误': '#f97316',
            '超时': '#fbbf24',
            '缺少输出': '#f43f5e',
            '拒收': '#ef4444',
        },
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        legend_orientation='h',
    )
    return fig


def make_runtime_scatter(board: pd.DataFrame) -> go.Figure:
    success = board[board['status'] == '成功'].copy()
    fig = px.scatter(
        success,
        x='runtime_seconds',
        y='f1',
        color='submission_group',
        hover_name='submission_name',
        size='samechar_f1',
        size_max=22,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.35)')))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title='运行时间 vs 总体 F1',
        xaxis_title='运行时间 (s)',
        yaxis_title='总体 F1',
    )
    return fig


def make_metric_heatmap(board: pd.DataFrame) -> go.Figure:
    success = board[board['status'] == '成功'].copy().sort_values('f1', ascending=False).head(10)
    matrix = success[[*METRIC_COLUMNS]].fillna(0.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[DISPLAY_NAMES[col] for col in matrix.columns],
            y=success['submission_name'],
            colorscale='Blues',
            zmin=0.0,
            zmax=1.0,
            text=[[f'{value:.4f}' for value in row] for row in matrix.values],
            texttemplate='%{text}',
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title='前 10 名多指标热力图',
    )
    return fig


def make_profile_radar(row: pd.Series) -> go.Figure:
    theta = [DISPLAY_NAMES[col] for col in METRIC_COLUMNS]
    values = [float(row.get(col, 0.0) or 0.0) for col in METRIC_COLUMNS]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + values[:1],
                theta=theta + theta[:1],
                fill='toself',
                name=str(row.get('submission_name', '')),
                line=dict(color='#4da3ff', width=3),
                fillcolor='rgba(77,163,255,0.28)',
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        polar=dict(radialaxis=dict(range=[0, 1], showline=False, gridcolor='rgba(152,168,202,0.2)')),
        title='单提交能力剖面',
    )
    return fig


def make_dataset_bar(report: dict) -> go.Figure:
    by_dataset = report.get('by_dataset', {}) or {}
    if not by_dataset:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6eefc',
            title='分数据集得分',
        )
        return fig
    rows = []
    for dataset, payload in by_dataset.items():
        rows.append({'dataset': dataset, 'f1': float(payload.get('f1', 0.0))})
    df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    fig = px.bar(df, x='dataset', y='f1', color='f1', color_continuous_scale='Blues')
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title='分数据集得分',
        coloraxis_showscale=False,
        xaxis_title='',
        yaxis_title='F1',
        yaxis=dict(range=[0, 1]),
    )
    return fig


def make_rank_bar(rows: list[dict]) -> go.Figure:
    frame = pd.DataFrame(rows or [])
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6eefc',
            title='当前 Top 10 排名条形榜',
        )
        return fig
    top = frame.head(10).copy()
    top = top.sort_values('f1', ascending=True)
    colors = ['#1d4ed8'] * len(top)
    if len(top) >= 1:
        colors[-1] = '#fbbf24'
    if len(top) >= 2:
        colors[-2] = '#e5e7eb'
    if len(top) >= 3:
        colors[-3] = '#fb7185'
    fig = go.Figure(
        go.Bar(
            x=top['f1'],
            y=top['submission_name'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{float(v):.4f}' for v in top['f1']],
            textposition='outside',
            hovertemplate='%{y}<br>总体F1=%{x:.4f}<extra></extra>',
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=30, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title='当前 Top 10 排名条形榜',
        xaxis_title='总体 F1',
        yaxis_title='',
        transition_duration=500,
    )
    return fig


def make_podium(rows: list[dict]) -> list[html.Div]:
    frame = pd.DataFrame(rows or [])
    if frame.empty:
        return [html.Div('当前没有可展示的成功提交。', className='text-secondary')]
    top = frame.head(3).copy().reset_index(drop=True)
    order = []
    if len(top) >= 2:
        order.append((1, top.iloc[1], 'silver', 0.82))
    if len(top) >= 1:
        order.append((0, top.iloc[0], 'gold', 1.0))
    if len(top) >= 3:
        order.append((2, top.iloc[2], 'bronze', 0.68))

    blocks = []
    for visual_rank, row, tone, height in order:
        rank = visual_rank + 1
        metric_bits = []
        for key in ['NLPCC-Weibo_f1', 'EvaHan-2022_f1', 'TCM-Ancient-Books_f1', 'samechar_f1']:
            value = row.get(key)
            if pd.notna(value):
                metric_bits.append(f"{DISPLAY_NAMES[key]} {float(value):.3f}")
        blocks.append(
            html.Div(
                [
                    html.Div(f'#{rank}', className=f'podium-rank podium-{tone}'),
                    html.Div(str(row.get('submission_name', '')), className='podium-name'),
                    html.Div(str(row.get('submission_group', '')), className='podium-group'),
                    html.Div(f"总体 F1 {float(row.get('f1', 0.0)):.4f}", className='podium-score'),
                    html.Div(f"运行时间 {float(row.get('runtime_seconds', 0.0)):.4f} s", className='podium-runtime'),
                    html.Div(' ｜ '.join(metric_bits[:2]), className='podium-meta'),
                    html.Div(className='podium-pillar', style={'height': f'{int(height*220)}px'}),
                ],
                className=f'podium-card podium-{tone}',
            )
        )
    return blocks


def create_app(results_dir: Path) -> Dash:
    leaderboard_path = results_dir / 'leaderboard.csv'
    reports_dir = results_dir / 'reports'
    session_summary = load_json(results_dir / 'session_summary.json')
    package_meta = load_json(Path(session_summary.get('package_meta', ''))) if session_summary.get('package_meta') else load_json(results_dir.parent / 'test_assets' / 'platform_eval_v2_draft' / 'package_meta.json')
    reports = load_reports(reports_dir)
    manifest_df = load_manifest(session_summary)
    common_failures = collect_common_failures(reports, manifest_df)

    board = pd.read_csv(leaderboard_path, encoding='utf-8-sig') if leaderboard_path.exists() else pd.DataFrame(columns=['submission_name'])
    board = normalize_board(board)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title='中文分词课堂平台',
        assets_folder=str((Path(__file__).resolve().parent / 'assets')),
    )
    app.layout = dbc.Container(
        [
            dcc.Store(id='board-store', data=board.to_dict('records')),
            dcc.Store(id='reports-store', data=reports),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div('中文分词课堂平台', className='display-5 fw-bold'),
                                html.Div(
                                    f"数据版本：{session_summary.get('dataset_version', '未知')}｜评分规则：{session_summary.get('scoring_rule_version', '未知')}｜共 {len(board)} 条提交",
                                    className='text-secondary mt-2',
                                ),
                            ]
                        ),
                        className='shadow border-0 mt-4 mb-4 leaderboard-hero',
                    )
                )
            ),
            dbc.Row(build_kpi_cards(board, session_summary), className='g-3 mb-4'),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div('当前领奖台', className='h4 mb-3'),
                                html.Div('根据当前筛选结果实时生成前三名展示，并附带动态柱体与高亮样式。', className='text-secondary mb-2'),
                                html.Div(id='podium-section', className='podium-grid'),
                            ]
                        ),
                        className='shadow-sm border-0 mb-4 podium-shell',
                    )
                )
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='status-fig', figure=make_status_figure(board))), className='shadow-sm border-0 h-100'), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='runtime-fig', figure=make_runtime_scatter(board))), className='shadow-sm border-0 h-100'), md=8),
                ],
                className='g-3 mb-4',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(id='rank-bar'),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('选中提交的分数据集表现', className='mb-3'),
                                    dcc.Graph(id='dataset-bar'),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=5,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('选中提交的错例样本', className='mb-3'),
                                    dash_table.DataTable(
                                        id='selected-wrong-cases',
                                        columns=[
                                            {'name': '行号', 'id': 'line_no'},
                                            {'name': '原句', 'id': 'raw_text'},
                                            {'name': '标准切分', 'id': 'gold'},
                                            {'name': '预测结果', 'id': 'pred'},
                                        ],
                                        data=[],
                                        page_size=6,
                                        style_table={'overflowX': 'auto'},
                                        style_header={'backgroundColor': '#112240', 'color': '#e6eefc', 'fontWeight': 'bold'},
                                        style_cell={'backgroundColor': '#0f172a', 'color': '#dbe7ff', 'border': '1px solid rgba(93,126,182,0.18)', 'textAlign': 'left', 'whiteSpace': 'normal'},
                                    ),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=7,
                    ),
                ],
                className='g-3 mb-4',
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='heatmap-fig', figure=make_metric_heatmap(board))), className='shadow-sm border-0 h-100'), lg=8),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('筛选与排序', className='mb-3'),
                                    dbc.Label('搜索提交名'),
                                    dbc.Input(id='search-input', placeholder='输入提交名、分组或说明'),
                                    dbc.Label('提交分组', className='mt-3'),
                                    dcc.Dropdown(
                                        id='group-filter',
                                        options=[{'label': '全部', 'value': 'all'}] + [{'label': value, 'value': value} for value in sorted(board['submission_group'].dropna().unique().tolist())],
                                        value='all',
                                        clearable=False,
                                    ),
                                    dbc.Label('提交模式', className='mt-3'),
                                    dcc.Dropdown(
                                        id='mode-filter',
                                        options=[
                                            {'label': '全部', 'value': 'all'},
                                            {'label': '预测文件', 'value': 'prediction_file_only'},
                                            {'label': '可执行包', 'value': 'executable_package'},
                                        ],
                                        value='all',
                                        clearable=False,
                                    ),
                                    dbc.Label('提交状态', className='mt-3'),
                                    dcc.Dropdown(
                                        id='status-filter',
                                        options=[{'label': '全部', 'value': 'all'}] + [{'label': value, 'value': value} for value in sorted(board['status'].dropna().unique().tolist())],
                                        value='all',
                                        clearable=False,
                                    ),
                                    dbc.Label('排序指标', className='mt-3'),
                                    dcc.Dropdown(
                                        id='sort-key',
                                        options=[{'label': DISPLAY_NAMES.get(col, col), 'value': col} for col in ['f1', 'NLPCC-Weibo_f1', 'EvaHan-2022_f1', 'TCM-Ancient-Books_f1', 'samechar_f1', 'high_f1', 'specialized_f1', 'runtime_seconds']],
                                        value='f1',
                                        clearable=False,
                                    ),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=4,
                    ),
                ],
                className='g-3 mb-4',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('排行榜', className='mb-3'),
                                    dash_table.DataTable(
                                        id='leaderboard-table',
                                        columns=[
                                            {'name': '排名', 'id': 'rank'},
                                            {'name': '提交名', 'id': 'submission_name'},
                                            {'name': '分组', 'id': 'submission_group'},
                                            {'name': '状态', 'id': 'status'},
                                            {'name': '模式', 'id': 'mode_label'},
                                            {'name': '总体F1', 'id': 'f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': 'NLPCC', 'id': 'NLPCC-Weibo_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': 'EvaHan', 'id': 'EvaHan-2022_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': 'TCM', 'id': 'TCM-Ancient-Books_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': 'samechar', 'id': 'samechar_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': '高难层', 'id': 'high_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': '专项层', 'id': 'specialized_f1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': '运行时间', 'id': 'runtime_seconds', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                            {'name': '说明', 'id': 'message'},
                                        ],
                                        data=board.to_dict('records'),
                                        sort_action='native',
                                        filter_action='none',
                                        page_size=12,
                                        row_selectable='single',
                                        selected_rows=[0] if not board.empty else [],
                                        style_as_list_view=True,
                                        style_table={'overflowX': 'auto'},
                                        style_header={'backgroundColor': '#112240', 'color': '#e6eefc', 'fontWeight': 'bold'},
                                        style_cell={'backgroundColor': '#0f172a', 'color': '#dbe7ff', 'border': '1px solid rgba(93,126,182,0.18)', 'textAlign': 'left', 'minWidth': '90px', 'maxWidth': '280px', 'whiteSpace': 'normal'},
                                        style_data_conditional=[
                                            {'if': {'filter_query': '{status} = 成功'}, 'backgroundColor': '#0f2e2a'},
                                            {'if': {'filter_query': '{status} contains 错误 || {status} = 拒收 || {status} = 超时'}, 'backgroundColor': '#3a1e2b'},
                                        ],
                                    ),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('提交详情', className='mb-3'),
                                    dcc.Graph(id='profile-radar'),
                                    html.Div(id='detail-meta', className='text-secondary small mt-2'),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=4,
                    ),
                ],
                className='g-3 mb-4',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('共性错例', className='mb-3'),
                                    dash_table.DataTable(
                                        id='common-failure-table',
                                        columns=[
                                            {'name': '行号', 'id': 'line_no'},
                                            {'name': '数据桶', 'id': 'dataset'},
                                            {'name': '样本编号', 'id': 'sample_id'},
                                            {'name': '原句', 'id': 'raw_text'},
                                            {'name': '标准切分', 'id': 'gold'},
                                            {'name': '失败次数', 'id': 'count'},
                                        ],
                                        data=common_failures.to_dict('records'),
                                        page_size=8,
                                        style_table={'overflowX': 'auto'},
                                        style_header={'backgroundColor': '#112240', 'color': '#e6eefc', 'fontWeight': 'bold'},
                                        style_cell={'backgroundColor': '#0f172a', 'color': '#dbe7ff', 'border': '1px solid rgba(93,126,182,0.18)', 'textAlign': 'left', 'whiteSpace': 'normal'},
                                    ),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5('数据包信息', className='mb-3'),
                                    html.Div(f"评测包：{package_meta.get('package_name', '未知')}", className='mb-2'),
                                    html.Div(f"句数：{package_meta.get('line_count', 0)}", className='mb-2'),
                                    html.Div(f"难度桶：{package_meta.get('difficulty_buckets', {})}", className='mb-2'),
                                    html.Div(f"review flags：{package_meta.get('review_flag_counts', {})}", className='mb-2'),
                                    html.Hr(),
                                    html.Div('启动方式', className='text-secondary small mb-2'),
                                    html.Pre(f'python app/leaderboard.py\n# 或\npython platform/app/dashboard.py', style={'whiteSpace': 'pre-wrap'}),
                                ]
                            ),
                            className='shadow-sm border-0 h-100',
                        ),
                        lg=5,
                    ),
                ],
                className='g-3 mb-4',
            ),
        ],
        fluid=True,
        className='pb-4',
    )

    @app.callback(
        Output('leaderboard-table', 'data'),
        Input('board-store', 'data'),
        Input('search-input', 'value'),
        Input('group-filter', 'value'),
        Input('mode-filter', 'value'),
        Input('status-filter', 'value'),
        Input('sort-key', 'value'),
    )
    def filter_board(rows: list[dict], query: str, group: str, mode: str, status: str, sort_key: str) -> list[dict]:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return []
        if query:
            q = query.lower().strip()
            mask = frame.apply(lambda row: q in ' '.join(str(row.get(col, '')).lower() for col in ['submission_name', 'submission_group', 'message']), axis=1)
            frame = frame[mask]
        if group != 'all':
            frame = frame[frame['submission_group'] == group]
        if mode != 'all':
            frame = frame[frame['mode'] == mode]
        if status != 'all':
            frame = frame[frame['status'] == status]
        if sort_key == 'runtime_seconds':
            frame = frame.sort_values(by=['runtime_seconds', 'f1'], ascending=[True, False], na_position='last')
        else:
            frame = frame.sort_values(by=[sort_key, 'runtime_seconds'], ascending=[False, True], na_position='last')
        frame = frame.copy().reset_index(drop=True)
        frame['rank'] = range(1, len(frame) + 1)
        return frame.to_dict('records')

    @app.callback(
        Output('profile-radar', 'figure'),
        Output('detail-meta', 'children'),
        Output('dataset-bar', 'figure'),
        Output('selected-wrong-cases', 'data'),
        Output('podium-section', 'children'),
        Output('rank-bar', 'figure'),
        Input('leaderboard-table', 'derived_virtual_data'),
        Input('leaderboard-table', 'selected_rows'),
        State('reports-store', 'data'),
    )
    def update_detail(rows: list[dict] | None, selected_rows: list[int] | None, reports_map: dict):
        frame = pd.DataFrame(rows or [])
        if frame.empty:
            return go.Figure(), '暂无可展示的提交。', go.Figure(), [], make_podium([]), make_rank_bar([])
        idx = selected_rows[0] if selected_rows else 0
        idx = min(max(idx, 0), len(frame) - 1)
        row = frame.iloc[idx]
        fig = make_profile_radar(row)
        meta = f"提交名：{row.get('submission_name', '')}｜状态：{row.get('status', '')}｜运行时间：{row.get('runtime_seconds', 0):.4f}s｜说明：{row.get('message', '') or '无'}"
        report = (reports_map or {}).get(str(row.get('submission_name', '')), {})
        dataset_bar = make_dataset_bar(report)
        wrong_cases = (report.get('wrong_cases', []) or [])[:12]
        return fig, meta, dataset_bar, wrong_cases, make_podium(rows or []), make_rank_bar(rows or [])

    return app


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    app = create_app(results_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
