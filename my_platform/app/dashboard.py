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
    'samechar_f1': 'samechar专项',
    'f1': '总体F1',
    'runtime_seconds': '运行时间(s)',
}

DATASET_DISPLAY_NAMES = {
    'NLPCC-Weibo': 'NLPCC微博',
    'EvaHan-2022': 'EvaHan',
    'TCM-Ancient-Books': 'TCM古籍',
    'samechar': 'samechar专项',
}

METRIC_COLUMNS = [
    'f1',
    'NLPCC-Weibo_f1',
    'EvaHan-2022_f1',
    'TCM-Ancient-Books_f1',
    'samechar_f1',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the interactive classroom leaderboard dashboard.')
    parser.add_argument('--results-dir', default='my_platform/results')
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
    if 'submission_group' in frame.columns:
        frame['comparison_kind'] = frame['submission_group'].fillna('').map(
            lambda value: 'reference' if any(key in str(value) for key in ['工具', 'tool', 'AI', '大模型', 'LLM']) else 'student'
        )
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
                        html.Div(title, className='text-secondary small mb-1'),
                        html.Div(f'{value}{suffix}', className='h4 fw-bold mb-0'),
                    ]
                ),
                className='shadow-sm border-0 leaderboard-card',
            ),
            md=3,
            style={'padding': '4px'}
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
        hole=0.6,
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
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=11,
        marker=dict(line=dict(color='rgba(255,255,255,0.1)', width=2))
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        legend_orientation='h',
        legend_y=-0.05,
        height=240,
        showlegend=True,
    )
    return fig


def make_runtime_scatter(board: pd.DataFrame) -> go.Figure:
    success = board[board['status'] == '成功'].copy()
    fig = px.scatter(
        success,
        x='runtime_seconds',
        y='f1',
        color='mode_label',
        hover_name='submission_name',
        size='samechar_f1',
        size_max=18,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.35)')))
    fig.update_layout(
        margin=dict(l=16, r=16, t=24, b=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title={'text': '运行时间 vs 总体 F1', 'font': {'size': 14}},
        xaxis_title='运行时间',
        yaxis_title='总体 F1',
        title_x=0.5,
        height=280,
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
            text=[[f'{value:.3f}' for value in row] for row in matrix.values],
            texttemplate='%{text}',
        )
    )
    fig.update_layout(
        margin=dict(l=16, r=16, t=24, b=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title={'text': '前 10 名多指标热力图', 'font': {'size': 14}},
        title_x=0.5,
        height=320,
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
                line=dict(color='#4da3ff', width=2.5),
                fillcolor='rgba(77,163,255,0.25)',
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        font_size=11,
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                showline=False,
                gridcolor='rgba(152,168,202,0.2)',
                tickfont=dict(size=9)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        title=None,
        height=250,
    )
    return fig


def make_dataset_bar(row: pd.Series) -> go.Figure:
    # 从CSV行数据中提取分数据集的F1分数
    dataset_metrics = {
        'NLPCC-Weibo_f1': 'NLPCC微博',
        'EvaHan-2022_f1': 'EvaHan',
        'TCM-Ancient-Books_f1': 'TCM古籍',
        'samechar_f1': 'samechar专项',
    }

    rows = []
    for metric, display_name in dataset_metrics.items():
        value = row.get(metric)
        if pd.notna(value) and value != '':
            f1_score = float(value)
            if f1_score > 0:  # 只显示有值的分数
                rows.append({'dataset': display_name, 'f1': f1_score})

    if not rows:
        # 如果没有数据，显示空图表
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=16, r=16, t=24, b=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6eefc',
            title={'text': '分数据集得分', 'font': {'size': 14}},
            title_x=0.5,
            height=240,
        )
        return fig

    df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    fig = px.bar(df, x='dataset', y='f1', color='f1', color_continuous_scale='Blues')
    fig.update_layout(
        margin=dict(l=16, r=16, t=24, b=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title={'text': '分数据集得分', 'font': {'size': 14}},
        title_x=0.5,
        coloraxis_showscale=False,
        xaxis_title='',
        yaxis_title='F1',
        yaxis=dict(range=[0, 1]),
        height=240,
    )
    return fig


def make_rank_bar(rows: list[dict]) -> go.Figure:
    frame = pd.DataFrame(rows or [])
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=16, r=16, t=24, b=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6eefc',
            title={'text': '当前 Top 10 排名条形榜', 'font': {'size': 14}},
            title_x=0.5,
            height=280,
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
            text=[f'{float(v):.3f}' for v in top['f1']],
            textposition='inside',
            insidetextanchor='end',
            insidetextfont=dict(color='white', size=11),
            hovertemplate='%{y}<br>总体F1=%{x:.4f}<extra></extra>',
        )
    )
    fig.update_layout(
        margin=dict(l=16, r=60, t=24, b=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eefc',
        title={'text': '当前 Top 10 排名条形榜', 'font': {'size': 14}},
        title_x=0.5,
        xaxis_title='总体 F1',
        yaxis_title='',
        transition_duration=500,
        height=280,
        bargap=0.3,
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

    # 处理package_meta路径 - 支持跨平台
    package_meta_path = None
    if session_summary.get('package_meta'):
        original_path = Path(session_summary.get('package_meta', ''))
        if original_path.exists():
            package_meta_path = original_path

    # 如果原路径不存在，使用相对路径作为备选
    if not package_meta_path:
        package_meta_path = results_dir.parent / 'test_assets' / 'platform_eval_v2_draft' / 'package_meta.json'

    package_meta = load_json(package_meta_path) if package_meta_path.exists() else {}
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
                                html.Div('中文分词竞赛平台', className='h3 fw-bold mb-2'),
                                html.Div(className='d-flex justify-content-between align-items-center flex-wrap', children=[
                                    html.Div(
                                        f"数据集 {session_summary.get('dataset_version', '未知')} • 评分规则 {session_summary.get('scoring_rule_version', '未知')}",
                                        className='text-secondary small me-3',
                                    ),
                                    html.Div(className='d-flex gap-3', children=[
                                        html.Div([
                                            html.Span(f"{len(board)} ", className='fw-bold text-info'),
                                            html.Span('提交', className='text-secondary small')
                                        ]),
                                        html.Div([
                                            html.Span(f"{int((board['status'] == '成功').sum())} ", className='fw-bold text-success'),
                                            html.Span('成功', className='text-secondary small')
                                        ]),
                                        html.Div([
                                            html.Span(f"{float(board[board['status'] == '成功']['f1'].max()) if not board[board['status'] == '成功'].empty else 0:.3f} ", className='fw-bold text-warning'),
                                            html.Span('最佳', className='text-secondary small')
                                        ]),
                                        html.Div([
                                            html.Span(f"{float(board[board['status'] == '成功']['runtime_seconds'].min()) if not board[board['status'] == '成功'].empty else 0:.2f}s ", className='fw-bold'),
                                            html.Span('最快', className='text-secondary small')
                                        ]),
                                    ])
                                ])
                            ]
                        ),
                        className='shadow border-0 mt-3 mb-3 leaderboard-hero',
                    )
                )
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div('🏆 实时领奖台', className='h6 mb-2 fw-bold'),
                                html.Div(id='podium-section', className='podium-grid'),
                            ]
                        ),
                        className='shadow-sm border-0 podium-shell',
                    ),
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div('📊 多指标热力图', className='h6 mb-2 fw-bold'),
                                    dcc.Graph(id='heatmap-fig', figure=make_metric_heatmap(board), style={'height': '280px'}),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div('🎯 选中提交分析', className='h6 mb-2 fw-bold'),
                                    html.Div(className='row g-2', children=[
                                        dbc.Col([
                                            dcc.Graph(id='dataset-bar', style={'height': '250px'})
                                        ], width=6),
                                        dbc.Col([
                                            dcc.Graph(id='profile-radar', style={'height': '250px'})
                                        ], width=6),
                                    ])
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=6,
                    ),
                ],
                className='g-3 mb-3',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div('📈 排名趋势', className='h6 mb-2 fw-bold'),
                                    dcc.Graph(id='rank-bar', style={'height': '320px'}),
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
                                    html.Div('🔍 筛选', className='h6 mb-2 fw-bold'),
                                    dbc.Label('搜索', className='small mb-1'),
                                    dbc.Input(id='search-input', placeholder='提交名、说明', size='sm', className='mb-2'),
                                    dbc.Checklist(
                                        id='reference-toggle',
                                        options=[{'label': '加入 AI / 工具对比', 'value': 'show'}],
                                        value=[],
                                        switch=True,
                                        className='mb-2',
                                    ),
                                    dbc.Label('模式', className='small mb-1'),
                                    dcc.Dropdown(id='mode-filter', options=[{'label': '全部', 'value': 'all'}, {'label': '预测文件', 'value': 'prediction_file_only'}, {'label': '可执行包', 'value': 'executable_package'}], value='all', clearable=False, className='mb-2'),
                                    dbc.Label('状态', className='small mb-1'),
                                    dcc.Dropdown(id='status-filter', options=[{'label': '全部', 'value': 'all'}] + [{'label': value, 'value': value} for value in sorted(board['status'].dropna().unique().tolist())], value='all', clearable=False, className='mb-2'),
                                    dbc.Label('排序', className='small mb-1'),
                                    dcc.Dropdown(id='sort-key', options=[{'label': DISPLAY_NAMES.get(col, col), 'value': col} for col in ['f1', 'NLPCC-Weibo_f1', 'EvaHan-2022_f1', 'TCM-Ancient-Books_f1', 'samechar_f1', 'runtime_seconds']], value='f1', clearable=False),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=4,
                    ),
                ],
                className='g-3 mb-3',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div('🏆 完整排行榜', className='h6 mb-2 fw-bold'),
                                    dash_table.DataTable(
                                        id='leaderboard-table',
                                        columns=[
                                            {'name': '排名', 'id': 'rank'},
                                            {'name': '提交名', 'id': 'submission_name'},
                                            {'name': '状态', 'id': 'status'},
                                            {'name': '总体F1', 'id': 'f1', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                            {'name': 'NLPCC', 'id': 'NLPCC-Weibo_f1', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                            {'name': 'EvaHan', 'id': 'EvaHan-2022_f1', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                            {'name': 'TCM', 'id': 'TCM-Ancient-Books_f1', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                            {'name': 'samechar专项', 'id': 'samechar_f1', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                            {'name': '时间', 'id': 'runtime_seconds', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                        ],
                                        data=board.to_dict('records'),
                                        sort_action='native',
                                        filter_action='none',
                                        page_size=20,
                                        row_selectable='single',
                                        selected_rows=[0] if not board.empty else [],
                                        style_table={'overflowX': 'auto'},
                                        style_header={'backgroundColor': '#112240', 'color': '#e6eefc', 'fontWeight': 'bold'},
                                        style_cell={'backgroundColor': '#0f172a', 'color': '#dbe7ff', 'border': '1px solid rgba(93,126,182,0.18)', 'textAlign': 'left', 'minWidth': '60px', 'maxWidth': '100px', 'whiteSpace': 'normal', 'fontSize': '0.8rem'},
                                        style_data_conditional=[
                                            {'if': {'filter_query': '{status} = 成功'}, 'backgroundColor': '#0f2e2a'},
                                            {'if': {'filter_query': '{status} contains 错误 || {status} = 拒收 || {status} = 超时'}, 'backgroundColor': '#3a1e2b'},
                                            {'if': {'filter_query': '{comparison_kind} = reference'}, 'backgroundColor': '#5a4b16', 'color': '#fff7d6'},
                                        ],
                                    ),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                    ),
                ],
                className='g-3 mb-3',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div('💡 共性错例', className='h6 mb-2 fw-bold'),
                                    dash_table.DataTable(
                                        id='common-failure-table',
                                        columns=[
                                            {'name': '行号', 'id': 'line_no'},
                                            {'name': '数据集', 'id': 'dataset'},
                                            {'name': '原句', 'id': 'raw_text'},
                                            {'name': '标准', 'id': 'gold'},
                                            {'name': '次数', 'id': 'count'},
                                        ],
                                        data=common_failures.to_dict('records'),
                                        page_size=6,
                                        style_table={'overflowX': 'auto'},
                                        style_header={'backgroundColor': '#112240', 'color': '#e6eefc', 'fontWeight': 'bold'},
                                        style_cell={'backgroundColor': '#0f172a', 'color': '#dbe7ff', 'border': '1px solid rgba(93,126,182,0.18)', 'textAlign': 'left', 'whiteSpace': 'normal', 'fontSize': '0.75rem'},
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
                                    html.Div('📦 数据包', className='h6 mb-2 fw-bold'),
                                    html.Div(f"📋 {package_meta.get('package_name', '未知')}", className='mb-1 small'),
                                    html.Div(f"📝 {package_meta.get('line_count', 0)} 句", className='mb-1 small'),
                                    html.Div(f"⚙️ 启动: python app/leaderboard.py", className='text-secondary small mt-2'),
                                ]
                            ),
                            className='shadow-sm border-0',
                        ),
                        lg=4,
                    ),
                ],
                className='g-3 mb-3',
            ),
        ],
        fluid=True,
        className='pb-4',
    )

    @app.callback(
        Output('leaderboard-table', 'data'),
        Input('board-store', 'data'),
        Input('search-input', 'value'),
        Input('reference-toggle', 'value'),
        Input('mode-filter', 'value'),
        Input('status-filter', 'value'),
        Input('sort-key', 'value'),
    )
    def filter_board(rows: list[dict], query: str, reference_toggle: list[str], mode: str, status: str, sort_key: str) -> list[dict]:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return []
        include_reference = 'show' in (reference_toggle or [])
        if not include_reference and 'comparison_kind' in frame.columns:
            frame = frame[frame['comparison_kind'] != 'reference']
        if query:
            q = query.lower().strip()
            mask = frame.apply(lambda row: q in ' '.join(str(row.get(col, '')).lower() for col in ['submission_name', 'message']), axis=1)
            frame = frame[mask]
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
        Output('dataset-bar', 'figure'),
        Output('podium-section', 'children'),
        Output('rank-bar', 'figure'),
        Input('leaderboard-table', 'derived_virtual_data'),
        Input('leaderboard-table', 'selected_rows'),
        State('reports-store', 'data'),
    )
    def update_detail(rows: list[dict] | None, selected_rows: list[int] | None, reports_map: dict):
        frame = pd.DataFrame(rows or [])
        if frame.empty:
            return go.Figure(), go.Figure(), make_podium([]), make_rank_bar([])
        idx = selected_rows[0] if selected_rows else 0
        idx = min(max(idx, 0), len(frame) - 1)
        row = frame.iloc[idx]
        fig = make_profile_radar(row)
        dataset_bar = make_dataset_bar(row)  # 直接传递行数据而不是报告
        return fig, dataset_bar, make_podium(rows or []), make_rank_bar(rows or [])

    return app


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    app = create_app(results_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
