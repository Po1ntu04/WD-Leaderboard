from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import dash
from dash import Dash, Input, Output, dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRIC_LABELS = {
    'word_f1': '词级 F1',
    'word_precision': '词级 Precision',
    'word_recall': '词级 Recall',
    'boundary_f1': '边界 F1',
    'boundary_precision': '边界 Precision',
    'boundary_recall': '边界 Recall',
    'exact_match_sentence_rate': '整句完全匹配率',
    'over_segmentation_count': '过切数',
    'under_segmentation_count': '欠切数',
    'runtime_seconds': '运行时间',
    'discrimination_index': '区分度',
    'sentence_avg_word_f1': '句均词级 F1',
}

PLOT_LAYOUT = {
    'paper_bgcolor': '#ffffff',
    'plot_bgcolor': '#ffffff',
    'font_color': '#1e293b',
    'margin': dict(l=42, r=24, t=54, b=42),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the Chinese Word Segmentation analytics dashboard.')
    parser.add_argument('--results-dir', default='platform/results')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding='utf-8-sig')


def read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else {}


def load_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    submission_table = read_csv(results_dir / 'submission_table.csv')
    if submission_table.empty and (results_dir / 'leaderboard.json').exists():
        submission_table = pd.DataFrame(read_json(results_dir / 'leaderboard.json'))
    return {
        'leaderboard': pd.DataFrame(read_json(results_dir / 'leaderboard.json')),
        'sentence_table': read_csv(results_dir / 'sentence_table.csv'),
        'submission_table': submission_table,
        'sentence_score_table': read_csv(results_dir / 'sentence_score_table.csv'),
        'boundary_table': read_csv(results_dir / 'boundary_table.csv'),
        'span_error_table': read_csv(results_dir / 'span_error_table.csv'),
    }


def table_records(frame: pd.DataFrame) -> list[dict]:
    return frame.fillna('').to_dict('records') if not frame.empty else []


def score_bar_styles(columns: list[str]) -> list[dict[str, Any]]:
    styles: list[dict[str, Any]] = []
    score_columns = [column for column in columns if column in {'word_f1', 'boundary_f1', 'exact_match_sentence_rate'}]
    bands = [
        (0.0, 0.2, 'linear-gradient(90deg, rgba(37,99,235,.12) 20%, transparent 20%)'),
        (0.2, 0.4, 'linear-gradient(90deg, rgba(37,99,235,.16) 40%, transparent 40%)'),
        (0.4, 0.6, 'linear-gradient(90deg, rgba(37,99,235,.20) 60%, transparent 60%)'),
        (0.6, 0.8, 'linear-gradient(90deg, rgba(37,99,235,.24) 80%, transparent 80%)'),
        (0.8, 1.01, 'linear-gradient(90deg, rgba(37,99,235,.30) 100%, transparent 100%)'),
    ]
    for column in score_columns:
        for low, high, background in bands:
            styles.append({
                'if': {'filter_query': f'{{{column}}} >= {low} && {{{column}}} < {high}', 'column_id': column},
                'background': background,
                'fontWeight': '600',
            })
    return styles


def datatable(
    table_id: str,
    frame: pd.DataFrame,
    page_size: int = 12,
    visible_columns: list[str] | None = None,
    conditional: list[dict[str, Any]] | None = None,
) -> dash_table.DataTable:
    if visible_columns is not None and not frame.empty:
        columns_to_show = [column for column in visible_columns if column in frame.columns]
        frame = frame[columns_to_show].copy()
    columns = [{'name': column, 'id': column} for column in frame.columns] if not frame.empty else []
    data = table_records(frame)
    if table_id in {'leaderboard-table', 'leaderboard-preview-table'} and data:
        for row in data:
            try:
                rank = int(row.get('rank', 0))
            except Exception:
                rank = 0
            if rank == 1:
                row['rank'] = '🥇 1'
            elif rank == 2:
                row['rank'] = '🥈 2'
            elif rank == 3:
                row['rank'] = '🥉 3'
            try:
                issue_count = int(float(row.get('tolerant_issue_count') or 0))
            except Exception:
                issue_count = 0
            if issue_count > 0:
                row['tolerant_issue_count'] = f'⚠️ {issue_count}'
    base_conditional: list[dict[str, Any]] = [{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}]
    if 'status' in frame.columns:
        base_conditional.append({'if': {'filter_query': '{status} != "成功"'}, 'backgroundColor': '#fef2f2', 'color': '#991b1b'})
    if 'tolerant_issue_count' in frame.columns:
        issue_query = '{tolerant_issue_count} contains "⚠️"' if table_id in {'leaderboard-table', 'leaderboard-preview-table'} else '{tolerant_issue_count} > 0'
        base_conditional.append({'if': {'filter_query': issue_query, 'column_id': 'tolerant_issue_count'}, 'backgroundColor': '#fff7ed', 'color': '#9a3412', 'fontWeight': '700'})
    if 'rank' in frame.columns:
        base_conditional.extend([
            {'if': {'row_index': 0, 'column_id': 'rank'}, 'backgroundColor': '#fef3c7', 'color': '#92400e', 'fontWeight': '800'},
            {'if': {'row_index': 1, 'column_id': 'rank'}, 'backgroundColor': '#f1f5f9', 'color': '#334155', 'fontWeight': '800'},
            {'if': {'row_index': 2, 'column_id': 'rank'}, 'backgroundColor': '#ffedd5', 'color': '#9a3412', 'fontWeight': '800'},
        ])
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=data,
        page_size=page_size,
        sort_action='native',
        filter_action='native',
        fixed_rows={'headers': True},
        css=[{'selector': '.dash-spreadsheet-menu', 'rule': 'position: sticky; top: 0; z-index: 3;'}],
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#ffffff',
            'color': '#1e293b',
            'border': '1px solid #e2e8f0',
            'fontFamily': 'system-ui, -apple-system, Segoe UI, sans-serif',
            'fontSize': 12,
            'maxWidth': 360,
            'whiteSpace': 'normal',
            'height': 'auto',
            'padding': '8px',
        },
        style_header={'backgroundColor': '#eff6ff', 'fontWeight': '700', 'color': '#1e3a8a', 'border': '1px solid #bfdbfe'},
        style_data_conditional=[
            *base_conditional,
            *(conditional or []),
            *score_bar_styles(list(frame.columns)),
        ],
    )


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in METRIC_LABELS if column in frame.columns]


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=320, **PLOT_LAYOUT)
    return fig


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors='coerce').fillna(0.0)


def top_bar(submissions: pd.DataFrame, metric: str = 'word_f1') -> go.Figure:
    if submissions.empty or metric not in submissions.columns:
        return empty_figure('暂无排行榜数据')
    frame = submissions.copy()
    frame[metric] = numeric(frame, metric)
    top = frame.sort_values(metric, ascending=False).head(15).sort_values(metric, ascending=True)
    fig = px.bar(top, x=metric, y='submission_name', orientation='h', color=metric, color_continuous_scale='Blues')
    fig.update_layout(title=f'Top 15 - {METRIC_LABELS.get(metric, metric)}', xaxis_title=METRIC_LABELS.get(metric, metric), yaxis_title='', height=380, coloraxis_showscale=False, **PLOT_LAYOUT)
    return fig


def metric_heatmap(submissions: pd.DataFrame) -> go.Figure:
    cols = [column for column in ['word_f1', 'boundary_f1', 'exact_match_sentence_rate'] if column in submissions.columns]
    if submissions.empty or not cols:
        return empty_figure('暂无多指标数据')
    frame = submissions.sort_values('word_f1' if 'word_f1' in submissions.columns else cols[0], ascending=False).head(20)
    matrix = frame[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=[METRIC_LABELS.get(col, col) for col in cols], y=frame['submission_name'], zmin=0, zmax=1, colorscale='Blues', text=[[f'{value:.3f}' for value in row] for row in matrix.values], texttemplate='%{text}'))
    fig.update_layout(title='提交多指标热力图（来自 submission_table）', height=390, **PLOT_LAYOUT)
    return fig


def subset_score_heatmap(submissions: pd.DataFrame) -> go.Figure:
    cols = [column for column in submissions.columns if column.startswith(('source:', 'difficulty:', 'sentence_type:')) and column.endswith(':word_f1')]
    if submissions.empty or not cols:
        return empty_figure('暂无 subset 指标列')
    frame = submissions.sort_values('word_f1' if 'word_f1' in submissions.columns else cols[0], ascending=False).head(20)
    matrix = frame[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    labels = [col.replace(':word_f1', '').replace(':', ' / ') for col in cols]
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=labels, y=frame['submission_name'], zmin=0, zmax=1, colorscale='PuBu', text=[[f'{value:.3f}' for value in row] for row in matrix.values], texttemplate='%{text}'))
    fig.update_layout(title='Subset Score Heatmap（来自 submission_table）', height=480, **PLOT_LAYOUT)
    return fig


def sentence_scatter(sentence_table: pd.DataFrame) -> go.Figure:
    if sentence_table.empty:
        return empty_figure('暂无句子难度数据')
    frame = sentence_table.copy()
    y = 'sentence_avg_word_f1' if 'sentence_avg_word_f1' in frame.columns else 'difficulty_score'
    frame[y] = pd.to_numeric(frame.get(y), errors='coerce').fillna(0.0)
    frame['discrimination_index'] = pd.to_numeric(frame.get('discrimination_index', 0), errors='coerce').fillna(0.0)
    fig = px.scatter(frame, x='sentence_id', y=y, color='source' if 'source' in frame.columns else None, symbol='gold_status' if 'gold_status' in frame.columns else None, size='discrimination_index', hover_data=[c for c in ['difficulty', 'sentence_type', 'gold_status', 'raw_text'] if c in frame.columns], title='Sentence Difficulty Scatter Plot（来自 sentence_table）')
    fig.update_layout(height=430, **PLOT_LAYOUT)
    return fig


def error_counts(span_errors: pd.DataFrame, boundary_table: pd.DataFrame) -> go.Figure:
    rows = []
    if not span_errors.empty and 'error_type' in span_errors.columns:
        rows.extend({'type': key, 'count': int(value), 'family': 'word_span'} for key, value in span_errors['error_type'].value_counts().items())
    if not boundary_table.empty and 'boundary_type' in boundary_table.columns:
        rows.extend({'type': key, 'count': int(value), 'family': 'boundary'} for key, value in boundary_table['boundary_type'].value_counts().items())
    if not rows:
        return empty_figure('暂无错误数据')
    fig = px.bar(pd.DataFrame(rows), x='type', y='count', color='family', barmode='group', title='Error Type Bar Chart（来自 boundary/span_error tables）')
    fig.update_layout(height=360, **PLOT_LAYOUT)
    return fig


def rank_delta_view(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty or 'timestamp' not in submissions.columns or 'rank' not in submissions.columns:
        return empty_figure('暂无排名变化数据')
    frame = submissions.copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    frame['rank'] = pd.to_numeric(frame['rank'], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'rank']).sort_values('timestamp')
    if frame.empty:
        return empty_figure('暂无排名变化数据')
    fig = px.line(frame, x='timestamp', y='rank', color='submission_name', markers=True, title='Rank Delta View（来自 submission_table）')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=420, **PLOT_LAYOUT)
    return fig


def simple_word_cloud(words: Counter[str], title: str) -> html.Div:
    if not words:
        return html.Div('暂无词云数据。', className='text-muted')
    max_count = max(words.values())
    spans = []
    for word, count in words.most_common(60):
        size = 12 + 28 * (count / max_count)
        spans.append(html.Span(word, title=str(count), style={'fontSize': f'{size:.1f}px', 'margin': '6px 10px', 'display': 'inline-block', 'color': '#1d4ed8'}))
    return html.Div([html.Div(title, className='fw-bold mb-2'), html.Div(spans, style={'lineHeight': '2.2', 'background': '#ffffff', 'border': '1px solid #cbd5e1', 'borderRadius': '8px', 'padding': '16px'})])


def dataset_word_counter(sentence_table: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in sentence_table.get('gold', pd.Series(dtype=str)).dropna().astype(str):
        for token in text.replace(' / ', '/').split('/'):
            token = token.strip()
            if token:
                counter[token] += 1
    return counter


def error_word_counter(span_errors: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for column in ('raw_span', 'text', 'gold_span_tokens', 'pred_span_tokens'):
        if column not in span_errors.columns:
            continue
        for value in span_errors[column].dropna().astype(str):
            for token in value.replace(' / ', '/').split('/'):
                token = token.strip()
                if token:
                    counter[token] += 1
    return counter


def sankey_chart(boundary_table: pd.DataFrame) -> go.Figure:
    if boundary_table.empty or not {'source', 'boundary_type'}.issubset(boundary_table.columns):
        return empty_figure('暂无 Sankey 数据')
    counts = boundary_table.groupby(['source', 'boundary_type']).size().reset_index(name='count')
    labels = list(pd.unique(pd.concat([counts['source'], counts['boundary_type']], ignore_index=True)))
    idx = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(node=dict(label=labels, color='#bfdbfe'), link=dict(source=[idx[v] for v in counts['source']], target=[idx[v] for v in counts['boundary_type']], value=counts['count']))])
    fig.update_layout(title='Sankey Chart: source → boundary error type（来自 boundary_table）', height=420, **PLOT_LAYOUT)
    return fig


def clustering_scatter(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty or not {'word_f1', 'boundary_f1'}.issubset(submissions.columns):
        return empty_figure('暂无聚类数据')
    frame = submissions.copy()
    frame['word_f1'] = numeric(frame, 'word_f1')
    frame['boundary_f1'] = numeric(frame, 'boundary_f1')
    frame['exact_match_sentence_rate'] = numeric(frame, 'exact_match_sentence_rate')
    fig = px.scatter(frame, x='word_f1', y='boundary_f1', color='status' if 'status' in frame.columns else None, size='exact_match_sentence_rate', hover_name='submission_name', title='Student Clustering（metric-space scatter，来自 submission_table）')
    fig.update_layout(height=420, **PLOT_LAYOUT)
    return fig


def network_graph(sentence_scores: pd.DataFrame) -> go.Figure:
    if sentence_scores.empty or 'submission_name' not in sentence_scores.columns:
        return empty_figure('暂无网络图数据')
    names = sorted(sentence_scores['submission_name'].dropna().astype(str).unique().tolist())[:12]
    if len(names) < 2:
        return empty_figure('提交数量不足，无法构建相似网络')
    pivot = sentence_scores[sentence_scores['submission_name'].isin(names)].pivot_table(index='submission_name', columns='sentence_id', values='word_f1', fill_value=0.0)
    positions = {name: (0.5 + 0.42 * __import__('math').cos(2 * __import__('math').pi * i / len(names)), 0.5 + 0.42 * __import__('math').sin(2 * __import__('math').pi * i / len(names))) for i, name in enumerate(names)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for a, b in itertools.combinations(names, 2):
        va = pivot.loc[a]
        vb = pivot.loc[b]
        sim = 1.0 - float((va - vb).abs().mean())
        if sim >= 0.9:
            edge_x.extend([positions[a][0], positions[b][0], None])
            edge_y.extend([positions[a][1], positions[b][1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#94a3b8', width=1), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in names], y=[positions[n][1] for n in names], mode='markers+text', text=names, textposition='top center', marker=dict(size=12, color='#2563eb')))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title='Network Graph: similar sentence-score profiles（来自 sentence_score_table）', height=430, **PLOT_LAYOUT)
    return fig



LEADERBOARD_VISIBLE_COLUMNS = [
    'rank',
    'submission_name',
    'status',
    'word_f1',
    'boundary_f1',
    'exact_match_sentence_rate',
    'over_segmentation_count',
    'under_segmentation_count',
    'tolerant_issue_count',
    'runtime_seconds',
]


def section_title(title: str, subtitle: str = '') -> html.Div:
    children: list[Any] = [html.Div(title, className='section-title')]
    if subtitle:
        children.append(html.Div(subtitle, className='section-subtitle'))
    return html.Div(children, className='section-heading')


def panel(children: list[Any] | Any, class_name: str = '') -> html.Div:
    return html.Div(children, className=f'analytics-panel {class_name}'.strip())


def source_summary_figure(sentence_table: pd.DataFrame) -> go.Figure:
    if sentence_table.empty or 'source' not in sentence_table.columns:
        return empty_figure('暂无数据集来源摘要')
    counts = sentence_table.groupby('source', dropna=False).size().reset_index(name='sentence_count')
    counts = counts.sort_values('sentence_count', ascending=True)
    fig = px.bar(counts, x='sentence_count', y='source', orientation='h', color='sentence_count', color_continuous_scale='Blues', title='Dataset / Source Summary（来自 sentence_table）')
    fig.update_layout(height=320, coloraxis_showscale=False, **PLOT_LAYOUT)
    return fig


def source_summary_table(sentence_table: pd.DataFrame) -> pd.DataFrame:
    if sentence_table.empty or 'source' not in sentence_table.columns:
        return pd.DataFrame()
    table = sentence_table.groupby('source', dropna=False).agg(
        sentence_count=('sentence_id', 'count'),
        avg_char_len=('char_len', 'mean') if 'char_len' in sentence_table.columns else ('sentence_id', 'count'),
    ).reset_index()
    if 'gold_status' in sentence_table.columns:
        status = sentence_table.pivot_table(index='source', columns='gold_status', values='sentence_id', aggfunc='count', fill_value=0).reset_index()
        table = table.merge(status, on='source', how='left')
    if 'avg_char_len' in table.columns:
        table['avg_char_len'] = pd.to_numeric(table['avg_char_len'], errors='coerce').fillna(0).round(2)
    return table


def kpi_cards(submissions: pd.DataFrame, sentence_table: pd.DataFrame) -> list[dbc.Col]:
    success = submissions[submissions.get('status', '') == '成功'] if 'status' in submissions.columns else submissions
    best_word_f1 = numeric(success, 'word_f1').max() if not success.empty else 0.0
    best_boundary_f1 = numeric(success, 'boundary_f1').max() if not success.empty else 0.0
    exact = numeric(success, 'exact_match_sentence_rate').max() if not success.empty else 0.0
    total_issues = int(numeric(submissions, 'tolerant_issue_count').sum()) if not submissions.empty else 0
    excluded = int((sentence_table.get('gold_status', pd.Series(dtype=str)) == 'excluded').sum()) if not sentence_table.empty else 0
    values = [
        ('Submissions', len(submissions), '参与提交 / 方法数'),
        ('Sentences', len(sentence_table), f'excluded gold: {excluded}'),
        ('Best Word F1', f'{best_word_f1:.4f}', 'span-level word metric'),
        ('Best Boundary F1', f'{best_boundary_f1:.4f}', 'boundary-position metric'),
        ('Best Exact Match', f'{exact:.4f}', 'sentence-level exact rate'),
        ('Tolerant Issues', total_issues, 'row-level warnings'),
    ]
    return [
        dbc.Col(
            html.Div([
                html.Div(label, className='kpi-label'),
                html.Div(str(value), className='kpi-value'),
                html.Div(str(subtitle), className='kpi-subtitle'),
            ], className='kpi-card'),
            lg=2,
            md=4,
            sm=6,
        )
        for label, value, subtitle in values
    ]


def boundary_case_class(row: pd.Series) -> str:
    case = str(row.get('boundary_case', '')).upper()
    return {'TP': 'boundary-tp', 'FP': 'boundary-fp', 'FN': 'boundary-fn'}.get(case, 'boundary-none')


def character_boundary_diff(raw_text: str, boundary_rows: pd.DataFrame) -> html.Div:
    if not raw_text:
        return html.Div('暂无 raw text。', className='text-muted')
    boundary_lookup: dict[int, pd.Series] = {}
    if not boundary_rows.empty and 'boundary_position' in boundary_rows.columns:
        for _, row in boundary_rows.iterrows():
            try:
                boundary_lookup[int(row.get('boundary_position'))] = row
            except Exception:
                continue

    cells: list[Any] = []
    for index, char in enumerate(raw_text):
        boundary_position = index + 1
        marker = html.Span('', className='boundary-marker boundary-none')
        tooltip = ''
        if boundary_position in boundary_lookup and boundary_position < len(raw_text):
            row = boundary_lookup[boundary_position]
            marker = html.Span('', className=f'boundary-marker {boundary_case_class(row)}')
            tooltip = f"pos={boundary_position} case={row.get('boundary_case', '')} gold={row.get('gold_boundary', '')} pred={row.get('pred_boundary', '')}"
        cells.append(html.Span([html.Span(char, className='char-glyph'), marker], className='char-cell', title=tooltip))
    return html.Div([
        html.Div(cells, className='boundary-cell-grid'),
        html.Div([
            html.Span([html.Span('', className='legend-swatch boundary-tp'), 'TP: gold + pred'], className='boundary-legend-item'),
            html.Span([html.Span('', className='legend-swatch boundary-fp'), 'FP: pred only'], className='boundary-legend-item'),
            html.Span([html.Span('', className='legend-swatch boundary-fn'), 'FN: gold only'], className='boundary-legend-item'),
        ], className='boundary-legend'),
    ])


def create_app(results_dir: Path) -> Dash:
    tables = load_tables(results_dir)
    submissions = tables['submission_table']
    sentence_table = tables['sentence_table']
    sentence_scores = tables['sentence_score_table']
    boundary_table = tables['boundary_table']
    span_errors = tables['span_error_table']
    session_summary = read_json_object(results_dir / 'session_summary.json')

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='中文分词排行榜与分析看板', assets_folder=str((Path(__file__).resolve().parent / 'assets')))
    metric_options = [{'label': METRIC_LABELS.get(column, column), 'value': column} for column in metric_columns(submissions)]
    default_metric = 'word_f1' if 'word_f1' in submissions.columns else (metric_options[0]['value'] if metric_options else '')
    submission_options = [{'label': str(row.get('submission_name', '')), 'value': str(row.get('submission_name', ''))} for row in table_records(submissions)]
    default_submission = submission_options[0]['value'] if submission_options else ''
    sentence_options = [{'label': f"#{int(row.get('sentence_id', 0))} {str(row.get('raw_text', ''))[:28]}", 'value': int(row.get('sentence_id', 0))} for row in table_records(sentence_table)]
    default_sentence = sentence_options[0]['value'] if sentence_options else None
    data_version = str(session_summary.get('dataset_version') or Path(str(session_summary.get('raw_path', ''))).parent.name or results_dir.name)

    leaderboard_preview = submissions.head(15).copy() if not submissions.empty else submissions
    source_table = source_summary_table(sentence_table)

    overview_tab = html.Div([
        html.Div([
            html.Div('Chinese Word Segmentation Leaderboard', className='hero-title'),
            html.Div('Academic analytics dashboard for span-level scoring, boundary diagnostics, and gold review.', className='hero-subtitle'),
            html.Div([
                html.Span(f'Results: {results_dir}', className='hero-pill'),
                html.Span(f'Data version: {data_version}', className='hero-pill'),
            ], className='hero-meta'),
        ], className='dashboard-hero'),
        dbc.Row(kpi_cards(submissions, sentence_table), className='g-3 mb-4'),
        dbc.Row([
            dbc.Col(panel([section_title('Leaderboard Preview', 'Top submissions with official ranking columns.'), datatable('leaderboard-preview-table', leaderboard_preview, page_size=10, visible_columns=LEADERBOARD_VISIBLE_COLUMNS)]), lg=7),
            dbc.Col(panel([section_title('Top 15', 'Primary metric ranking preview.'), dcc.Graph(figure=top_bar(submissions, default_metric), className='dashboard-graph')]), lg=5),
        ], className='g-3'),
        dbc.Row([
            dbc.Col(panel([section_title('Metric Heatmap', 'Cross-metric comparison for leading submissions.'), dcc.Graph(figure=metric_heatmap(submissions), className='dashboard-graph')]), lg=7),
            dbc.Col(panel([section_title('Dataset / Source Summary', 'Sentence distribution by source.'), dcc.Graph(figure=source_summary_figure(sentence_table), className='dashboard-graph'), datatable('source-summary-table', source_table, page_size=8)]), lg=5),
        ], className='g-3 mt-1'),
    ], className='tab-body')

    leaderboard_tab = html.Div([
        panel([
            section_title('Official Leaderboard', 'Default columns emphasize ranking metrics; subset metrics are available in details.'),
            datatable('leaderboard-table', submissions, page_size=18, visible_columns=LEADERBOARD_VISIBLE_COLUMNS),
            html.Details([html.Summary('Show full submission table with subset columns', className='details-summary'), datatable('leaderboard-full-table', submissions, page_size=12)], className='details-panel'),
        ]),
        dbc.Row([
            dbc.Col(panel([section_title('Subset Score Heatmap', 'Scores by source, difficulty, and sentence type.'), dcc.Graph(figure=subset_score_heatmap(submissions), className='dashboard-graph')]), lg=7),
            dbc.Col(panel([section_title('Rank Delta View', 'Timestamp order view for ranking changes.'), dcc.Graph(figure=rank_delta_view(submissions), className='dashboard-graph')]), lg=5),
        ], className='g-3 mt-1'),
        panel([
            section_title('Student / Method Profile', 'Submission-level radar and weakest sentence breakdown.'),
            dbc.Row(dbc.Col(dcc.Dropdown(id='profile-submission', options=submission_options, value=default_submission, clearable=False), md=5), className='mb-3'),
            dbc.Row([dbc.Col(dcc.Graph(id='profile-radar'), lg=5), dbc.Col(dcc.Graph(id='profile-sentence-bar'), lg=7)], className='g-3'),
            html.Div(id='profile-summary', className='mt-2'),
            datatable('profile-sentence-table', sentence_scores.head(0), page_size=12),
        ], 'mt-3'),
    ], className='tab-body')

    diagnostics_tab = html.Div([
        panel([
            section_title('Boundary Diff Viewer', 'Character-level boundary comparison for a selected submission and sentence.'),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='boundary-submission', options=submission_options, value=default_submission, clearable=False), md=5),
                dbc.Col(dcc.Dropdown(id='boundary-sentence', options=sentence_options, value=default_sentence, clearable=False), md=7),
            ], className='mb-3'),
            html.Div(id='boundary-diff-viewer'),
        ]),
        dbc.Row([
            dbc.Col(panel([section_title('Error Type Bar Chart', 'Aggregated boundary and span-error categories.'), dcc.Graph(figure=error_counts(span_errors, boundary_table), className='dashboard-graph')]), lg=5),
            dbc.Col(panel([section_title('Sentence Difficulty Map', 'Average sentence score and discrimination by source.'), dcc.Graph(figure=sentence_scatter(sentence_table), className='dashboard-graph')]), lg=7),
        ], className='g-3 mt-1'),
        panel([
            section_title('Diagnostics Long Tables', 'Raw rows are expandable; these tables come directly from exported artifacts.'),
            html.Details([html.Summary('Show span_error_table', className='details-summary'), datatable('span-error-table', span_errors, page_size=12)], className='details-panel'),
            html.Details([html.Summary('Show boundary_table', className='details-summary'), datatable('boundary-table', boundary_table, page_size=12)], className='details-panel'),
            html.Details([html.Summary('Show sentence_table', className='details-summary'), datatable('sentence-difficulty-table', sentence_table, page_size=12)], className='details-panel'),
        ], 'mt-3'),
    ], className='tab-body')

    gold_review_tab = html.Div([
        panel([
            section_title('Gold Review Console', 'Review confirmed / suspicious / excluded gold rows. Excluded rows are visible but not ranked.'),
            dbc.Alert('gold_status=excluded rows are excluded from official denominators; suspicious rows remain scored and should be reviewed.', color='info', className='academic-alert'),
            datatable('gold-review-table', sentence_table, page_size=15),
        ]),
    ], className='tab-body')

    experimental_tab = html.Div([
        dbc.Alert('Experimental visualizations are exploratory only and are not used for official ranking.', color='warning', className='academic-alert'),
        dbc.Row([
            dbc.Col(panel(simple_word_cloud(dataset_word_counter(sentence_table), 'Dataset overview word cloud（来自 sentence_table.gold）')), lg=6),
            dbc.Col(panel(simple_word_cloud(error_word_counter(span_errors), 'Common error spans word cloud（来自 span_error_table.raw_span）')), lg=6),
        ], className='g-3'),
        dbc.Row([
            dbc.Col(panel([section_title('Sankey Chart', 'Source to boundary error flow.'), dcc.Graph(figure=sankey_chart(boundary_table), className='dashboard-graph')]), lg=6),
            dbc.Col(panel([section_title('Student Clustering', 'Metric-space exploratory scatter.'), dcc.Graph(figure=clustering_scatter(submissions), className='dashboard-graph')]), lg=6),
        ], className='g-3 mt-1'),
        panel([section_title('Network Graph', 'Exploratory similarity network from sentence-score profiles.'), dcc.Graph(figure=network_graph(sentence_scores), className='dashboard-graph')], 'mt-3'),
    ], className='tab-body experimental-tab')

    app.layout = dbc.Container([
        dbc.Tabs([
            dbc.Tab(overview_tab, label='Overview', tab_id='overview'),
            dbc.Tab(leaderboard_tab, label='Leaderboard', tab_id='leaderboard'),
            dbc.Tab(diagnostics_tab, label='Diagnostics', tab_id='diagnostics'),
            dbc.Tab(gold_review_tab, label='Gold Review', tab_id='gold-review'),
            dbc.Tab(experimental_tab, label='Experimental', tab_id='experimental'),
        ], id='main-tabs', active_tab='overview', className='main-nav-tabs'),
    ], fluid=True, className='dashboard-shell')

    @app.callback(Output('boundary-diff-viewer', 'children'), Input('boundary-submission', 'value'), Input('boundary-sentence', 'value'))
    def update_boundary_diff(submission_name: str, sentence_id: int) -> Any:
        sentence_row = sentence_table[sentence_table['sentence_id'] == sentence_id] if not sentence_table.empty and sentence_id is not None else pd.DataFrame()
        boundary_rows = boundary_table[(boundary_table.get('submission_name', '') == submission_name) & (boundary_table.get('sentence_id', -1) == sentence_id)] if not boundary_table.empty else pd.DataFrame()
        score_rows = sentence_scores[(sentence_scores.get('submission_name', '') == submission_name) & (sentence_scores.get('sentence_id', -1) == sentence_id)] if not sentence_scores.empty else pd.DataFrame()
        raw = str(sentence_row.iloc[0].get('raw_text', '')) if not sentence_row.empty else ''
        metadata = ''
        if not score_rows.empty:
            row = score_rows.iloc[0]
            metadata = f"validation={row.get('validation_status', '')} ｜ word_f1={row.get('word_f1', 0)} ｜ boundary_f1={row.get('boundary_f1', 0)}"
        return panel([
            html.Div(f'句子 #{sentence_id}: {raw}', className='boundary-sentence-title'),
            html.Div(metadata, className='boundary-sentence-meta'),
            character_boundary_diff(raw, boundary_rows),
            html.Div('Selected sentence score', className='subsection-title'),
            datatable('selected-sentence-score', score_rows, page_size=4),
            html.Details([html.Summary('Show raw boundary rows', className='details-summary'), datatable('selected-boundary-diff', boundary_rows, page_size=10)], className='details-panel'),
        ])

    @app.callback(Output('profile-radar', 'figure'), Output('profile-sentence-bar', 'figure'), Output('profile-summary', 'children'), Output('profile-sentence-table', 'data'), Output('profile-sentence-table', 'columns'), Input('profile-submission', 'value'))
    def update_profile(submission_name: str):
        if submissions.empty or not submission_name:
            return empty_figure('暂无提交'), empty_figure('暂无句级数据'), '', [], []
        row = submissions[submissions['submission_name'].astype(str) == str(submission_name)]
        if row.empty:
            return empty_figure('未找到提交'), empty_figure('未找到句级数据'), '', [], []
        series = row.iloc[0]
        radar_cols = [column for column in ['word_f1', 'boundary_f1', 'exact_match_sentence_rate'] if column in submissions.columns]
        radar = go.Figure(data=[go.Scatterpolar(r=[float(series.get(column, 0) or 0) for column in radar_cols] + ([float(series.get(radar_cols[0], 0) or 0)] if radar_cols else []), theta=[METRIC_LABELS.get(column, column) for column in radar_cols] + ([METRIC_LABELS.get(radar_cols[0], radar_cols[0])] if radar_cols else []), fill='toself', name=str(submission_name), line=dict(color='#2563eb'))])
        radar.update_layout(paper_bgcolor='#ffffff', font_color='#1e293b', polar=dict(radialaxis=dict(range=[0, 1])), height=330, title='Profile radar（来自 submission_table）')
        rows = sentence_scores[sentence_scores['submission_name'].astype(str) == str(submission_name)].copy() if not sentence_scores.empty else pd.DataFrame()
        if not rows.empty:
            rows['word_f1'] = pd.to_numeric(rows.get('word_f1'), errors='coerce')
            bar = px.bar(rows.sort_values('word_f1').head(20), x='word_f1', y='sentence_id', orientation='h', color='source', title='Lowest word-F1 sentences（来自 sentence_score_table）')
            bar.update_layout(height=330, **PLOT_LAYOUT)
        else:
            bar = empty_figure('暂无句级数据')
        summary = dbc.Alert(f"状态：{series.get('status', '')} ｜ 词级F1：{series.get('word_f1', 0)} ｜ 边界F1：{series.get('boundary_f1', 0)} ｜ 过切：{series.get('over_segmentation_count', 0)} ｜ 欠切：{series.get('under_segmentation_count', 0)}", color='primary', className='academic-alert')
        columns = [{'name': column, 'id': column} for column in rows.columns]
        return radar, bar, summary, table_records(rows), columns

    return app

def main() -> None:
    args = parse_args()
    app = create_app(Path(args.results_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
