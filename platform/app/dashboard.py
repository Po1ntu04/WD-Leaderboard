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


def datatable(table_id: str, frame: pd.DataFrame, page_size: int = 12) -> dash_table.DataTable:
    columns = [{'name': column, 'id': column} for column in frame.columns] if not frame.empty else []
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=table_records(frame),
        page_size=page_size,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#ffffff',
            'color': '#1e293b',
            'border': '1px solid #cbd5e1',
            'fontFamily': 'system-ui, -apple-system, Segoe UI, sans-serif',
            'fontSize': 12,
            'maxWidth': 360,
            'whiteSpace': 'normal',
            'height': 'auto',
            'padding': '6px',
        },
        style_header={'backgroundColor': '#e2e8f0', 'fontWeight': '700', 'color': '#0f172a'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}],
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
    if 'text' in span_errors.columns:
        for token in span_errors['text'].dropna().astype(str):
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


def kpi_cards(submissions: pd.DataFrame, sentence_table: pd.DataFrame) -> list[dbc.Col]:
    success = submissions[submissions.get('status', '') == '成功'] if 'status' in submissions.columns else submissions
    best_word_f1 = numeric(success, 'word_f1').max() if not success.empty else 0.0
    best_boundary_f1 = numeric(success, 'boundary_f1').max() if not success.empty else 0.0
    excluded = int((sentence_table.get('gold_status', pd.Series(dtype=str)) == 'excluded').sum()) if not sentence_table.empty else 0
    values = [('提交数', len(submissions)), ('句子数', len(sentence_table)), ('排除 gold', excluded), ('最佳词级 F1', f'{best_word_f1:.4f}'), ('最佳边界 F1', f'{best_boundary_f1:.4f}')]
    return [dbc.Col(dbc.Card(dbc.CardBody([html.Div(label, className='text-secondary small'), html.Div(str(value), className='h4 fw-bold text-primary')]), className='border-0 shadow-sm'), md=2) for label, value in values]


def create_app(results_dir: Path) -> Dash:
    tables = load_tables(results_dir)
    submissions = tables['submission_table']
    sentence_table = tables['sentence_table']
    sentence_scores = tables['sentence_score_table']
    boundary_table = tables['boundary_table']
    span_errors = tables['span_error_table']

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='中文分词排行榜与分析看板', assets_folder=str((Path(__file__).resolve().parent / 'assets')))
    metric_options = [{'label': METRIC_LABELS.get(column, column), 'value': column} for column in metric_columns(submissions)]
    default_metric = 'word_f1' if 'word_f1' in submissions.columns else (metric_options[0]['value'] if metric_options else '')
    submission_options = [{'label': str(row.get('submission_name', '')), 'value': str(row.get('submission_name', ''))} for row in table_records(submissions)]
    default_submission = submission_options[0]['value'] if submission_options else ''
    sentence_options = [{'label': f"#{int(row.get('sentence_id', 0))} {str(row.get('raw_text', ''))[:28]}", 'value': int(row.get('sentence_id', 0))} for row in table_records(sentence_table)]
    default_sentence = sentence_options[0]['value'] if sentence_options else None

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.Div('中文分词排行榜与视觉分析平台', className='h2 fw-bold mt-4 mb-1 text-primary'))),
        dbc.Row(dbc.Col(html.Div(f'所有图表均只读取导出表：{results_dir}', className='text-secondary mb-3'))),
        dbc.Row(kpi_cards(submissions, sentence_table), className='g-3 mb-3'),
        dbc.Tabs([
            dbc.Tab([dbc.Row([dbc.Col(dcc.Graph(figure=top_bar(submissions, default_metric)), lg=6), dbc.Col(dcc.Graph(figure=metric_heatmap(submissions)), lg=6)], className='g-3 mt-2')], label='P0-1 Overview Dashboard'),
            dbc.Tab([html.Div(className='mt-3'), datatable('leaderboard-table', submissions, page_size=18)], label='P0-2 Leaderboard Table'),
            dbc.Tab([dcc.Graph(figure=subset_score_heatmap(submissions), className='mt-3')], label='P0-3 Subset Score Heatmap'),
            dbc.Tab([dbc.Row([dbc.Col(dcc.Dropdown(id='boundary-submission', options=submission_options, value=default_submission, clearable=False), md=5), dbc.Col(dcc.Dropdown(id='boundary-sentence', options=sentence_options, value=default_sentence, clearable=False), md=7)], className='mt-3 mb-2'), html.Div(id='boundary-diff-viewer')], label='P0-4 Boundary Diff Viewer'),
            dbc.Tab([dcc.Graph(figure=sentence_scatter(sentence_table), className='mt-3'), datatable('sentence-difficulty-table', sentence_table, page_size=12)], label='P0-5 Sentence Difficulty Scatter Plot'),
            dbc.Tab([dbc.Row(dbc.Col(dcc.Dropdown(id='profile-submission', options=submission_options, value=default_submission, clearable=False), md=5), className='mt-3 mb-2'), dbc.Row([dbc.Col(dcc.Graph(id='profile-radar'), lg=5), dbc.Col(dcc.Graph(id='profile-sentence-bar'), lg=7)], className='g-3'), html.Div(id='profile-summary', className='mt-2'), datatable('profile-sentence-table', sentence_scores.head(0), page_size=12)], label='P1-6 Student / Method Profile'),
            dbc.Tab([dcc.Graph(figure=error_counts(span_errors, boundary_table), className='mt-3'), datatable('span-error-table', span_errors, page_size=12), html.Div(className='mt-3'), datatable('boundary-table', boundary_table, page_size=12)], label='P1-7 Error Type Bar Chart'),
            dbc.Tab([dcc.Graph(figure=rank_delta_view(submissions), className='mt-3')], label='P1-8 Rank Delta View'),
            dbc.Tab([html.Div('Gold Review Console：复核 gold_status=confirmed/suspicious/excluded；excluded 不参与排名评分。', className='text-secondary mt-3 mb-2'), datatable('gold-review-table', sentence_table, page_size=15)], label='P1-9 Gold Review Console'),
            dbc.Tab([dbc.Row([dbc.Col(simple_word_cloud(dataset_word_counter(sentence_table), 'Dataset overview word cloud（来自 sentence_table.gold）'), lg=6), dbc.Col(simple_word_cloud(error_word_counter(span_errors), 'Common error spans word cloud（来自 span_error_table.text）'), lg=6)], className='g-3 mt-3')], label='P2-10 Word Cloud'),
            dbc.Tab([dcc.Graph(figure=sankey_chart(boundary_table), className='mt-3')], label='P2-11 Sankey Chart'),
            dbc.Tab([dcc.Graph(figure=clustering_scatter(submissions), className='mt-3')], label='P2-12 Student Clustering'),
            dbc.Tab([dcc.Graph(figure=network_graph(sentence_scores), className='mt-3')], label='P2-13 Network Graph'),
        ])
    ], fluid=True, style={'backgroundColor': '#f8fafc', 'minHeight': '100vh', 'paddingBottom': '32px'})

    @app.callback(Output('boundary-diff-viewer', 'children'), Input('boundary-submission', 'value'), Input('boundary-sentence', 'value'))
    def update_boundary_diff(submission_name: str, sentence_id: int) -> Any:
        sentence_row = sentence_table[sentence_table['sentence_id'] == sentence_id] if not sentence_table.empty and sentence_id is not None else pd.DataFrame()
        boundary_rows = boundary_table[(boundary_table.get('submission_name', '') == submission_name) & (boundary_table.get('sentence_id', -1) == sentence_id)] if not boundary_table.empty else pd.DataFrame()
        score_rows = sentence_scores[(sentence_scores.get('submission_name', '') == submission_name) & (sentence_scores.get('sentence_id', -1) == sentence_id)] if not sentence_scores.empty else pd.DataFrame()
        raw = sentence_row.iloc[0].get('raw_text', '') if not sentence_row.empty else ''
        return html.Div([dbc.Alert(f'句子 #{sentence_id}: {raw}', color='light'), html.Div('句级得分', className='fw-bold'), datatable('selected-sentence-score', score_rows, page_size=4), html.Div('边界差异 true_positive / over_segmentation / under_segmentation', className='fw-bold mt-3'), datatable('selected-boundary-diff', boundary_rows, page_size=10)])

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
            bar = px.bar(rows.sort_values('word_f1').head(20), x='word_f1', y='sentence_id', orientation='h', color='source', title='最低词级 F1 的句子 Top 20（来自 sentence_score_table）')
            bar.update_layout(height=330, **PLOT_LAYOUT)
        else:
            bar = empty_figure('暂无句级数据')
        summary = dbc.Alert(f"状态：{series.get('status', '')} ｜ 词级F1：{series.get('word_f1', 0)} ｜ 边界F1：{series.get('boundary_f1', 0)} ｜ 过切：{series.get('over_segmentation_count', 0)} ｜ 欠切：{series.get('under_segmentation_count', 0)}", color='primary')
        columns = [{'name': column, 'id': column} for column in rows.columns]
        return radar, bar, summary, table_records(rows), columns

    return app


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.results_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
