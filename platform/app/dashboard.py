from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import dash
from dash import Dash, Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRIC_LABELS = {
    'word_f1': 'Word F1',
    'word_precision': 'Word Precision',
    'word_recall': 'Word Recall',
    'boundary_f1': 'Boundary F1',
    'boundary_precision': 'Boundary Precision',
    'boundary_recall': 'Boundary Recall',
    'exact_match_sentence_rate': 'Exact Match',
    'over_segmentation_count': 'Over-seg',
    'under_segmentation_count': 'Under-seg',
    'runtime_seconds': 'Runtime',
    'discrimination_index': 'Discrimination',
    'sentence_avg_word_f1': 'Avg Word F1',
}

DISPLAY_LABELS = {
    'rank': 'Rank',
    'submission_name': 'Submission',
    'submission_group': 'Group',
    'mode': 'Mode',
    'status': 'Status',
    'timestamp': 'Timestamp',
    'runtime_seconds': 'Runtime',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1',
    'word_precision': 'Word Precision',
    'word_recall': 'Word Recall',
    'word_f1': 'Word F1',
    'boundary_precision': 'Boundary Precision',
    'boundary_recall': 'Boundary Recall',
    'boundary_f1': 'Boundary F1',
    'exact_match_sentence_rate': 'Exact Match',
    'over_segmentation_count': 'Over-seg',
    'under_segmentation_count': 'Under-seg',
    'tolerant_issue_count': 'Warnings',
    'sentence_id': 'Sentence ID',
    'source': 'Source',
    'difficulty': 'Difficulty',
    'sentence_type': 'Type',
    'gold_status': 'Gold Status',
    'validation_status': 'Validation',
    'pred_valid': 'Pred Valid',
    'is_evaluable': 'Evaluable',
    'exact_match': 'Exact Match',
    'raw_text': 'Raw Text',
    'raw_text_preview': 'Raw Preview',
    'gold': 'Gold',
    'gold_preview': 'Gold Preview',
    'review_flags': 'Review Flags',
    'review_reason': 'Review Reason',
    'sentence_avg_word_f1': 'Avg Word F1',
    'sentence_avg_boundary_f1': 'Avg Boundary F1',
    'avg_f1': 'Avg F1',
    'discrimination_index': 'Discrimination',
    'boundary_position': 'Boundary Pos',
    'left_char': 'Left',
    'right_char': 'Right',
    'left_context': 'Left Context',
    'right_context': 'Right Context',
    'gold_boundary': 'Gold Boundary',
    'pred_boundary': 'Pred Boundary',
    'boundary_case': 'Boundary Case',
    'boundary_type': 'Boundary Type',
    'raw_span': 'Raw Span',
    'gold_span_tokens': 'Gold Span Tokens',
    'pred_span_tokens': 'Pred Span Tokens',
    'start_char': 'Start',
    'end_char': 'End',
    'error_type': 'Error Type',
    'main_error_type': 'Main Error',
    'severity': 'Severity',
}

SCORE_COLUMNS = {
    'precision',
    'recall',
    'f1',
    'word_precision',
    'word_recall',
    'word_f1',
    'boundary_precision',
    'boundary_recall',
    'boundary_f1',
    'exact_match_sentence_rate',
    'sentence_avg_word_f1',
    'sentence_avg_boundary_f1',
    'sentence_exact_match_rate',
    'avg_f1',
    'discrimination_index',
}

COUNT_COLUMNS = {
    'rank',
    'over_segmentation_count',
    'under_segmentation_count',
    'tolerant_issue_count',
    'wrong_sentence_count',
    'sentence_id',
    'line_no',
    'char_len',
    'gold_word_count',
    'pred_word_count',
    'correct_word_count',
    'gold_boundary_count',
    'pred_boundary_count',
    'correct_boundary_count',
    'participant_count',
    'exact_match',
    'pred_valid',
    'is_evaluable',
    'is_scored',
    'boundary_position',
    'gold_boundary',
    'pred_boundary',
    'start_char',
    'end_char',
    'severity',
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


def display_label(column: str) -> str:
    if column in DISPLAY_LABELS:
        return DISPLAY_LABELS[column]
    if column.startswith(('source:', 'difficulty:', 'sentence_type:')):
        return column.replace(':word_f1', '').replace(':', ' / ')
    return column.replace('_', ' ').title()


def format_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if column in SCORE_COLUMNS or column.endswith(':word_f1') or column.endswith(':boundary_f1'):
            out[column] = pd.to_numeric(out[column], errors='coerce').round(4)
        elif column == 'runtime_seconds':
            out[column] = pd.to_numeric(out[column], errors='coerce').round(3)
        elif column in COUNT_COLUMNS:
            values = pd.to_numeric(out[column], errors='coerce')
            out[column] = values.map(lambda value: '' if pd.isna(value) else int(value))
    return out


def text_preview(value: Any, max_len: int = 52) -> str:
    text = str(value or '')
    return text if len(text) <= max_len else f'{text[: max_len - 1]}…'


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
    filter_action: str = 'native',
    sort_action: str = 'native',
    row_selectable: str | None = None,
    selected_rows: list[int] | None = None,
) -> dash_table.DataTable:
    if visible_columns is not None and not frame.empty:
        columns_to_show = [column for column in visible_columns if column in frame.columns]
        frame = frame[columns_to_show].copy()
    frame = format_frame(frame)
    columns = [{'name': display_label(column), 'id': column} for column in frame.columns] if not frame.empty else []
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
        sort_action=sort_action,
        filter_action=filter_action,
        row_selectable=row_selectable,
        selected_rows=selected_rows or [],
        css=[{'selector': '.dash-spreadsheet-menu', 'rule': 'position: sticky; top: 0; z-index: 3;'}],
        style_table={'overflowX': 'auto', 'maxWidth': '100%'},
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
    fig.update_layout(title=title, height=360, **PLOT_LAYOUT)
    return fig


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors='coerce').fillna(0.0)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return default if pd.isna(value) else value


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def top_bar(submissions: pd.DataFrame, metric: str = 'word_f1') -> go.Figure:
    if submissions.empty or metric not in submissions.columns:
        return empty_figure('暂无排行榜数据')
    frame = submissions.copy()
    frame[metric] = numeric(frame, metric)
    top = frame.sort_values(metric, ascending=False).head(15).sort_values(metric, ascending=True)
    top['display_name'] = top['submission_name'].astype(str).map(lambda value: text_preview(value, 22))
    top['value_text'] = top[metric].map(lambda value: f'{float(value):.4f}')
    fig = px.bar(top, x=metric, y='display_name', orientation='h', color=metric, color_continuous_scale='Blues', text='value_text', hover_name='submission_name')
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(title=f'Top 15 - {METRIC_LABELS.get(metric, metric)}', xaxis_title=METRIC_LABELS.get(metric, metric), yaxis_title='', height=410, coloraxis_showscale=False, **PLOT_LAYOUT)
    fig.update_xaxes(range=[0, min(1.05, max(0.1, float(top[metric].max()) * 1.12))])
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
    fig.update_layout(title='Subset Score Heatmap（来自 submission_table）', height=450, **PLOT_LAYOUT)
    return fig


def sentence_scatter(sentence_table: pd.DataFrame) -> go.Figure:
    if sentence_table.empty:
        return empty_figure('暂无句子难度数据')
    frame = sentence_table.copy()
    y = 'sentence_avg_word_f1' if 'sentence_avg_word_f1' in frame.columns else 'difficulty_score'
    frame[y] = pd.to_numeric(frame.get(y), errors='coerce').fillna(0.0)
    frame['avg_f1'] = frame[y]
    frame['raw_text_preview'] = frame.get('raw_text', pd.Series(dtype=str)).fillna('').astype(str).map(lambda value: text_preview(value, 80))
    frame['discrimination_index'] = pd.to_numeric(frame.get('discrimination_index', 0), errors='coerce').fillna(0.0)
    fig = px.scatter(
        frame,
        x='sentence_id',
        y=y,
        color='source' if 'source' in frame.columns else None,
        symbol='gold_status' if 'gold_status' in frame.columns else None,
        size='discrimination_index',
        hover_data={
            'raw_text_preview': True,
            'source': True,
            'difficulty': True,
            'gold_status': True,
            'avg_f1': ':.4f',
            'discrimination_index': ':.4f',
            'sentence_id': True,
        },
        title='Sentence Difficulty Scatter Plot（来自 sentence_table）',
    )
    fig.update_layout(height=430, **PLOT_LAYOUT)
    return fig


def error_counts(span_errors: pd.DataFrame, boundary_table: pd.DataFrame) -> go.Figure:
    rows = []
    if not span_errors.empty and 'error_type' in span_errors.columns:
        rows.extend({'type': key, 'count': int(value), 'family': 'word_span'} for key, value in span_errors['error_type'].value_counts().items())
    if not boundary_table.empty and 'boundary_type' in boundary_table.columns:
        boundary_errors = boundary_table[boundary_table['boundary_type'].isin(['over_segmentation', 'under_segmentation'])]
        rows.extend({'type': key, 'count': int(value), 'family': 'boundary'} for key, value in boundary_errors['boundary_type'].value_counts().items())
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


def metric_rank_comparison(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty or not {'submission_name', 'word_f1', 'boundary_f1'}.issubset(submissions.columns):
        return empty_figure('暂无 metric-scope rank 数据')
    frame = submissions.copy()
    frame['word_f1'] = numeric(frame, 'word_f1')
    frame['boundary_f1'] = numeric(frame, 'boundary_f1')
    frame['word_rank'] = frame['word_f1'].rank(method='min', ascending=False).astype(int)
    frame['boundary_rank'] = frame['boundary_f1'].rank(method='min', ascending=False).astype(int)
    if 'rank' in frame.columns:
        frame['rank'] = pd.to_numeric(frame['rank'], errors='coerce').fillna(999999).astype(int)
        frame = frame.sort_values('rank').head(20)
    else:
        frame = frame.sort_values('word_rank').head(20)
    rows = []
    for _, row in frame.iterrows():
        rows.append({'submission_name': row['submission_name'], 'metric': 'Word F1 rank', 'rank': row['word_rank'], 'score': row['word_f1']})
        rows.append({'submission_name': row['submission_name'], 'metric': 'Boundary F1 rank', 'rank': row['boundary_rank'], 'score': row['boundary_f1']})
    long = pd.DataFrame(rows)
    fig = px.scatter(
        long,
        x='rank',
        y='submission_name',
        color='metric',
        symbol='metric',
        hover_data={'score': ':.4f', 'rank': True},
        title='Metric-scope Rank Comparison（lower rank is better）',
    )
    for name, group in long.groupby('submission_name'):
        if len(group) == 2:
            fig.add_trace(go.Scatter(
                x=group['rank'],
                y=group['submission_name'],
                mode='lines',
                line=dict(color='#cbd5e1', width=1.3),
                hoverinfo='skip',
                showlegend=False,
            ))
    fig.update_xaxes(autorange='reversed', title='Rank')
    fig.update_layout(height=430, yaxis_title='', **PLOT_LAYOUT)
    return fig


def truncate_token(token: str, max_len: int = 18) -> str:
    token = str(token)
    return token if len(token) <= max_len else f"{token[: max_len - 1]}…"


def counter_to_frame(words: Counter[str], top_n: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        [{"token": token, "display_token": truncate_token(token), "count": int(count)} for token, count in words.most_common(top_n)]
    )


def token_bar_chart(words: Counter[str], title: str, top_n: int = 25) -> go.Figure:
    frame = counter_to_frame(words, top_n=top_n)
    if frame.empty:
        return empty_figure('暂无 token frequency 数据')
    frame = frame.sort_values('count', ascending=True)
    fig = px.bar(
        frame,
        x='count',
        y='display_token',
        orientation='h',
        hover_data={'token': True, 'display_token': False, 'count': True},
        color='count',
        color_continuous_scale='Blues',
        title=title,
    )
    fig.update_layout(height=400, yaxis_title='', xaxis_title='Count', coloraxis_showscale=False, **PLOT_LAYOUT)
    fig.update_traces(marker_line_width=0, cliponaxis=False)
    return fig


def simple_word_cloud(words: Counter[str], title: str) -> html.Div:
    if not words:
        return html.Div('暂无词云数据。', className='text-muted')
    max_count = max(words.values())
    spans = []
    for word, count in words.most_common(45):
        size = min(24, 11 + 16 * (count / max_count))
        display = truncate_token(word, max_len=16)
        spans.append(html.Span(display, title=f'{word} ({count})', className='token-cloud-chip', style={'fontSize': f'{size:.1f}px'}))
    return html.Div([html.Div(title, className='fw-bold mb-2'), html.Div(spans, className='token-cloud')])


def dataset_word_counter(sentence_table: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in sentence_table.get('gold', pd.Series(dtype=str)).dropna().astype(str):
        for token in text.replace(' / ', '/').split('/'):
            token = token.strip()
            if token:
                counter[token] += 1
    return counter


def error_span_counter(span_errors: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    if 'raw_span' not in span_errors.columns:
        return counter
    for token in span_errors['raw_span'].dropna().astype(str):
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


ERROR_COLOR_MAP = {
    'over_segmentation': '#f97316',
    'under_segmentation': '#dc2626',
    'true_positive': '#16a34a',
}


def sankey_chart(boundary_table: pd.DataFrame, *, normalized: bool = False) -> go.Figure:
    required = {'source', 'boundary_type'}
    if boundary_table.empty or not required.issubset(boundary_table.columns):
        return empty_figure('暂无 Sankey 数据')
    frame = boundary_table[boundary_table['boundary_type'].isin(['over_segmentation', 'under_segmentation'])].copy()
    if frame.empty:
        return empty_figure('暂无错误流数据（已排除 true_positive）')
    counts = frame.groupby(['source', 'boundary_type']).size().reset_index(name='count')
    if normalized:
        totals = counts.groupby('source')['count'].transform('sum')
        counts['value'] = counts['count'] / totals.replace(0, 1)
        value_label = 'normalized share'
        title = 'Normalized Error Flow — This chart visualizes error flow only, not official ranking.'
    else:
        counts['value'] = counts['count']
        value_label = 'count'
        title = 'Error Flow — This chart visualizes error flow only, not official ranking.'
    labels = list(pd.unique(pd.concat([counts['source'].astype(str), counts['boundary_type'].astype(str)], ignore_index=True)))
    idx = {label: i for i, label in enumerate(labels)}
    node_colors = ['#dbeafe' if label not in ERROR_COLOR_MAP else ERROR_COLOR_MAP[label] for label in labels]
    link_colors = [ERROR_COLOR_MAP.get(str(row['boundary_type']), '#94a3b8') for _, row in counts.iterrows()]
    custom = [f"count={int(row['count'])}<br>{value_label}={float(row['value']):.4f}" for _, row in counts.iterrows()]
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(label=labels, color=node_colors, pad=12, thickness=14),
        link=dict(
            source=[idx[str(v)] for v in counts['source']],
            target=[idx[str(v)] for v in counts['boundary_type']],
            value=counts['value'],
            color=link_colors,
            customdata=custom,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>',
        ),
    )])
    fig.update_layout(title=title, height=420, **PLOT_LAYOUT)
    return fig


def clustering_scatter(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty or not {'word_f1', 'boundary_f1'}.issubset(submissions.columns):
        return empty_figure('暂无 metric space 数据')
    frame = submissions.copy()
    frame['word_f1'] = numeric(frame, 'word_f1')
    frame['boundary_f1'] = numeric(frame, 'boundary_f1')
    frame['exact_match_sentence_rate'] = numeric(frame, 'exact_match_sentence_rate')
    frame['tolerant_issue_count'] = numeric(frame, 'tolerant_issue_count')
    fig = px.scatter(
        frame,
        x='word_f1',
        y='boundary_f1',
        color='status' if 'status' in frame.columns else None,
        size='exact_match_sentence_rate',
        hover_name='submission_name',
        hover_data={
            'word_f1': ':.4f',
            'boundary_f1': ':.4f',
            'exact_match_sentence_rate': ':.4f',
            'tolerant_issue_count': ':.0f',
        },
        title='Metric Space Scatter（not clustering; official ranking is unchanged）',
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#94a3b8', width=1, dash='dash'), name='y = x'))
    fig.update_xaxes(range=[0, 1], title='Word F1')
    fig.update_yaxes(range=[0, 1], title='Boundary F1')
    fig.update_layout(height=420, **PLOT_LAYOUT)
    return fig


def short_label(name: str, max_len: int = 10) -> str:
    name = str(name)
    return name if len(name) <= max_len else f'{name[: max_len - 1]}…'


def network_graph(sentence_scores: pd.DataFrame, submissions: pd.DataFrame | None = None, *, top_n: int = 20, k: int = 2) -> go.Figure:
    if sentence_scores.empty or 'submission_name' not in sentence_scores.columns:
        return empty_figure('暂无网络图数据')
    if submissions is not None and not submissions.empty and {'submission_name', 'rank'}.issubset(submissions.columns):
        ranked = submissions.copy()
        ranked['rank'] = pd.to_numeric(ranked['rank'], errors='coerce')
        names = ranked.sort_values('rank')['submission_name'].dropna().astype(str).head(top_n).tolist()
    else:
        names = sorted(sentence_scores['submission_name'].dropna().astype(str).unique().tolist())[:top_n]
    if len(names) < 3:
        return empty_figure('提交数量不足，无法构建相似网络')
    pivot = sentence_scores[sentence_scores['submission_name'].isin(names)].pivot_table(index='submission_name', columns='sentence_id', values='word_f1', fill_value=0.0)
    names = list(pivot.index.astype(str))
    edges: set[tuple[str, str]] = set()
    edge_scores: dict[tuple[str, str], float] = {}
    for source in names:
        sims: list[tuple[str, float]] = []
        for target in names:
            if source == target:
                continue
            sim = 1.0 - float((pivot.loc[source] - pivot.loc[target]).abs().mean())
            sims.append((target, sim))
        for target, sim in sorted(sims, key=lambda item: item[1], reverse=True)[:k]:
            edge = tuple(sorted((source, target)))
            edges.add(edge)
            edge_scores[edge] = max(edge_scores.get(edge, 0.0), sim)
    if len(edges) < 3:
        return empty_figure('相似边少于 3 条；不绘制空网络。')

    import math

    positions = {name: (0.5 + 0.38 * math.cos(2 * math.pi * i / len(names)), 0.5 + 0.38 * math.sin(2 * math.pi * i / len(names))) for i, name in enumerate(names)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_text: list[str] = []
    for a, b in sorted(edges):
        edge_x.extend([positions[a][0], positions[b][0], None])
        edge_y.extend([positions[a][1], positions[b][1], None])
        edge_text.extend([f'{a} ↔ {b}<br>similarity={edge_scores[(a, b)]:.4f}', f'{a} ↔ {b}<br>similarity={edge_scores[(a, b)]:.4f}', ''])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#cbd5e1', width=1.2), text=edge_text, hoverinfo='text', name='2-nearest-neighbor links'))
    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in names],
        y=[positions[n][1] for n in names],
        mode='markers+text',
        text=[short_label(n) for n in names],
        customdata=names,
        hovertemplate='%{customdata}<extra></extra>',
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(size=12, color='#2563eb', line=dict(color='#ffffff', width=1)),
        name='submissions',
    ))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1], scaleanchor='x', scaleratio=1)
    fig.update_layout(title='Similarity Network: top-ranked 20, k=2 nearest neighbors（exploratory only）', height=430, showlegend=False, **PLOT_LAYOUT)
    return fig


def network_visual(sentence_scores: pd.DataFrame, submissions: pd.DataFrame | None = None) -> Any:
    fig = network_graph(sentence_scores, submissions)
    if not fig.data:
        title = getattr(fig.layout, 'title', None)
        message = title.text if title and title.text else 'Similarity network is not available for this result set.'
        return html.Div(message, className='network-note text-muted')
    return dashboard_graph(figure=fig)


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


def dashboard_graph(figure: go.Figure | None = None, graph_id: str | None = None) -> dcc.Graph:
    kwargs: dict[str, Any] = {
        'figure': figure,
        'className': 'dashboard-graph',
        'responsive': True,
        'config': {'responsive': True, 'displayModeBar': False},
    }
    if graph_id is not None:
        kwargs['id'] = graph_id
    return dcc.Graph(**kwargs)


def source_summary_figure(sentence_table: pd.DataFrame) -> go.Figure:
    if sentence_table.empty or 'source' not in sentence_table.columns:
        return empty_figure('暂无数据集来源摘要')
    counts = sentence_table.groupby('source', dropna=False).size().reset_index(name='sentence_count')
    counts = counts.sort_values('sentence_count', ascending=True)
    fig = px.bar(counts, x='sentence_count', y='source', orientation='h', color='sentence_count', color_continuous_scale='Blues', title='Dataset / Source Summary（来自 sentence_table）')
    fig.update_layout(height=360, coloraxis_showscale=False, **PLOT_LAYOUT)
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


def source_summary_cards(sentence_table: pd.DataFrame) -> html.Div:
    if sentence_table.empty:
        return html.Div('暂无 source summary。', className='text-muted')
    source_count = int(sentence_table['source'].nunique()) if 'source' in sentence_table.columns else 0
    largest = ''
    if 'source' in sentence_table.columns:
        counts = sentence_table['source'].fillna('unknown').astype(str).value_counts()
        if not counts.empty:
            largest = f'{counts.index[0]} ({int(counts.iloc[0])})'
    suspicious = int((sentence_table.get('gold_status', pd.Series(dtype=str)) == 'suspicious').sum())
    excluded = int((sentence_table.get('gold_status', pd.Series(dtype=str)) == 'excluded').sum())
    avg_len = pd.to_numeric(sentence_table.get('char_len', pd.Series(dtype=float)), errors='coerce').mean()
    items = [
        ('Sources', source_count),
        ('Largest Source', largest or '—'),
        ('Avg Length', f'{avg_len:.1f}' if pd.notna(avg_len) else '—'),
        ('Suspicious / Excluded', f'{suspicious} / {excluded}'),
    ]
    return html.Div([html.Div([html.Div(label, className='mini-card-label'), html.Div(str(value), className='mini-card-value')], className='summary-mini-card') for label, value in items], className='summary-card-grid')


def kpi_cards(submissions: pd.DataFrame, sentence_table: pd.DataFrame) -> list[dbc.Col]:
    success = submissions[submissions.get('status', '') == '成功'] if 'status' in submissions.columns else submissions
    best_word_f1 = numeric(success, 'word_f1').max() if not success.empty else 0.0
    best_boundary_f1 = numeric(success, 'boundary_f1').max() if not success.empty else 0.0
    exact = numeric(success, 'exact_match_sentence_rate').max() if not success.empty else 0.0
    best_word_name = '—'
    best_boundary_name = '—'
    if not success.empty and 'submission_name' in success.columns:
        word_values = numeric(success, 'word_f1')
        boundary_values = numeric(success, 'boundary_f1')
        if not word_values.empty:
            best_word_name = text_preview(success.loc[word_values.idxmax(), 'submission_name'], 22)
        if not boundary_values.empty:
            best_boundary_name = text_preview(success.loc[boundary_values.idxmax(), 'submission_name'], 22)
    total_issues = int(numeric(submissions, 'tolerant_issue_count').sum()) if not submissions.empty else 0
    excluded = int((sentence_table.get('gold_status', pd.Series(dtype=str)) == 'excluded').sum()) if not sentence_table.empty else 0
    values = [
        ('Submissions', len(submissions), '参与提交 / 方法数'),
        ('Sentences', len(sentence_table), f'excluded gold: {excluded}'),
        ('Best Word F1', f'{best_word_f1:.4f}', best_word_name),
        ('Best Boundary F1', f'{best_boundary_f1:.4f}', best_boundary_name),
        ('Best Exact Match', f'{exact:.4f}', 'strict sentence-level exact rate'),
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
    since_break = 0
    punctuation_breaks = set('，。！？；：、,.!?;:')
    for index, char in enumerate(raw_text):
        boundary_position = index + 1
        marker = html.Span('', className='boundary-marker boundary-none')
        tooltip = ''
        if boundary_position in boundary_lookup and boundary_position < len(raw_text):
            row = boundary_lookup[boundary_position]
            marker = html.Span('', className=f'boundary-marker {boundary_case_class(row)}')
            tooltip = f"pos={boundary_position} case={row.get('boundary_case', '')} gold={row.get('gold_boundary', '')} pred={row.get('pred_boundary', '')}"
        cells.append(html.Span([html.Span(char, className='char-glyph'), marker], className='char-cell', title=tooltip))
        since_break += 1
        if (char in punctuation_breaks and since_break >= 18) or since_break >= 38:
            cells.append(html.Span('', className='boundary-line-break'))
            since_break = 0
    return html.Div([
        html.Div(cells, className='boundary-cell-grid'),
        html.Div([
            html.Span([html.Span('', className='legend-swatch boundary-tp'), 'TP: gold + pred'], className='boundary-legend-item'),
            html.Span([html.Span('', className='legend-swatch boundary-fp'), 'FP: pred only'], className='boundary-legend-item'),
            html.Span([html.Span('', className='legend-swatch boundary-fn'), 'FN: gold only'], className='boundary-legend-item'),
        ], className='boundary-legend'),
    ])


def tokens_from_boundaries(raw_text: str, boundaries: list[int] | set[int]) -> list[str]:
    clean_boundaries = sorted({int(boundary) for boundary in boundaries if 0 < int(boundary) < len(raw_text)})
    starts = [0] + clean_boundaries
    ends = clean_boundaries + [len(raw_text)]
    return [raw_text[start:end] for start, end in zip(starts, ends)]


def token_row(tokens: list[str], label: str, class_name: str = '', reference_tokens: list[str] | None = None) -> html.Div:
    reference = Counter(reference_tokens or [])
    seen: Counter[str] = Counter()
    chips = []
    for token in tokens:
        seen[token] += 1
        mismatch = bool(reference_tokens is not None and seen[token] > reference.get(token, 0))
        chips.append(html.Span(token, className=f"token-chip {class_name} {'token-mismatch' if mismatch else ''}".strip(), title=token))
    return html.Div([html.Div(label, className='token-row-label'), html.Div(chips, className='token-row-wrap')], className='token-row')


def sentence_review_frame(sentence_table: pd.DataFrame) -> pd.DataFrame:
    if sentence_table.empty:
        return pd.DataFrame()
    frame = sentence_table.copy()
    frame['avg_f1'] = pd.to_numeric(frame.get('sentence_avg_word_f1', 0), errors='coerce').fillna(0.0)
    frame['raw_text_preview'] = frame.get('raw_text', pd.Series(dtype=str)).fillna('').astype(str).map(lambda value: text_preview(value, 72))
    frame['gold_preview'] = frame.get('gold', pd.Series(dtype=str)).fillna('').astype(str).map(lambda value: text_preview(value, 72))
    discrim = pd.to_numeric(frame.get('discrimination_index', 0), errors='coerce').fillna(0.0)
    if 'review_flags' not in frame.columns:
        frame['review_flags'] = ''
    frame['review_flags'] = frame['review_flags'].fillna('').astype(str)
    frame['review_reason'] = [
        review_reason(status, avg, disc, flags)
        for status, avg, disc, flags in zip(frame.get('gold_status', ''), frame['avg_f1'], discrim, frame['review_flags'])
    ]
    columns = ['sentence_id', 'source', 'difficulty', 'gold_status', 'avg_f1', 'discrimination_index', 'review_reason', 'raw_text_preview', 'gold_preview', 'review_flags']
    return frame[[column for column in columns if column in frame.columns]]


def review_reason(gold_status: Any, avg_f1: float, discrimination: float, flags: Any = '') -> str:
    reasons: list[str] = []
    if safe_float(avg_f1) < 0.5:
        reasons.append('low average F1')
    if safe_float(discrimination) >= 0.25:
        reasons.append('high discrimination')
    flag_text = str(flags or '').lower()
    if flag_text:
        reasons.append('existing review flag')
    if str(gold_status) == 'suspicious':
        reasons.append('possible gold ambiguity')
    if str(gold_status) == 'excluded':
        reasons.append('excluded from ranking')
    return '; '.join(reasons) or 'routine review'


def gold_review_cards(sentence_table: pd.DataFrame) -> list[dbc.Col]:
    if sentence_table.empty:
        values = [('confirmed', 'Confirmed', 0), ('suspicious', 'Suspicious', 0), ('excluded', 'Excluded', 0), ('low_avg', 'Low Avg F1', 0), ('high_discrimination', 'High Discrimination', 0)]
    else:
        status = sentence_table.get('gold_status', pd.Series(dtype=str)).fillna('').astype(str)
        avg_f1 = pd.to_numeric(sentence_table.get('sentence_avg_word_f1', 0), errors='coerce').fillna(0.0)
        discrim = pd.to_numeric(sentence_table.get('discrimination_index', 0), errors='coerce').fillna(0.0)
        values = [
            ('confirmed', 'Confirmed', int((status == 'confirmed').sum())),
            ('suspicious', 'Suspicious', int((status == 'suspicious').sum())),
            ('excluded', 'Excluded', int((status == 'excluded').sum())),
            ('low_avg', 'Low Avg F1', int((avg_f1 < 0.5).sum())),
            ('high_discrimination', 'High Discrimination', int((discrim >= 0.25).sum())),
        ]
    return [
        dbc.Col(
            html.Button(
                [html.Div(label, className='kpi-label'), html.Div(str(value), className='kpi-value')],
                id=f'gold-card-{key}',
                n_clicks=0,
                className='kpi-card compact-kpi review-filter-card',
                title='Click to filter review rows',
            ),
            lg=2,
            md=4,
            sm=6,
        )
        for key, label, value in values
    ]


def profile_subset_frame(row: pd.Series) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prefixes = [('source:', 'Source'), ('difficulty:', 'Difficulty'), ('sentence_type:', 'Type')]
    for prefix, group in prefixes:
        for column, value in row.items():
            if str(column).startswith(prefix) and str(column).endswith(':word_f1'):
                label = str(column).replace(prefix, '').replace(':word_f1', '')
                records.append({'group': group, 'subset': label, 'word_f1': safe_float(value)})
    return pd.DataFrame(records)


def profile_subset_bar(row: pd.Series) -> go.Figure:
    frame = profile_subset_frame(row)
    if frame.empty:
        return empty_figure('No subset scores available')
    frame = frame.sort_values('word_f1', ascending=True)
    fig = px.bar(frame, x='word_f1', y='subset', color='group', orientation='h', text=frame['word_f1'].map(lambda value: f'{value:.4f}'), title='Subset Scores by Source / Difficulty / Type')
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_xaxes(range=[0, 1.05], title='Word F1')
    fig.update_layout(height=420, yaxis_title='', **PLOT_LAYOUT)
    return fig


def error_distribution_figure(submission_name: str, span_errors: pd.DataFrame) -> go.Figure:
    if span_errors.empty or 'error_type' not in span_errors.columns:
        return empty_figure('No error distribution available')
    all_counts = span_errors.groupby(['submission_name', 'error_type']).size().reset_index(name='count')
    avg_counts = all_counts.groupby('error_type')['count'].mean().reset_index(name='All-submission avg')
    selected = span_errors[span_errors['submission_name'].astype(str) == str(submission_name)].groupby('error_type').size().reset_index(name='Selected')
    merged = avg_counts.merge(selected, on='error_type', how='outer').fillna(0)
    long = merged.melt(id_vars='error_type', var_name='series', value_name='count')
    fig = px.bar(long, x='error_type', y='count', color='series', barmode='group', title='Error Type Distribution vs All-submission Average')
    fig.update_layout(height=360, xaxis_title='', yaxis_title='Count', **PLOT_LAYOUT)
    return fig


def lowest_sentence_table(submission_name: str, sentence_scores: pd.DataFrame, sentence_table: pd.DataFrame, span_errors: pd.DataFrame) -> pd.DataFrame:
    if sentence_scores.empty:
        return pd.DataFrame()
    rows = sentence_scores[sentence_scores['submission_name'].astype(str) == str(submission_name)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows['word_f1'] = pd.to_numeric(rows.get('word_f1'), errors='coerce').fillna(0.0)
    rows['boundary_f1'] = pd.to_numeric(rows.get('boundary_f1'), errors='coerce').fillna(0.0)
    columns = ['sentence_id', 'source', 'word_f1', 'boundary_f1', 'validation_status']
    out = rows.sort_values(['word_f1', 'boundary_f1']).head(20)[columns].copy()
    if not sentence_table.empty and {'sentence_id', 'raw_text'}.issubset(sentence_table.columns):
        preview = sentence_table[['sentence_id', 'raw_text']].copy()
        preview['raw_text_preview'] = preview['raw_text'].map(lambda value: text_preview(value, 72))
        out = out.merge(preview[['sentence_id', 'raw_text_preview']], on='sentence_id', how='left')
    if not span_errors.empty and {'submission_name', 'sentence_id', 'error_type'}.issubset(span_errors.columns):
        err = span_errors[span_errors['submission_name'].astype(str) == str(submission_name)]
        if not err.empty:
            main = err.groupby(['sentence_id', 'error_type']).size().reset_index(name='count').sort_values(['sentence_id', 'count'], ascending=[True, False]).drop_duplicates('sentence_id')
            out = out.merge(main[['sentence_id', 'error_type']], on='sentence_id', how='left')
    if 'error_type' not in out.columns:
        out['error_type'] = ''
    return out[['sentence_id', 'source', 'raw_text_preview', 'word_f1', 'boundary_f1', 'error_type', 'validation_status']]


def filter_options(frame: pd.DataFrame, column: str) -> list[dict[str, str]]:
    if frame.empty or column not in frame.columns:
        return []
    values = sorted(value for value in frame[column].fillna('').astype(str).unique() if value)
    return [{'label': value, 'value': value} for value in values]


def review_flag_options(sentence_table: pd.DataFrame) -> list[dict[str, str]]:
    if sentence_table.empty or 'review_flags' not in sentence_table.columns:
        return []
    flags: set[str] = set()
    for value in sentence_table['review_flags'].dropna().astype(str):
        for part in value.replace('|', ',').replace(';', ',').split(','):
            part = part.strip()
            if part:
                flags.add(part)
    return [{'label': flag, 'value': flag} for flag in sorted(flags)]


def apply_gold_filters(
    sentence_table: pd.DataFrame,
    gold_status: list[str] | None,
    sources: list[str] | None,
    difficulties: list[str] | None,
    review_flags: list[str] | None,
    avg_range: list[float] | None,
    quick_filter: dict[str, Any] | None = None,
) -> pd.DataFrame:
    frame = sentence_review_frame(sentence_table)
    if frame.empty:
        return frame
    if gold_status:
        frame = frame[frame.get('gold_status', '').astype(str).isin(gold_status)]
    if sources:
        frame = frame[frame.get('source', '').astype(str).isin(sources)]
    if difficulties:
        frame = frame[frame.get('difficulty', '').astype(str).isin(difficulties)]
    if review_flags:
        flags = [str(flag) for flag in review_flags]
        frame = frame[frame.get('review_flags', '').fillna('').astype(str).map(lambda value: any(flag in value for flag in flags))]
    if avg_range and len(avg_range) == 2:
        lo, hi = float(avg_range[0]), float(avg_range[1])
        avg = pd.to_numeric(frame.get('avg_f1', 0), errors='coerce').fillna(0.0)
        frame = frame[(avg >= lo) & (avg <= hi)]
    quick_type = (quick_filter or {}).get('type')
    if quick_type == 'low_avg':
        frame = frame[pd.to_numeric(frame.get('avg_f1', 0), errors='coerce').fillna(0.0) < 0.5]
    elif quick_type == 'high_discrimination':
        frame = frame[pd.to_numeric(frame.get('discrimination_index', 0), errors='coerce').fillna(0.0) >= 0.25]
    elif quick_type in {'confirmed', 'suspicious', 'excluded'}:
        frame = frame[frame.get('gold_status', '').astype(str) == quick_type]
    return frame.sort_values(['avg_f1', 'discrimination_index'], ascending=[True, False])


def gold_detail_card(sentence_id: int | None, sentence_table: pd.DataFrame, span_errors: pd.DataFrame, sentence_scores: pd.DataFrame) -> html.Div:
    if sentence_id is None or sentence_table.empty:
        return html.Div('Select a sentence to inspect gold details.', className='text-muted')
    row = sentence_table[sentence_table['sentence_id'] == sentence_id]
    if row.empty:
        return html.Div('Selected sentence is no longer visible after filtering.', className='text-muted')
    series = row.iloc[0]
    errors = span_errors[span_errors.get('sentence_id', -1) == sentence_id].copy() if not span_errors.empty else pd.DataFrame()
    variants: list[Any] = []
    if not errors.empty and 'pred_span_tokens' in errors.columns:
        counts = errors['pred_span_tokens'].dropna().astype(str).value_counts().head(6)
        variants = [html.Span(f'{text} ×{count}', className='variant-chip', title=text) for text, count in counts.items()]
    top_errors = pd.DataFrame()
    if not errors.empty:
        cols = [column for column in ['raw_span', 'gold_span_tokens', 'pred_span_tokens', 'error_type', 'severity'] if column in errors.columns]
        top_errors = errors.sort_values('severity', ascending=False).head(8)[cols]
    score_rows = sentence_scores[sentence_scores.get('sentence_id', -1) == sentence_id] if not sentence_scores.empty else pd.DataFrame()
    if not score_rows.empty and 'word_f1' in score_rows.columns:
        score_note = f"Average submitted Word F1: {pd.to_numeric(score_rows['word_f1'], errors='coerce').mean():.4f}"
    else:
        score_note = ''
    return html.Div([
        html.Div(f"Sentence #{sentence_id}", className='review-detail-title'),
        html.Div(score_note, className='review-detail-meta'),
        html.Div('Raw text', className='subsection-title'),
        html.Div(str(series.get('raw_text', '')), className='review-text-block'),
        html.Div('Gold segmentation', className='subsection-title'),
        html.Div(str(series.get('gold', '')), className='review-text-block gold-text'),
        html.Div('Review reason', className='subsection-title'),
        html.Div(review_reason(series.get('gold_status', ''), series.get('sentence_avg_word_f1', 0), series.get('discrimination_index', 0), series.get('review_flags', '')), className='review-text-block'),
        html.Div('Common prediction variants', className='subsection-title'),
        html.Div(variants or [html.Span('No repeated variants available.', className='text-muted')], className='variant-row'),
        html.Div('Top span errors', className='subsection-title'),
        datatable('gold-detail-errors-default', top_errors, page_size=8, filter_action='none'),
        html.Details([
            html.Summary('Show raw span error rows for this sentence', className='details-summary'),
            datatable('gold-detail-errors', errors, page_size=8),
        ], className='details-panel'),
    ], className='review-detail-card')


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

    if not submissions.empty and 'rank' in submissions.columns:
        submissions_ranked = submissions.assign(_rank=pd.to_numeric(submissions['rank'], errors='coerce')).sort_values('_rank').drop(columns=['_rank'])
        leaderboard_preview = submissions_ranked.head(12)
    else:
        submissions_ranked = submissions
        leaderboard_preview = submissions.head(12).copy() if not submissions.empty else submissions
    gold_review_initial = apply_gold_filters(sentence_table, None, None, None, None, [0, 1])

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
            dbc.Col(panel([section_title('Leaderboard Preview', 'Top submissions with official ranking columns.'), datatable('leaderboard-preview-table', leaderboard_preview, page_size=12, visible_columns=LEADERBOARD_VISIBLE_COLUMNS, filter_action='none', sort_action='none')]), lg=7),
            dbc.Col(panel([section_title('Top 15', 'Primary metric ranking preview.'), dashboard_graph(figure=top_bar(submissions, default_metric))]), lg=5),
        ], className='g-3 overview-equal-row'),
        dbc.Row([
            dbc.Col(panel([section_title('Metric Heatmap', 'Cross-metric comparison for leading submissions.'), dashboard_graph(figure=metric_heatmap(submissions))]), lg=7),
            dbc.Col(panel([section_title('Dataset / Source Summary', 'Sentence distribution by source.'), source_summary_cards(sentence_table), dashboard_graph(figure=source_summary_figure(sentence_table))]), lg=5),
        ], className='g-3 mt-1 overview-equal-row'),
    ], className='tab-body')

    leaderboard_tab = html.Div([
        panel([
            section_title('Official Leaderboard', 'Default columns emphasize ranking metrics; subset metrics are available in details.'),
            datatable('leaderboard-table', submissions_ranked, page_size=18, visible_columns=LEADERBOARD_VISIBLE_COLUMNS),
            html.Details([html.Summary('Full metrics table', className='details-summary'), datatable('leaderboard-full-table', submissions_ranked, page_size=12)], className='details-panel'),
        ]),
        dbc.Row([
            dbc.Col(panel([section_title('Subset Score Heatmap', 'Scores by source, difficulty, and sentence type.'), dashboard_graph(figure=subset_score_heatmap(submissions))]), lg=7),
            dbc.Col(panel([section_title('Metric-scope Rank Comparison', 'Compares official Word F1 and Boundary F1 ranks without using timestamps.'), dashboard_graph(figure=metric_rank_comparison(submissions))]), lg=5),
        ], className='g-3 mt-1'),
        panel([
            section_title('Student / Method Profile', 'Subset performance, weakest sentences, and error mix for one submission.'),
            dbc.Row(dbc.Col(dcc.Dropdown(id='profile-submission', options=submission_options, value=default_submission, clearable=False), md=5), className='mb-3'),
            html.Div(id='profile-summary', className='mt-2'),
            dbc.Row([dbc.Col(dashboard_graph(graph_id='profile-subset-bar'), lg=7), dbc.Col(dashboard_graph(graph_id='profile-error-distribution'), lg=5)], className='g-3'),
            html.Div('Lowest word-F1 sentences', className='subsection-title'),
            datatable('profile-sentence-table', sentence_scores.head(0), page_size=12, filter_action='none'),
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
            dbc.Col(panel([section_title('Error Type Bar Chart', 'Aggregated boundary and span-error categories; true_positive is hidden.'), dashboard_graph(figure=error_counts(span_errors, boundary_table))]), lg=5),
            dbc.Col(panel([section_title('Sentence Difficulty Map', 'Average sentence score and discrimination by source.'), dashboard_graph(figure=sentence_scatter(sentence_table))]), lg=7),
        ], className='g-3 mt-1'),
        panel([
            section_title('Developer / Raw Artifact Views', 'Collapsed by default; these long tables are for audit and handoff validation.'),
            html.Details([html.Summary('Show span_error_table', className='details-summary'), datatable('span-error-table', span_errors, page_size=12)], className='details-panel'),
            html.Details([html.Summary('Show boundary_table', className='details-summary'), datatable('boundary-table', boundary_table, page_size=12)], className='details-panel'),
            html.Details([html.Summary('Show sentence_table', className='details-summary'), datatable('sentence-difficulty-table', sentence_table, page_size=12)], className='details-panel'),
        ], 'mt-3'),
    ], className='tab-body')

    gold_review_tab = html.Div([
        panel([
            section_title('Gold Review Console', 'Review confirmed / suspicious / excluded gold rows. Excluded rows are visible but not ranked.'),
            dbc.Alert('gold_status=excluded rows are excluded from official denominators; suspicious rows remain scored and should be reviewed.', color='info', className='academic-alert'),
            dcc.Store(id='gold-quick-filter', data={}),
            dbc.Row(gold_review_cards(sentence_table), className='g-3 mb-3'),
            html.Div([
                html.Div([html.Label('Gold Status'), dcc.Dropdown(id='gold-status-filter', options=filter_options(sentence_table, 'gold_status'), multi=True)], className='filter-control'),
                html.Div([html.Label('Source'), dcc.Dropdown(id='gold-source-filter', options=filter_options(sentence_table, 'source'), multi=True)], className='filter-control'),
                html.Div([html.Label('Difficulty'), dcc.Dropdown(id='gold-difficulty-filter', options=filter_options(sentence_table, 'difficulty'), multi=True)], className='filter-control'),
                html.Div([html.Label('Review Flags'), dcc.Dropdown(id='gold-review-flags-filter', options=review_flag_options(sentence_table), multi=True)], className='filter-control'),
                html.Div([html.Label('Avg F1 Range'), dcc.RangeSlider(id='gold-avg-f1-range', min=0, max=1, step=0.01, value=[0, 1], tooltip={'placement': 'bottom', 'always_visible': False})], className='filter-control wide-filter'),
            ], className='filter-grid'),
            datatable('gold-review-table', gold_review_initial, page_size=12, filter_action='none', row_selectable='single', selected_rows=[0] if not gold_review_initial.empty else []),
            html.Div(id='gold-detail-card', className='mt-3'),
            html.Details([html.Summary('Show raw full sentence_table', className='details-summary'), datatable('gold-full-table', sentence_table, page_size=12)], className='details-panel'),
        ]),
    ], className='tab-body')

    experimental_note = 'Experimental visualizations are exploratory only and are not used for official ranking.'

    def experimental_section(title: str, subtitle: str, children: list[Any], *, open_by_default: bool = False) -> html.Details:
        return html.Details([
            html.Summary([html.Span(title, className='experimental-summary-title'), html.Span(subtitle, className='experimental-summary-subtitle')], className='experimental-summary'),
            html.Div(children, className='experimental-section-body'),
        ], open=open_by_default, className='experimental-section')

    dataset_tokens = dataset_word_counter(sentence_table)
    error_spans = error_span_counter(span_errors)
    experimental_tab = html.Div([
        dbc.Alert(experimental_note, color='warning', className='academic-alert'),
        experimental_section('Token Frequency', 'Top-N bars first; compact word clouds are auxiliary.', [
            dbc.Row([
                dbc.Col(panel([section_title('Dataset top tokens', 'Top tokens from sentence_table.gold.'), dashboard_graph(figure=token_bar_chart(dataset_tokens, 'Dataset Top Tokens'))]), lg=6),
                dbc.Col(panel([section_title('Common error spans', 'Top local raw spans from span_error_table.raw_span.'), dashboard_graph(figure=token_bar_chart(error_spans, 'Common Error Spans'))]), lg=6),
            ], className='g-3'),
        ], open_by_default=True),
        experimental_section('Error Flow', 'Sankey charts exclude true_positive and visualize errors only.', [
            dbc.Row([
                dbc.Col(panel([section_title('Raw error flow', 'Only over_segmentation and under_segmentation are shown.'), dashboard_graph(figure=sankey_chart(boundary_table, normalized=False))]), lg=6),
                dbc.Col(panel([section_title('Normalized error flow', 'Within-source normalization prevents large subsets from dominating.'), dashboard_graph(figure=sankey_chart(boundary_table, normalized=True))]), lg=6),
            ], className='g-3'),
        ], open_by_default=False),
        experimental_section('Metric Space', 'Scatter plot of official metrics, not a clustering model.', [
            panel([section_title('Metric Space Scatter', 'Includes y=x reference line and rich tooltips.'), dashboard_graph(figure=clustering_scatter(submissions))]),
        ], open_by_default=True),
        experimental_section('Similarity Network', 'Top-ranked 20 submissions; each node links to its 2 nearest neighbors.', [
            panel([section_title('Similarity Network', 'Collapsed by default to avoid visual noise.'), network_visual(sentence_scores, submissions)]),
        ], open_by_default=False),
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
        gold_boundaries: set[int] = set()
        pred_boundaries: set[int] = set()
        if not boundary_rows.empty:
            for _, row in boundary_rows.iterrows():
                try:
                    position = int(row.get('boundary_position'))
                except Exception:
                    continue
                if int(row.get('gold_boundary', 0) or 0) == 1:
                    gold_boundaries.add(position)
                if int(row.get('pred_boundary', 0) or 0) == 1:
                    pred_boundaries.add(position)
        gold_tokens = tokens_from_boundaries(raw, gold_boundaries) if raw else []
        pred_tokens = tokens_from_boundaries(raw, pred_boundaries) if raw else []
        return panel([
            html.Div(f'句子 #{sentence_id}: {text_preview(raw, 140)}', className='boundary-sentence-title'),
            html.Div(metadata, className='boundary-sentence-meta'),
            character_boundary_diff(raw, boundary_rows),
            token_row(gold_tokens, 'Gold segmentation', 'gold-token', pred_tokens),
            token_row(pred_tokens, 'Pred segmentation', 'pred-token', gold_tokens),
            html.Div('Selected sentence score', className='subsection-title'),
            datatable('selected-sentence-score', score_rows, page_size=4),
            html.Details([html.Summary('Show raw boundary rows', className='details-summary'), datatable('selected-boundary-diff', boundary_rows, page_size=10)], className='details-panel'),
        ])

    @app.callback(Output('profile-subset-bar', 'figure'), Output('profile-error-distribution', 'figure'), Output('profile-summary', 'children'), Output('profile-sentence-table', 'data'), Output('profile-sentence-table', 'columns'), Input('profile-submission', 'value'))
    def update_profile(submission_name: str):
        if submissions.empty or not submission_name:
            return empty_figure('暂无提交'), empty_figure('暂无句级数据'), '', [], []
        row = submissions[submissions['submission_name'].astype(str) == str(submission_name)]
        if row.empty:
            return empty_figure('未找到提交'), empty_figure('未找到句级数据'), '', [], []
        series = row.iloc[0]
        subset_bar = profile_subset_bar(series)
        distribution = error_distribution_figure(str(submission_name), span_errors)
        weakest = lowest_sentence_table(str(submission_name), sentence_scores, sentence_table, span_errors)
        summary = dbc.Alert(
            f"Status: {series.get('status', '')} ｜ Word F1: {safe_float(series.get('word_f1')):.4f} ｜ "
            f"Boundary F1: {safe_float(series.get('boundary_f1')):.4f} ｜ Over-seg: {safe_int(series.get('over_segmentation_count'))} ｜ "
            f"Under-seg: {safe_int(series.get('under_segmentation_count'))} ｜ Warnings: {safe_int(series.get('tolerant_issue_count'))}",
            color='primary',
            className='academic-alert',
        )
        columns = [{'name': display_label(column), 'id': column} for column in weakest.columns]
        return subset_bar, distribution, summary, table_records(format_frame(weakest)), columns

    @app.callback(
        Output('gold-review-table', 'data'),
        Output('gold-review-table', 'columns'),
        Output('gold-review-table', 'selected_rows'),
        Input('gold-status-filter', 'value'),
        Input('gold-source-filter', 'value'),
        Input('gold-difficulty-filter', 'value'),
        Input('gold-review-flags-filter', 'value'),
        Input('gold-avg-f1-range', 'value'),
        Input('gold-quick-filter', 'data'),
    )
    def update_gold_review_table(gold_status, sources, difficulties, review_flags, avg_range, quick_filter):
        frame = apply_gold_filters(sentence_table, gold_status, sources, difficulties, review_flags, avg_range, quick_filter)
        columns = [{'name': display_label(column), 'id': column} for column in frame.columns]
        return table_records(format_frame(frame)), columns, ([0] if not frame.empty else [])

    @app.callback(
        Output('gold-quick-filter', 'data'),
        Input('gold-card-low_avg', 'n_clicks'),
        Input('gold-card-high_discrimination', 'n_clicks'),
        Input('gold-card-suspicious', 'n_clicks'),
        Input('gold-card-excluded', 'n_clicks'),
        Input('gold-card-confirmed', 'n_clicks'),
        State('gold-quick-filter', 'data'),
        prevent_initial_call=True,
    )
    def set_gold_quick_filter(low_avg, high_discrimination, suspicious, excluded, confirmed, current):
        del low_avg, high_discrimination, suspicious, excluded, confirmed
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate
        key = triggered[0]['prop_id'].split('.')[0].replace('gold-card-', '')
        current_type = (current or {}).get('type')
        return {} if current_type == key else {'type': key}

    @app.callback(Output('gold-detail-card', 'children'), Input('gold-review-table', 'data'), Input('gold-review-table', 'selected_rows'))
    def update_gold_detail(rows, selected_rows):
        if not rows or not selected_rows:
            return gold_detail_card(None, sentence_table, span_errors, sentence_scores)
        index = selected_rows[0]
        if index >= len(rows):
            return gold_detail_card(None, sentence_table, span_errors, sentence_scores)
        try:
            sentence_id = int(rows[index].get('sentence_id'))
        except Exception:
            sentence_id = None
        return gold_detail_card(sentence_id, sentence_table, span_errors, sentence_scores)

    return app

def main() -> None:
    args = parse_args()
    app = create_app(Path(args.results_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
