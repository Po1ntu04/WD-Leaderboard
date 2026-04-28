from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard import (  # noqa: E402
    LEADERBOARD_VISIBLE_COLUMNS,
    METRIC_LABELS,
    PLOT_LAYOUT,
    display_label,
    error_counts,
    load_tables,
    metric_rank_comparison,
    sentence_scatter,
    source_summary_figure,
    subset_score_heatmap,
    text_preview,
)


ACADEMIC_BLUE = '#1e3a8a'
ACCENT_BLUE = '#2563eb'
MUTED = '#64748b'
EXPORT_WIDTH = 1600
EXPORT_HEIGHT = 900
FUNCTION_WORDS = set('的一是在和及与或了着过也又而并但被把于以为之其此那这各每及等同')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export PPT-ready static figures from leaderboard result tables.')
    parser.add_argument('--results-dir', default='platform/results')
    parser.add_argument('--out-dir', default='platform/results/figures')
    parser.add_argument('--font-path', default='', help='Optional Chinese TTF/OTF font path for word cloud export.')
    parser.add_argument('--width', type=int, default=EXPORT_WIDTH)
    parser.add_argument('--height', type=int, default=EXPORT_HEIGHT)
    return parser.parse_args()


def ensure_static_export_available() -> None:
    try:
        import kaleido  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit('Static Plotly PNG export requires kaleido. Install with: python -m pip install kaleido') from exc


def apply_export_layout(fig: go.Figure, title: str, subtitle: str = '', *, width: int, height: int) -> go.Figure:
    title_text = f'{title}<br><sup>{subtitle}</sup>' if subtitle else title
    fig.update_layout(
        title=dict(text=title_text, x=0.02, xanchor='left', font=dict(size=34, color=ACADEMIC_BLUE)),
        width=width,
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif', size=22, color='#1e293b'),
        margin=dict(l=90, r=70, t=120, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    return fig


def write_figure(fig: go.Figure, path: Path, title: str, subtitle: str = '', *, width: int, height: int) -> None:
    ensure_static_export_available()
    apply_export_layout(fig, title, subtitle, width=width, height=height)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height, scale=1)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors='coerce').fillna(0.0)


def leaderboard_top12(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty:
        return go.Figure()
    frame = submissions.copy()
    frame['word_f1'] = numeric(frame, 'word_f1')
    if 'rank' in frame.columns:
        frame['_rank'] = pd.to_numeric(frame['rank'], errors='coerce').fillna(999999)
        frame = frame.sort_values('_rank')
    else:
        frame = frame.sort_values('word_f1', ascending=False)
    top = frame.head(12).iloc[::-1].copy()
    top['Submission'] = top['submission_name'].astype(str).map(lambda value: text_preview(value, 26))
    top['Word F1'] = top['word_f1']
    top['value_label'] = top['Word F1'].map(lambda value: f'{value:.4f}')
    fig = px.bar(top, x='Word F1', y='Submission', orientation='h', text='value_label', color='Word F1', color_continuous_scale='Blues')
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_xaxes(range=[0, 1.08], title='Word F1')
    fig.update_yaxes(title='')
    fig.update_layout(coloraxis_showscale=False)
    return fig


def metric_space(submissions: pd.DataFrame) -> go.Figure:
    if submissions.empty:
        return go.Figure()
    frame = submissions.copy()
    frame['Word F1'] = numeric(frame, 'word_f1')
    frame['Boundary F1'] = numeric(frame, 'boundary_f1')
    frame['Exact Match'] = numeric(frame, 'exact_match_sentence_rate')
    frame['Warnings'] = numeric(frame, 'tolerant_issue_count')
    fig = px.scatter(frame, x='Word F1', y='Boundary F1', size='Exact Match', color='Warnings', hover_name='submission_name', color_continuous_scale='Blues')
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#94a3b8', dash='dash', width=2), name='y = x'))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def low_f1_cases(sentence_table: pd.DataFrame) -> go.Figure:
    if sentence_table.empty:
        return go.Figure()
    frame = sentence_table.copy()
    frame['Avg F1'] = numeric(frame, 'sentence_avg_word_f1')
    frame['Discrimination'] = numeric(frame, 'discrimination_index')
    frame['Sentence'] = frame.get('raw_text', pd.Series(dtype=str)).fillna('').astype(str).map(lambda value: text_preview(value, 34))
    frame = frame.sort_values(['Avg F1', 'Discrimination'], ascending=[True, False]).head(12).iloc[::-1]
    fig = px.bar(frame, x='Avg F1', y='Sentence', orientation='h', color='Discrimination', color_continuous_scale='Oranges', text=frame['Avg F1'].map(lambda value: f'{value:.3f}'))
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_xaxes(range=[0, 1.05])
    fig.update_layout(coloraxis_colorbar_title='Discrimination')
    return fig


def boundary_diff_case(boundary_table: pd.DataFrame, sentence_table: pd.DataFrame, submissions: pd.DataFrame) -> go.Figure:
    if boundary_table.empty or sentence_table.empty:
        return go.Figure()
    frame = boundary_table[boundary_table.get('boundary_case', '') != 'TP'].copy()
    if frame.empty:
        frame = boundary_table.copy()
    if 'rank' in submissions.columns and 'submission_name' in submissions.columns:
        top_names = submissions.assign(_rank=pd.to_numeric(submissions['rank'], errors='coerce')).sort_values('_rank')['submission_name'].astype(str).tolist()
        frame['_submission_order'] = frame['submission_name'].astype(str).map({name: i for i, name in enumerate(top_names)}).fillna(9999)
        frame = frame.sort_values(['_submission_order', 'sentence_id'])
    row = frame.iloc[0]
    submission_name = str(row.get('submission_name', ''))
    sentence_id = int(row.get('sentence_id'))
    raw_row = sentence_table[sentence_table['sentence_id'] == sentence_id]
    raw_text = str(raw_row.iloc[0].get('raw_text', '')) if not raw_row.empty else ''
    rows = boundary_table[(boundary_table['submission_name'].astype(str) == submission_name) & (boundary_table['sentence_id'] == sentence_id)]
    max_chars = min(len(raw_text), 90)
    x = list(range(max_chars))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[1] * max_chars, mode='text', text=list(raw_text[:max_chars]), textfont=dict(size=28, color='#0f172a'), name='Characters', hoverinfo='skip'))
    color_map = {'TP': '#16a34a', 'FP': '#f97316', 'FN': '#dc2626'}
    for case, group in rows.groupby('boundary_case'):
        positions = [int(pos) - 0.5 for pos in group['boundary_position'] if 0 < int(pos) <= max_chars]
        if not positions:
            continue
        fig.add_trace(go.Scatter(x=positions, y=[0.76] * len(positions), mode='markers', marker=dict(size=18, color=color_map.get(str(case), MUTED), symbol='line-ns-open', line=dict(width=4)), name=str(case)))
    fig.update_yaxes(visible=False, range=[0.55, 1.15])
    fig.update_xaxes(visible=False, range=[-1, max_chars + 1])
    fig.update_layout(showlegend=True)
    return fig


def clean_token(token: Any, *, remove_single_function_words: bool = True) -> str:
    token = str(token or '').strip()
    if not token:
        return ''
    token = re.sub(r'\s+', '', token)
    if not token or token.isdigit():
        return ''
    if all(unicodedata.category(ch).startswith('P') or unicodedata.category(ch).startswith('S') for ch in token):
        return ''
    if remove_single_function_words and len(token) == 1 and token in FUNCTION_WORDS:
        return ''
    return token


def token_counter_from_gold(sentence_table: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in sentence_table.get('gold', pd.Series(dtype=str)).dropna().astype(str):
        for token in text.replace(' / ', '/').split('/'):
            token = clean_token(token)
            if token:
                counter[token] += 1
    return counter


def token_counter_from_errors(span_errors: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for token in span_errors.get('raw_span', pd.Series(dtype=str)).dropna().astype(str):
        token = clean_token(token, remove_single_function_words=False)
        if token:
            counter[token] += 1
    return counter


def export_wordcloud(counter: Counter[str], path: Path, font_path: str = '', *, width: int, height: int) -> None:
    try:
        from wordcloud import WordCloud
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit('Word cloud PNG export requires wordcloud. Install with: python -m pip install wordcloud') from exc
    if not counter:
        counter = Counter({'No data': 1})
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='Blues',
        max_words=160,
        prefer_horizontal=0.92,
        font_path=font_path or None,
        random_state=42,
        relative_scaling=0.4,
        collocations=False,
        margin=12,
    ).generate_from_frequencies(counter)
    path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(path))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(results_dir)
    submissions = tables['submission_table']
    sentence_table = tables['sentence_table']
    sentence_scores = tables['sentence_score_table']
    boundary_table = tables['boundary_table']
    span_errors = tables['span_error_table']
    width, height = args.width, args.height

    write_figure(leaderboard_top12(submissions), out_dir / '01_leaderboard_top12.png', 'Leaderboard Top 12', 'Official Word F1 ranking preview', width=width, height=height)
    write_figure(metric_space(submissions), out_dir / '02_metric_space_scatter.png', 'Metric Space Scatter', 'Word F1 vs Boundary F1; point size is strict exact match', width=width, height=height)
    write_figure(subset_score_heatmap(submissions), out_dir / '03_subset_heatmap_top20.png', 'Subset Heatmap Top 20', 'Source / difficulty / sentence type Word F1', width=width, height=height)
    write_figure(error_counts(span_errors, boundary_table), out_dir / '04_error_type_distribution.png', 'Error Type Distribution', 'True positives hidden; span and boundary errors separated', width=width, height=height)
    write_figure(sentence_scatter(sentence_table), out_dir / '05_sentence_difficulty_map.png', 'Sentence Difficulty Map', 'Average sentence Word F1 and discrimination', width=width, height=height)
    write_figure(source_summary_figure(sentence_table), out_dir / '06_dataset_source_distribution.png', 'Dataset Source Distribution', 'Sentence count by source', width=width, height=height)
    write_figure(boundary_diff_case(boundary_table, sentence_table, submissions), out_dir / '07_boundary_diff_case.png', 'Boundary Diff Case', 'TP / FP / FN boundary positions for one representative sentence', width=width, height=height)
    write_figure(low_f1_cases(sentence_table), out_dir / '08_gold_review_low_f1_cases.png', 'Gold Review: Low-F1 Cases', 'Low average F1 first; color shows discrimination', width=width, height=height)

    export_wordcloud(token_counter_from_gold(sentence_table), out_dir / 'wordcloud_dataset_gold.png', args.font_path, width=width, height=height)
    export_wordcloud(token_counter_from_errors(span_errors), out_dir / 'wordcloud_common_error_spans.png', args.font_path, width=width, height=height)
    print(f'Exported PPT-ready figures to {out_dir}')


if __name__ == '__main__':
    main()
