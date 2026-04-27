from __future__ import annotations

from pathlib import Path

from algorithms.common.io import write_lines
from my_platform.app.analytics_exports import (
    aggregate_score_rows,
    build_sentence_table,
    evaluate_submission,
    read_manifest_frame,
    score_sentence,
    validate_prediction_rows,
)


def test_span_and_boundary_metrics_distinguish_under_and_over_segmentation() -> None:
    score_row, boundary_rows, span_errors = score_sentence(
        submission_name='method_a',
        sentence_id=1,
        raw_text='我爱北京',
        gold_tokens=['我', '爱', '北京'],
        pred_tokens=['我爱', '北京'],
        subsets={'source': 'toy', 'difficulty': 'medium', 'sentence_type': 'standard'},
        validation_status='ok',
    )

    assert score_row['word_precision'] == 0.5
    assert score_row['word_recall'] == 0.333333
    assert score_row['boundary_precision'] == 1.0
    assert score_row['boundary_recall'] == 0.5
    assert score_row['under_segmentation_count'] == 1
    assert score_row['over_segmentation_count'] == 0
    assert {row['boundary_type'] for row in boundary_rows} == {'true_positive', 'under_segmentation'}
    assert {row['error_type'] for row in span_errors} == {'false_positive_span', 'false_negative_span'}

    over_row, _, _ = score_sentence(
        submission_name='method_b',
        sentence_id=2,
        raw_text='abc',
        gold_tokens=['ab', 'c'],
        pred_tokens=['a', 'b', 'c'],
        subsets={'source': 'toy', 'difficulty': 'medium', 'sentence_type': 'standard'},
        validation_status='ok',
    )
    assert over_row['over_segmentation_count'] == 1
    assert over_row['under_segmentation_count'] == 0


def test_validation_requires_line_count_and_exact_reconstruction() -> None:
    errors, per_sentence = validate_prediction_rows(['abc', 'de'], [['a', 'bc']])
    assert errors[0] == '行数不匹配：raw=2 pred=1'
    assert per_sentence == {1: 'line_count_mismatch', 2: 'line_count_mismatch'}

    errors, per_sentence = validate_prediction_rows(['abc'], [['a', 'b']])
    assert errors == ['第 1 行分词结果无法还原原句。']
    assert per_sentence == {1: 'reconstruction_mismatch'}


def test_evaluate_submission_exports_flat_submission_row(tmp_path: Path) -> None:
    raw_path = tmp_path / 'raw.txt'
    gold_path = tmp_path / 'gold.txt'
    manifest_path = tmp_path / 'manifest.csv'
    write_lines(raw_path, ['我爱北京', 'abc'])
    write_lines(gold_path, ['我 / 爱 / 北京', 'ab / c'])
    manifest_path.write_text(
        'line_no,dataset,difficulty_bucket,selection_tags,raw_text\n'
        '1,toy,medium,sentence_level,我爱北京\n'
        '2,toy,high,mixed_script,abc\n',
        encoding='utf-8',
    )

    raw_rows = ['我爱北京', 'abc']
    gold_rows = [['我', '爱', '北京'], ['ab', 'c']]
    manifest = read_manifest_frame(manifest_path, raw_rows)
    sentence_table = build_sentence_table(raw_rows, gold_rows, manifest)
    payload = evaluate_submission(
        raw_rows=raw_rows,
        gold_rows=gold_rows,
        pred_rows=[['我', '爱', '北京'], ['a', 'b', 'c']],
        sentence_table=sentence_table,
        submission_name='demo',
        status='成功',
        validation_errors=[],
        submission_group='课堂提交',
        mode='prediction_file_only',
    )

    row = payload['submission_row']
    assert row['submission_name'] == 'demo'
    assert row['word_f1'] > 0
    assert row['boundary_f1'] > 0
    assert row['source:toy:word_f1'] == row['word_f1']
    assert 'difficulty:high:word_f1' in row
    assert len(payload['sentence_score_rows']) == 2
    assert aggregate_score_rows(payload['sentence_score_rows'])['total_sentences'] == 2


def test_dashboard_sample_covers_required_scoring_cases(tmp_path: Path) -> None:
    from my_platform.app.eval_core import score_prediction_submission

    root = Path('test_assets/dashboard_sample')
    results_dir = tmp_path / 'results'

    perfect_report, _ = score_prediction_submission(
        submission_path=root / 'submissions/perfect_pred.txt',
        submission_name='perfect',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert perfect_report['status'] == '成功'
    assert perfect_report['overall']['word_f1'] == 1.0
    assert perfect_report['overall']['boundary_f1'] == 1.0
    assert perfect_report['overall']['total_sentences'] == 4  # excluded gold is not ranked/scored
    assert any(row['gold_status'] == 'excluded' and row['is_scored'] == 0 for row in perfect_report['sentence_score_rows'])

    over_report, _ = score_prediction_submission(
        submission_path=root / 'submissions/over_pred.txt',
        submission_name='over',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert over_report['overall']['over_segmentation_count'] > 0

    under_report, _ = score_prediction_submission(
        submission_path=root / 'submissions/under_pred.txt',
        submission_name='under',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert under_report['overall']['under_segmentation_count'] > 0

    invalid_report, _ = score_prediction_submission(
        submission_path=root / 'submissions/invalid_pred.txt',
        submission_name='invalid',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert invalid_report['status'] == '格式错误'
    assert '无法还原原句' in invalid_report['message']

    sentence_table = (results_dir / 'sentence_table.csv').read_text(encoding='utf-8-sig')
    submission_table = (results_dir / 'submission_table.csv').read_text(encoding='utf-8-sig')
    assert 'sentence_avg_word_f1' in sentence_table
    assert 'discrimination_index' in sentence_table
    assert 'leaderboard.json' in {path.name for path in results_dir.iterdir()}
    assert 'perfect' in submission_table
