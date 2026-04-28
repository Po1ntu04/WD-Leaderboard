from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform' / 'app'))

from algorithms.common.io import write_lines
from analytics_exports import (
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
    assert {row['boundary_case'] for row in boundary_rows} == {'TP', 'FN'}
    assert span_errors[0]['error_type'] == 'under_seg'
    assert span_errors[0]['raw_span'] == '我爱'
    assert {'left_char', 'right_char', 'left_context', 'right_context', 'gold_boundary', 'pred_boundary', 'boundary_case'} <= set(boundary_rows[0])

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
    from eval_core import score_prediction_submission

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
    assert any(row['gold_status'] == 'excluded' and row['is_evaluable'] == 0 for row in perfect_report['sentence_score_rows'])

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
    assert invalid_report['status'] == '成功'
    assert invalid_report['tolerant_issue_count'] == 1
    assert '无法还原原句' in invalid_report['message']
    bad_rows = [row for row in invalid_report['sentence_score_rows'] if row['validation_status'] == 'reconstruction_mismatch']
    assert len(bad_rows) == 1
    assert bad_rows[0]['pred_valid'] == 0
    assert bad_rows[0]['is_evaluable'] == 1
    assert bad_rows[0]['pred_word_count'] == 0
    assert bad_rows[0]['word_f1'] == 0
    assert bad_rows[0]['boundary_f1'] == 0
    assert invalid_report['overall']['total_sentences'] == 4
    assert invalid_report['overall']['gold_words'] == perfect_report['overall']['gold_words']
    assert invalid_report['overall']['word_f1'] < perfect_report['overall']['word_f1']

    sentence_table = (results_dir / 'sentence_table.csv').read_text(encoding='utf-8-sig')
    submission_table = (results_dir / 'submission_table.csv').read_text(encoding='utf-8-sig')
    assert 'sentence_avg_word_f1' in sentence_table
    assert 'discrimination_index' in sentence_table
    assert 'leaderboard.json' in {path.name for path in results_dir.iterdir()}
    assert 'perfect' in submission_table

    import pandas as pd

    sentence_score_table = pd.read_csv(results_dir / 'sentence_score_table.csv')
    boundary_table = pd.read_csv(results_dir / 'boundary_table.csv')
    span_error_table = pd.read_csv(results_dir / 'span_error_table.csv')
    assert not sentence_score_table.empty
    assert {
        'submission_name',
        'sentence_id',
        'validation_status',
        'pred_valid',
        'is_evaluable',
        'word_f1',
        'boundary_f1',
        'exact_match',
        'source',
        'difficulty',
        'sentence_type',
    } <= set(sentence_score_table.columns)
    assert not boundary_table.empty
    assert {
        'submission_name',
        'sentence_id',
        'boundary_position',
        'left_char',
        'right_char',
        'left_context',
        'right_context',
        'gold_boundary',
        'pred_boundary',
        'boundary_case',
        'boundary_type',
    } <= set(boundary_table.columns)
    assert not span_error_table.empty
    assert {
        'submission_name',
        'sentence_id',
        'raw_span',
        'gold_span_tokens',
        'pred_span_tokens',
        'start_char',
        'end_char',
        'error_type',
        'severity',
    } <= set(span_error_table.columns)
    assert (results_dir / 'sentence_score_table.csv.gz').exists()
    assert (results_dir / 'boundary_table.csv.gz').exists()
    assert (results_dir / 'span_error_table.csv.gz').exists()
    assert (results_dir / 'long_tables_manifest.json').exists()


def test_tolerant_policy_handles_missing_and_extra_lines(tmp_path: Path) -> None:
    from eval_core import score_prediction_submission

    root = Path('test_assets/dashboard_sample')
    results_dir = tmp_path / 'results'
    missing = tmp_path / 'missing_pred.txt'
    missing.write_text('我 / 爱 / 北京\nab / c\n重复 / 重复\n坏 / 样例\n', encoding='utf-8')
    missing_report, _ = score_prediction_submission(
        submission_path=missing,
        submission_name='missing',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert missing_report['status'] == '成功'
    assert missing_report['tolerant_issue_count'] == 1
    assert '缺失' in missing_report['message']
    missing_rows = [row for row in missing_report['sentence_score_rows'] if row['validation_status'] == 'missing_line']
    assert len(missing_rows) == 1
    assert missing_rows[0]['is_evaluable'] == 1
    assert missing_rows[0]['pred_word_count'] == 0
    assert missing_rows[0]['word_f1'] == 0
    assert missing_report['overall']['total_sentences'] == 4

    extra = tmp_path / 'extra_pred.txt'
    extra.write_text((root / 'submissions/perfect_pred.txt').read_text(encoding='utf-8') + '额外 / 行\n', encoding='utf-8')
    extra_report, _ = score_prediction_submission(
        submission_path=extra,
        submission_name='extra',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert extra_report['status'] == '成功'
    assert extra_report['tolerant_issue_count'] == 1
    assert '额外输出' in extra_report['message']
    assert all(row['validation_status'] == 'ok' or row['gold_status'] == 'excluded' for row in extra_report['sentence_score_rows'])


def test_fatal_file_and_encoding_errors_fail_submission(tmp_path: Path) -> None:
    from eval_core import score_prediction_submission

    root = Path('test_assets/dashboard_sample')
    results_dir = tmp_path / 'results'
    missing_report, _ = score_prediction_submission(
        submission_path=tmp_path / 'does_not_exist.txt',
        submission_name='missing_file',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert missing_report['status'] == '格式错误'
    assert '未找到输出文件' in missing_report['message']

    bad_encoding = tmp_path / 'bad_encoding.txt'
    bad_encoding.write_bytes(b'\xff\xfe\x00\x00')
    encoding_report, _ = score_prediction_submission(
        submission_path=bad_encoding,
        submission_name='bad_encoding',
        mode='prediction_file_only',
        raw_path=root / 'raw.txt',
        gold_path=root / 'gold.txt',
        manifest_path=root / 'manifest.csv',
        leaderboard_path=results_dir / 'leaderboard.csv',
        reports_dir=results_dir / 'reports',
    )
    assert encoding_report['status'] == '格式错误'
    assert 'UTF-8' in encoding_report['message']


def test_repeated_single_character_tokens_are_scored_by_span() -> None:
    score_row, _, _ = score_sentence(
        submission_name='repeat',
        sentence_id=1,
        raw_text='哈哈',
        gold_tokens=['哈', '哈'],
        pred_tokens=['哈', '哈'],
        subsets={'source': 'toy', 'difficulty': 'medium', 'sentence_type': 'repeated'},
        validation_status='ok',
    )
    assert score_row['correct_word_count'] == 2
    assert score_row['word_f1'] == 1.0
