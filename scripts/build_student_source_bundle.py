from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.common.io import parse_segmented_line, write_segmented_file


BUNDLE_ROOT = ROOT / 'student_resources' / 'source_datasets'


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_raw(path: Path, rows: list[str]) -> None:
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def segmented_file_to_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    return [parse_segmented_line(line) for line in lines if line.strip()]


def build_nlpcc() -> dict:
    src = ROOT / 'NLPCC-WordSeg-Weibo' / 'datasets'
    out = ensure_dir(BUNDLE_ROOT / 'NLPCC-Weibo')
    mapping = {
        'train': src / 'nlpcc2016-word-seg-train.dat',
        'dev': src / 'nlpcc2016-wordseg-dev.dat',
        'test': src / 'nlpcc2016-wordseg-test.dat',
    }
    summary = {}
    for split, file_path in mapping.items():
        rows = segmented_file_to_rows(file_path)
        split_dir = ensure_dir(out / split)
        write_raw(split_dir / 'raw.txt', [''.join(row) for row in rows])
        write_segmented_file(split_dir / 'gold.txt', rows)
        summary[split] = len(rows)
    (out / 'README.md').write_text(
        '# NLPCC-Weibo\n\n'
        '来源：`NLPCC-WordSeg-Weibo/datasets/`\n\n'
        '- `train/`、`dev/`、`test/` 均整理为 `raw.txt` + `gold.txt`\n'
        '- 适合同学自行训练、验证与测试\n',
        encoding='utf-8',
    )
    return {'dataset': 'NLPCC-Weibo', 'splits': summary}


def build_evahan() -> dict:
    src = ROOT / 'LT4HALA' / '2022' / 'data_and_doc'
    out = ensure_dir(BUNDLE_ROOT / 'EvaHan-2022')
    train_zip = src / 'zuozhuan_train_utf8.zip'
    with zipfile.ZipFile(train_zip) as zf:
        train_text = zf.read('zuozhuan_train_utf8.txt').decode('utf-8-sig')
    train_rows = [parse_segmented_line(line) for line in train_text.splitlines() if line.strip()]
    train_dir = ensure_dir(out / 'train')
    write_raw(train_dir / 'raw.txt', [''.join(row) for row in train_rows])
    write_segmented_file(train_dir / 'gold.txt', train_rows)

    testa_raw = src / 'EvaHan_testa_raw.txt'
    testa_gold = src / 'EvaHan_testa_gold.txt'
    testb_raw = src / 'EvaHan_testb_raw.txt'
    testb_gold = src / 'EvaHan_testb_gold.txt'
    split_map = {
        'test_a': (testa_raw, testa_gold),
        'test_b': (testb_raw, testb_gold),
    }
    summary = {'train': len(train_rows)}
    for split, (raw_path, gold_path) in split_map.items():
        split_dir = ensure_dir(out / split)
        raw_rows = [line.strip() for line in raw_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        gold_rows = segmented_file_to_rows(gold_path)
        write_raw(split_dir / 'raw.txt', raw_rows)
        write_segmented_file(split_dir / 'gold.txt', gold_rows)
        summary[split] = len(raw_rows)
    (out / 'README.md').write_text(
        '# EvaHan-2022\n\n'
        '来源：`LT4HALA/2022/data_and_doc/`\n\n'
        '- 训练集来自 `zuozhuan_train_utf8.zip`\n'
        '- `test_a/` 与 `test_b/` 整理为 `raw.txt` + `gold.txt`\n'
        '- 适合同学自行训练、验证古汉语分词能力\n',
        encoding='utf-8',
    )
    return {'dataset': 'EvaHan-2022', 'splits': summary}


def build_tcm() -> dict:
    out = ensure_dir(BUNDLE_ROOT / 'TCM-Ancient-Books')
    raw_rows = [line.strip() for line in (ROOT / 'TCM' / 'corpus_sentences.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    gold_rows = segmented_file_to_rows(ROOT / 'TCM' / 'corpus_segmented.txt')
    full_dir = ensure_dir(out / 'full')
    write_raw(full_dir / 'raw.txt', raw_rows)
    write_segmented_file(full_dir / 'gold.txt', gold_rows)
    (out / 'README.md').write_text(
        '# TCM-Ancient-Books\n\n'
        '来源：`TCM/`\n\n'
        '- 当前提供整理后的完整语料：`full/raw.txt` + `full/gold.txt`\n'
        '- 原始资源本身没有官方 train/dev/test 划分，可由同学自行划分训练与测试\n',
        encoding='utf-8',
    )
    return {'dataset': 'TCM-Ancient-Books', 'splits': {'full': len(raw_rows)}}


def build_samechar() -> dict:
    out = ensure_dir(BUNDLE_ROOT / 'samechar')
    rows = []
    with (ROOT / 'samechar' / 'chinese_samechar_tongue_twister_testset.csv').open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = row.get('text', '').strip()
            gold = row.get('gold_segmentation', '').strip()
            if text and gold:
                rows.append((text, parse_segmented_line(gold)))
    full_dir = ensure_dir(out / 'full')
    write_raw(full_dir / 'raw.txt', [text for text, _ in rows])
    write_segmented_file(full_dir / 'gold.txt', [tokens for _, tokens in rows])
    (out / 'README.md').write_text(
        '# samechar\n\n'
        '来源：`samechar/chinese_samechar_tongue_twister_testset.csv`\n\n'
        '- 当前提供完整专项集：`full/raw.txt` + `full/gold.txt`\n'
        '- 适合做 challenge / stress test，不建议单独代替通用训练语料\n',
        encoding='utf-8',
    )
    return {'dataset': 'samechar', 'splits': {'full': len(rows)}}


def main() -> None:
    ensure_dir(BUNDLE_ROOT)
    summary = {
        'bundle_root': str(BUNDLE_ROOT.relative_to(ROOT)),
        'datasets': [
            build_nlpcc(),
            build_evahan(),
            build_tcm(),
            build_samechar(),
        ],
    }
    (BUNDLE_ROOT / 'README.md').write_text(
        '# source_datasets\n\n'
        '本目录整理了课堂评测对应的原始数据资源，统一为 `raw.txt` + `gold.txt` 形式，便于同学后续自行训练与测试。\n\n'
        '包含：\n'
        '- `NLPCC-Weibo/`\n'
        '- `EvaHan-2022/`\n'
        '- `TCM-Ancient-Books/`\n'
        '- `samechar/`\n',
        encoding='utf-8',
    )
    (BUNDLE_ROOT / 'bundle_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
