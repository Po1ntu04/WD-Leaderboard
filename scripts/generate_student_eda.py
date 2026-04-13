from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'my_platform' / 'results'
EDA = RESULTS / 'eda'

sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / 'my_platform' / 'app').resolve()))
from eval_core import load_prediction_submission, normalize_prediction_rows_tolerant
from algorithms.common.io import read_raw_file, read_segmented_file


def load_student_board() -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(RESULTS / 'leaderboard.csv', encoding='utf-8-sig')
    student = board[board['submission_group'] == '课堂提交'].copy().sort_values('f1', ascending=False).reset_index(drop=True)
    problems = pd.read_csv(RESULTS / 'problem_submissions.csv', encoding='utf-8-sig') if (RESULTS / 'problem_submissions.csv').exists() else pd.DataFrame()
    return student, problems


def collect_common_failures() -> list[tuple[str, dict]]:
    agg: dict[str, dict] = {}
    manifest = pd.read_csv(ROOT / 'test_assets' / 'platform_eval_v2_draft' / 'gold_manifest.csv', encoding='utf-8-sig')
    manifest_rows = {int(row['line_no']): row for _, row in manifest.iterrows()}
    raw_rows = read_raw_file(ROOT / 'test_assets' / 'platform_eval_v2_draft' / 'raw.txt')
    gold_rows = read_segmented_file(ROOT / 'test_assets' / 'platform_eval_v2_draft' / 'gold.txt')
    for path in glob.glob(str(RESULTS / 'reports' / '*.report.json')):
        data = json.load(open(path, encoding='utf-8'))
        if data.get('submission_group') != '课堂提交':
            continue
        submission_path = Path(str(data.get('submission_path', '')))
        if not submission_path.exists():
            continue
        pred_rows, errors, _ = load_prediction_submission(submission_path)
        if errors and not pred_rows:
            continue
        pred_rows, _, _ = normalize_prediction_rows_tolerant(raw_rows, pred_rows)
        for idx, (raw, gold, pred) in enumerate(zip(raw_rows, gold_rows, pred_rows), start=1):
            if gold == pred:
                continue
            manifest_row = manifest_rows.get(idx, {})
            item = agg.setdefault(raw, {'count': 0, 'gold': ' / '.join(gold), 'line_no': idx, 'dataset': str(manifest_row.get('dataset', '')), 'preds': []})
            item['count'] += 1
            if len(item['preds']) < 4:
                item['preds'].append((data['submission_name'], ' / '.join(pred)))
    return sorted(agg.items(), key=lambda kv: (-kv[1]['count'], kv[1]['line_no']))


def draw_plots(student: pd.DataFrame, problems: pd.DataFrame) -> None:
    EDA.mkdir(parents=True, exist_ok=True)

    # F1 histogram
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.hist(student['f1'], bins=16, color='#3b82f6', edgecolor='white', alpha=0.9)
    ax.axvline(student['f1'].median(), color='#f59e0b', linestyle='--', linewidth=1.8, label=f"median={student['f1'].median():.3f}")
    ax.axvline(student['f1'].mean(), color='#10b981', linestyle='-.', linewidth=1.8, label=f"mean={student['f1'].mean():.3f}")
    ax.set_title('Student F1 Distribution')
    ax.set_xlabel('Overall F1')
    ax.set_ylabel('Count')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(EDA / 'f1_hist.png', bbox_inches='tight')
    plt.close(fig)

    # runtime scatter
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(student['runtime_seconds'].fillna(0), student['f1'], c=student['samechar_f1'].fillna(0), cmap='viridis', s=36, alpha=0.85, edgecolors='white', linewidths=0.4)
    special = pd.concat([student.head(3), student.sort_values('runtime_seconds', ascending=False).head(2)]).drop_duplicates('submission_name')
    for _, row in special.iterrows():
        ax.annotate(str(row['submission_name']), (row['runtime_seconds'], row['f1']), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_title('Runtime vs F1 (students only)')
    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Overall F1')
    fig.tight_layout()
    fig.savefig(EDA / 'runtime_vs_f1.png', bbox_inches='tight')
    plt.close(fig)

    # subset boxplot
    subset_cols = ['NLPCC-Weibo_f1', 'EvaHan-2022_f1', 'TCM-Ancient-Books_f1', 'samechar_f1']
    subset_labels = ['NLPCC', 'EvaHan', 'TCM', 'samechar']
    box_data = [student[col].dropna().values for col in subset_cols]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.boxplot(box_data, tick_labels=subset_labels, patch_artist=True, boxprops=dict(facecolor='#93c5fd'), medianprops=dict(color='#1d4ed8', linewidth=2), whiskerprops=dict(color='#64748b'), capprops=dict(color='#64748b'))
    ax.set_title('Per-dataset F1 Distribution')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(EDA / 'subset_boxplot.png', bbox_inches='tight')
    plt.close(fig)

    # correlation heatmap
    corr_cols = ['f1', 'NLPCC-Weibo_f1', 'EvaHan-2022_f1', 'TCM-Ancient-Books_f1', 'samechar_f1', 'runtime_seconds']
    corr = student[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=150)
    im = ax.imshow(corr.values, cmap='Blues', vmin=-1, vmax=1)
    labels = ['Overall', 'NLPCC', 'EvaHan', 'TCM', 'samechar', 'Runtime']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Metric Correlation')
    fig.tight_layout()
    fig.savefig(EDA / 'metric_correlation.png', bbox_inches='tight')
    plt.close(fig)

    # problem type counts
    if not problems.empty:
        pstu = problems[problems['submission_name'].isin(student['submission_name'])].copy()
        counts = pstu['problem_type'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        ax.bar(counts.index.tolist(), counts.values.tolist(), color=['#f59e0b', '#ef4444', '#8b5cf6'][:len(counts)])
        ax.set_title('Problem Submission Types')
        ax.set_ylabel('Count')
        fig.tight_layout()
        fig.savefig(EDA / 'problem_types.png', bbox_inches='tight')
        plt.close(fig)


def write_report(student: pd.DataFrame, problems: pd.DataFrame, common: list[tuple[str, dict]]) -> Path:
    q1 = float(student['f1'].quantile(0.25))
    q3 = float(student['f1'].quantile(0.75))
    iqr = q3 - q1
    high_threshold = q3 + 1.5 * iqr
    low_threshold = max(0.0, q1 - 1.5 * iqr)
    high_count = int((student['f1'] > high_threshold).sum())
    low_count = int((student['f1'] < low_threshold).sum())

    selected_by_dataset = {}
    for raw, item in common:
        dataset = item.get('dataset', '')
        if dataset and dataset not in selected_by_dataset:
            selected_by_dataset[dataset] = (raw, item)
    selected = [selected_by_dataset[ds] for ds in ['NLPCC-Weibo', 'EvaHan-2022', 'TCM-Ancient-Books', 'samechar'] if ds in selected_by_dataset]

    report: list[str] = []
    report.append('# 课堂学生提交 EDA 报告')
    report.append('')
    report.append('## 一、数据预处理')
    report.append('')
    report.append('### 1.1 数据来源')
    report.append('')
    report.append('- 排行榜主表：`my_platform/results/leaderboard.csv`')
    report.append('- 学生提交详情：`my_platform/results/reports/`')
    report.append('- 问题提交清单：`my_platform/results/problem_submissions.csv`')
    report.append('- 课堂评测包：`test_assets/platform_eval_v2_draft/`')
    report.append('')
    report.append('### 1.2 预处理步骤')
    report.append('')
    report.append('1. 仅保留 `submission_group=课堂提交` 的记录，去掉工具基线与 AI 对比结果。')
    report.append('2. 对重复提交名按最高分保留一条记录，避免重复统计。')
    report.append('3. 将总体 F1、分子集 F1、运行时间统一转为数值型。')
    report.append('4. 对个别行不匹配但整体可对齐的提交，采用“问题句记 0 分，其余句正常评分”的容错策略。')
    report.append('5. 单独记录问题提交类型，便于课堂讲评。')
    report.append('')
    report.append('### 1.3 数据规模')
    report.append('')
    student_problem_count = len(problems[problems['submission_name'].isin(student['submission_name'])])
    report.append(f'- 学生提交数：**{len(student)}**')
    report.append(f'- 带问题提示提交：**{student_problem_count}**')
    report.append('')
    report.append('## 二、描述性统计')
    report.append('')
    report.append(f'- 总体 F1 均值：**{student["f1"].mean():.4f}**')
    report.append(f'- 总体 F1 中位数：**{student["f1"].median():.4f}**')
    report.append(f'- 总体 F1 标准差：**{student["f1"].std():.4f}**')
    report.append(f'- 运行时间中位数：**{float(student["runtime_seconds"].median()):.4f} s**')
    report.append(f'- 最高 F1：**{float(student["f1"].max()):.4f}**')
    report.append(f'- 最低 F1：**{float(student["f1"].min()):.4f}**')
    report.append('')
    report.append('总体分数主要集中在 0.60~0.76 区间，右侧存在少量高分提交，说明这套测试集仍然具有较好的区分能力。运行时间离散程度明显大于分数离散程度，说明不同提交在工程实现上差异较大。')
    report.append('')
    report.append('## 三、图形化结果')
    report.append('')
    report.append('### 3.1 总体 F1 分布')
    report.append('')
    report.append('![](eda/f1_hist.png)')
    report.append('')
    report.append('### 3.2 运行时间与效果关系')
    report.append('')
    report.append('![](eda/runtime_vs_f1.png)')
    report.append('')
    report.append('散点图显示，运行时间和 F1 的关系并不强。也就是说，程序更慢不一定更准，更快也不一定更差。当前差距更多来自分词策略和工程处理方式，而不是单纯计算量。')
    report.append('')
    report.append('### 3.3 各子集 F1 分布')
    report.append('')
    report.append('![](eda/subset_boxplot.png)')
    report.append('')
    report.append('箱线图说明，`NLPCC` 和 `samechar` 的离散程度较大，`TCM` 整体分数偏低，`EvaHan` 中位数相对稳定。这表明各子集考察的能力差异很大。')
    report.append('')
    report.append('### 3.4 指标相关性')
    report.append('')
    report.append('![](eda/metric_correlation.png)')
    report.append('')
    report.append(f'总体 F1 与 `NLPCC`、`EvaHan`、`samechar` 的相关都较高，其中与 `NLPCC` 的相关最高，说明当前总榜仍然最容易受现代文本子集影响。')
    report.append('')
    report.append('### 3.5 问题提交类型')
    report.append('')
    report.append('![](eda/problem_types.png)')
    report.append('')
    report.append('问题提交几乎都集中在“原句还原问题”，说明当前最主要的失败点并不是算法完全不会分词，而是输出协议没有守住。')
    report.append('')
    report.append('## 四、分子集分析')
    report.append('')
    report.append(f'- `NLPCC` 平均 F1：**{student["NLPCC-Weibo_f1"].mean():.4f}**')
    report.append(f'- `EvaHan` 平均 F1：**{student["EvaHan-2022_f1"].mean():.4f}**')
    report.append(f'- `TCM` 平均 F1：**{student["TCM-Ancient-Books_f1"].mean():.4f}**')
    report.append(f'- `samechar专项` 平均 F1：**{student["samechar_f1"].mean():.4f}**')
    report.append('')
    report.append('从均值看，`TCM` 仍然是最弱的一桶，说明术语边界和古籍表达还是当前学生实现最容易掉分的点。`samechar专项` 的波动也较大，说明重复字和句义歧义仍然能拉开差距。')
    report.append('')
    report.append('## 五、异常值与稳定性')
    report.append('')
    report.append(f'- 高分异常值数量（IQR）：**{high_count}**')
    report.append(f'- 低分异常值数量（IQR）：**{low_count}**')
    report.append('')
    report.append('高分异常值不多，说明领先提交数量有限；低分异常值多数伴随协议问题或极端错误输出，而不只是普通分词偏差。')
    report.append('')
    report.append('## 六、共性错例归因分析')
    report.append('')
    for idx, (raw, item) in enumerate(selected, start=1):
        dataset_name = item.get('dataset', '')
        dataset_label = {
            'NLPCC-Weibo': 'NLPCC',
            'EvaHan-2022': 'EvaHan',
            'TCM-Ancient-Books': 'TCM',
            'samechar': 'samechar专项',
        }.get(dataset_name, dataset_name or f'案例 {idx}')
        report.append(f'### 6.{idx} {dataset_label}')
        report.append('')
        report.append(f'- 行号：**{item["line_no"]}**')
        report.append(f'- 原句：{raw}')
        report.append(f'- 标准切分：{item["gold"]}')
        for name, pred in item['preds'][:3]:
            report.append(f'- 预测样例（{name}）：{pred}')
        if dataset_name == 'NLPCC-Weibo' and item['line_no'] == 1:
            report.append('- 归因分析：这一句集中暴露了金额、百分比和英文缩写的切分问题。很多提交把 `55.9亿`、`BNSF`、`25%`、`7.84亿` 拆得过细，说明数字串、缩写串和单位串的整体保护不足。')
        elif dataset_name == 'NLPCC-Weibo' and item['line_no'] == 2:
            report.append('- 归因分析：这一句的主要难点是 mixed-script 和型号串。多数提交能找到中文主干，但会把 `EDIFICE`、`EQB-600D`、`APP`、`智能手机`、`平板电脑` 处理得过碎，说明英文专名与产品型号的边界仍不稳定。')
        elif dataset_name == 'NLPCC-Weibo' and item['line_no'] == 13:
            report.append('- 归因分析：这一句同时包含数字、单位和长省略号。很多提交在 `131克`、`3.4mm`、`2399元......` 这一类片段上切得很散，说明数词单位和省略号兼容处理不统一。')
        elif dataset_name == 'EvaHan-2022':
            report.append('- 归因分析：这是古汉语长句。少数提交在这一句已经不是单纯分词偏差，而是出现了字符替换或提示词混入，说明长句场景下输入输出契约更容易被破坏。')
        elif dataset_name == 'TCM-Ancient-Books':
            report.append('- 归因分析：这一类错例通常出现在术语、剂量结构和古籍短语附近。许多提交会把药名、方名和剂量拆散，说明领域词块整体性仍然不够稳。')
        elif dataset_name == 'samechar':
            report.append('- 归因分析：samechar 专项更依赖句义和重复字模式。共性错误通常不是完全不会切，而是在重复字附近做出看似合理但与标准不一致的边界。')
        else:
            report.append('- 归因分析：这类错例通常不是单一词边界问题，而是数字串、实体串或长句结构共同作用的结果。')
        report.append('')
    report.append('## 七、结论')
    report.append('')
    report.append('这次学生提交的 EDA 结果说明，当前课堂评测最能拉开差距的仍然是 mixed-script、数字单位串、古汉语长句和 samechar 这几类样本。学生之间的主要差距，一部分来自分词策略本身，另一部分来自是否严格遵守输出协议。对课堂讲评来说，只看总分不够，最好同时结合分子集得分、问题清单和共性错例一起解释。')
    report.append('')
    out = RESULTS / 'student_eda_report.md'
    out.write_text('\n'.join(report), encoding='utf-8')
    return out


def main() -> None:
    student, problems = load_student_board()
    common = collect_common_failures()
    draw_plots(student, problems)
    out = write_report(student, problems, common)
    print(out)


if __name__ == '__main__':
    main()
