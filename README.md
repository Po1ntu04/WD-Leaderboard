# WD-Leaderboard

面向课堂中文分词评测与投屏展示的最小交接仓库。

## 包含内容

- 课堂评测与单文件/批量评分脚本
- Dash 交互式排行榜前端
- 当前课堂评测包 `test_assets/platform_eval_v2_draft`
- GUI/提交说明配套样例 `test_assets/submission_sample_v1`
- 当前一份可直接展示的 demo 结果 `platform/results`

## 环境依赖

推荐 Python 3.10+。安装依赖：

```bash
pip install -r requirements.txt
```

依赖说明：

- `pandas`：读取 leaderboard / manifest / report
- `dash`：交互式前端
- `plotly`：图表绘制
- `dash-bootstrap-components`：前端样式

## 启动方式

### 1. 打开交互式排行榜前端

```bash
python app/leaderboard.py
```

默认地址：

- `http://127.0.0.1:8050/`

Windows 下也可以直接双击：

- `run_dashboard.bat`

### 1b. 导出 PPT 静态图片

```bash
python app/export_figures.py --results-dir platform/results --out-dir platform/results/figures
```

如需中文词云正常显示，请提供本机中文字体路径：

```bash
python app/export_figures.py \
  --results-dir platform/results \
  --out-dir platform/results/figures \
  --font-path /path/to/ChineseFont.ttf
```

### 2. 评测单个 `pred.txt`

```bash
python app/score.py --submission path/to/pred.txt --name 学号_姓名
```

### 3. 批量评测一个目录下的提交

```bash
python app/session.py --prediction-dir path/to/predictions
```

## `pred.txt` 约定

- 每一行对应 `raw.txt` 中的一行
- 规范格式：`词 / 词 / 词`
- 标点必须单独切分
- 若记录运行时间，请写在最后一行：

```text
# runtime_seconds: 0.183
```

## 目录说明

- `platform/app/`：评分、前端、批量会话脚本
- `platform/app/export_figures.py`：PPT-ready 静态图导出脚本
- `platform/results/`：当前 demo 结果
- `app/`：简化入口
- `algorithms/common/`：评分依赖的通用读写与 scorer
- `test_assets/platform_eval_v2_draft/`：课堂评测包
- `test_assets/submission_sample_v1/`：测试样例
- `docs/classroom_pred_submission_protocol.md`：提交协议

## 说明

- 当前仓库重点是 leaderboard 展示与课堂评测，不含完整算法训练工程。
- `platform/results/` 中已带一套 demo 结果，便于同事直接接手前端继续优化。
- 若要继续收集 `pred.txt` 并投屏展示，只需更新 `platform/results/leaderboard.csv`、`platform/results/reports/`，或直接重新跑 `python app/session.py`。
