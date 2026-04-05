# 风险预测数据与脚本

## 当前推荐使用的文件

- `risk_train_clean.jsonl`
- `risk_val_clean.jsonl`
- `risk_cases_master.jsonl`
- `prepare_risk_dataset.py`
- `risk_scoring.py`

## 数据说明

当前目录中的训练与验证数据已经统一为“风险预测”单任务数据，不再混入情感分析样本。

推荐使用：

- `risk_train_clean.jsonl` 作为训练集
- `risk_val_clean.jsonl` 作为验证集

这两个文件同时包含两套字段：

1. 结构化字段

- `sample_id`
- `task`
- `case_type`
- `title`
- `description`
- `case_text`
- `risk_level`
- `risk_score_min`
- `risk_score_max`
- `recommendation`
- `label_confidence`
- `label_votes`
- `label_conflict`
- `resolution_strategy`

2. 微调兼容字段

- `instruction`
- `input`
- `output`
- `messages`

因此现有基于 `instruction/input/output` 的训练脚本可以直接把训练入口切换到：

```bash
--train-data risk_data/risk_train_clean.jsonl
--val-data risk_data/risk_val_clean.jsonl
```

## 微调示例

```bash
python llama.cpp/finetune/finetune.py \
  --model qwen3.5:4b \
  --train-data risk_data/risk_train_clean.jsonl \
  --val-data risk_data/risk_val_clean.jsonl \
  --output ./qwen3.5-risktuned
```

## 数据生成

如果目录里仍保留原始混合数据，可以运行：

```bash
python risk_data/prepare_risk_dataset.py
```

该脚本会输出：

- `risk_cases_master.jsonl`
- `risk_train_clean.jsonl`
- `risk_val_clean.jsonl`
- `risk_dataset_report.json`

如果原始 `train.jsonl` / `val.jsonl` 已经删除，但 `risk_cases_master.jsonl` 还在，脚本仍然可以基于主数据重新切分训练集和验证集。

## 风险评分

命令行调用示例：

```bash
python -m risk_data.risk_scoring --text "案件类型：劳动争议\n案情：农民工工伤后急需赔偿，多次协商无果。" --pretty
```
