# 风险预测标准数据格式

## 目标

将原始 `train.jsonl` / `val.jsonl` 中混合的多任务样本，整理为只面向风险预测的标准化 JSONL 数据。

## 输出文件

- `risk_cases_master.jsonl`
- `risk_train_clean.jsonl`
- `risk_val_clean.jsonl`
- `risk_dataset_report.json`

## 单条样本字段

```json
{
  "sample_id": "risk-0001",
  "task": "risk_prediction",
  "split": "train",
  "case_type": "环境污染",
  "title": "",
  "description": "餐饮店水电费扰民，住户苦不堪言",
  "case_text": "案件类型：环境污染\n案情：餐饮店水电费扰民，住户苦不堪言",
  "risk_level": "L3",
  "risk_score_min": 53,
  "risk_score_max": 80,
  "recommendation": "启动街道或相关部门联合处置，提前做好稳控与预案。",
  "label_confidence": 0.6667,
  "label_votes": {
    "L2": 1,
    "L3": 2
  },
  "label_conflict": true,
  "resolution_strategy": "majority_vote",
  "source_count": 3,
  "source_refs": [
    "train.jsonl:2",
    "train.jsonl:44",
    "val.jsonl:19"
  ],
  "instruction": "请根据案件类型和案件描述，预测风险等级并给出处置建议。",
  "input": "案件类型：环境污染\n案情：餐饮店水电费扰民，住户苦不堪言",
  "output": "{\"risk_level\": \"L3\", \"risk_score_range\": [53, 80], \"recommendation\": \"启动街道或相关部门联合处置，提前做好稳控与预案。\"}",
  "messages": [
    {
      "role": "system",
      "content": "你是基层矛盾纠纷风险预测助手，需要基于案件信息输出风险等级和处置建议。"
    },
    {
      "role": "user",
      "content": "请根据案件类型和案件描述，预测风险等级并给出处置建议。\n\n案件类型：环境污染\n案情：餐饮店水电费扰民，住户苦不堪言"
    },
    {
      "role": "assistant",
      "content": "{\"risk_level\": \"L3\", \"risk_score_range\": [53, 80], \"recommendation\": \"启动街道或相关部门联合处置，提前做好稳控与预案。\"}"
    }
  ],
  "sft_instruction": "请根据案件类型和案件描述，预测风险等级并给出处置建议。",
  "sft_input": "案件类型：环境污染\n案情：餐饮店水电费扰民，住户苦不堪言",
  "sft_output": "{\"risk_level\": \"L3\", \"risk_score_range\": [53, 80], \"recommendation\": \"启动街道或相关部门联合处置，提前做好稳控与预案。\"}"
}
```

## 说明

- `case_type`、`title`、`description`：从原始文本中拆出的结构化字段。
- `case_text`：统一重建后的标准输入文本。
- `risk_level`：冲突标签合并后的最终标签。
- `label_votes`：同一案件在原始数据中的标签投票分布。
- `label_conflict`：是否存在标签冲突。
- `instruction` / `input` / `output`：兼容旧版监督微调脚本。
- `messages`：兼容聊天式微调格式。
- `resolution_strategy`：冲突解决方式，可能为：
  - `single_source`
  - `unanimous`
  - `majority_vote`
  - `heuristic_tiebreak`
  - `conservative_tiebreak`
- `sft_*`：为后续监督微调保留的标准输入输出。
