# 风险预测与情感评估训练数据

## 数据概览

| 指标 | 数值 |
|------|------|
| 总数据条数 | 700条 |
| 训练集 | 560条 |
| 验证集 | 140条 |
| 数据格式 | JSONL |
| 适用模型 | qwen3.5:4b |
| 任务类型 | 情感分析、风险预测 |

## 文件说明

```
risk_data/
├── train.jsonl             # 训练集 (560条)
├── val.jsonl               # 验证集 (140条)
├── ollama_train.jsonl      # Ollama 对话格式训练集
├── risk_scoring.py         # 可解释风险评分引擎
├── risk_scoring_method.md  # 评分方法说明
└── README.md               # 本文件
```

## 数据格式

每条数据包含三个字段：

```json
{
  "instruction": "任务指令",
  "input": "输入上下文/案件信息",
  "output": "期望输出/预测结果"
}
```

## Ollama + QLoRA 微调步骤

### 1. 安装依赖

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j4
pip install -r requirements.txt

# 或使用 Axolotl
pip install axolotl
```

### 2. 转换数据格式

数据已转换为 Ollama 对话格式（messages 数组），包含 system/user/assistant 三轮对话。

### 3. 开始微调

```bash
# 使用 llama.cpp + QLoRA 微调
python llama.cpp/finetune/finetune.py \
  --model qwen3.5:4b \
  --train-data risk_data/train.jsonl \
  --val-data risk_data/val.jsonl \
  --output ./qwen3.5-risktuned

# 转换为 GGUF 格式
python llama.cpp/convert.py ./qwen3.5-risktuned --outfile qwen3.5-risk.gguf

# 导入 Ollama
ollama create qwen3.5:4b-risk -f ./qwen3.5-risk.gguf
```

## 任务说明

### 情感评估
- **输入**: 案件描述、人物角色
- **输出**: 情感状态（愤怒/焦虑/担忧/无奈/满意/平静）+ 分析

### 风险预测
- **输入**: 案件类型、案情描述
- **输出**: 风险等级（L1-L4）+ 风险分值 + 处置建议

## 风险等级说明

| 等级 | 分值 | 处置方式 |
|------|------|----------|
| L1 | 0-25 | 网格员/物业自行处理 |
| L2 | 25-50 | 调解员介入调解 |
| L3 | 50-75 | 政府部门联合处置 |
| L4 | 75-100 | 法院/公安介入 |

## 新增评分工具

### 1. 运行风险评分

```bash
python -m risk_data.risk_scoring --text "案件类型：劳动争议
案情：农民工工伤后急需赔偿，多次协商无果，并扬言聚集维权。" --pretty
```

输出字段包括：

- `risk_score`
- `risk_level`
- `P / I / D`
- `trigger_flags`
- `reason_text`

### 2. 方法说明

完整的公式、指标、权重和文献依据见：

- `risk_data/risk_scoring_method.md`

---
*数据来源: 广州市白云区矛盾风险预警系统案例库*
