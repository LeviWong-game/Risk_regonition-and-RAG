# 风险评分脚本说明与运行指南

本目录的核心脚本已经按“入口 / 引擎 / 匹配 / 配置”拆分，职责如下。

## 1. 脚本功能

### `risk_scoring.py`
- **主入口脚本**（建议统一从这里启动）。
- 负责把调用转发到 `risk_engine.py`。
- 兼容历史导入方式（例如测试里 `from risk_scoring import ...`）。

### `risk_engine.py`
- **评分引擎主流程**。
- 包含：
  - `assess_case()`：执行完整风险评分
  - 二级指标汇总、P/I/D 计算、总分计算
  - 触发规则抬升等级
  - CLI（命令行参数 + 手动输入）

### `risk_matching.py`
- **字段提取与关键词匹配模块**。
- 包含：
  - 案件类型/案情等字段提取
  - 关键词规则、触发规则、案件类型提示规则
  - 别名匹配、归一化匹配、模糊匹配

### `risk_config.py`
- **评分配置常量模块**。
- 包含：
  - 指标权重、等级阈值
  - 指标标签与维度映射
  - 锚点文本（解释文案）

## 2. 脚本调用关系

`risk_scoring.py` -> `risk_engine.py` -> (`risk_matching.py` + `risk_config.py`)

## 3. 如何运行

建议在项目根目录 `e:\风险预测和评估` 运行。

### 方式 A：手动输入（推荐）
```powershell
python -m risk_data.risk_scoring --pretty
```
然后输入多行案件文本，最后输入一个空行结束。

### 方式 B：命令行直接传文本
```powershell
python -m risk_data.risk_scoring --text "案件类型：劳动争议`n案情：农民工工伤后急需赔偿，多次协商无果，并扬言聚集维权。" --pretty
```

### 方式 C：从文件读取
```powershell
python -m risk_data.risk_scoring --input-file "path\\to\\case.txt" --pretty
```

## 4. 调试命令

```powershell
python tests/debug_matching.py
python tests/debug_full.py
```

## 5. 结果说明

输出 JSON 关键字段：
- `risk_score`：0-100
- `risk_level`：L1-L4
- `P/I/D`：三个一级维度分
- `trigger_flags`：触发的红旗规则
- `reason_text`：解释文本

## 6. 是否需要本地大模型

不需要。  
当前这套脚本是**规则引擎**，不依赖本地 LLM（如 Ollama/Transformers）。
