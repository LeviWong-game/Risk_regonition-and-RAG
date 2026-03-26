# 快速测试命令

## 1) 规则引擎测试（risk_scoring.py）

```powershell
python -m risk_data.risk_scoring --input-file "RAG_data/test_case_labor.txt" --pretty
```

预期：返回 `risk_score`、`risk_level`、`P/I/D`、`trigger_flags`、`reason_text` 等字段。

## 2) RAG 检索问答测试（rag_answer.py）

```powershell
python RAG_data/rag_answer.py `
  --question "案件类型：劳动争议。农民工工伤后急需赔偿，多次协商无果，该怎么判断风险等级？" `
  --model "qwen3.5:4b" `
  --top_k 3 `
  --pretty
```

## 3) 可选：启用 embedding 重排（如果本地有 embedding 模型）

```powershell
python RAG_data/rag_answer.py `
  --question "案件类型：劳动争议。农民工工伤后急需赔偿，多次协商无果，该怎么判断风险等级？" `
  --model "qwen3.5:4b" `
  --embed_model "qwen3-embedding:8b" `
  --top_k 3 `
  --pretty
```

## 4) 环境检查（可选）

```powershell
python --version
ollama --version
ollama list
```
