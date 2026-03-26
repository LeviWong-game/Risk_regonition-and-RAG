# RAG_data

该目录用于构建、整合、分类风险项目的 RAG 知识库资产。

## 主要文件

- `build_rag_kb.py`: 重建并分类 RAG 数据资产
- `rag_answer.py`: 本地检索问答脚本（BM25，可选 embedding 重排）
- `kb_documents.jsonl`: 合并后的总知识库
- `kb_risk_cases.jsonl`: 风险评估案例库
- `kb_emotion_cases.jsonl`: 情绪分析案例库
- `kb_disposition_cases.jsonl`: 处置建议案例库
- `kb_case_summaries.jsonl`: 案件类型摘要库
- `kb_reference_docs.jsonl`: 方法、规范、说明文档库
- `kb_stats.json`: 构建统计
- `kb_manifest.json`: 构建清单和参数

## 构建知识库

默认只使用训练集，避免评估泄漏：

```powershell
python RAG_data/build_rag_kb.py
```

如需把验证集也并入知识库（不建议用于评估）：

```powershell
python RAG_data/build_rag_kb.py --include_val
```

可指定输出目录和版本号：

```powershell
python RAG_data/build_rag_kb.py `
  --output_dir RAG_data `
  --output RAG_data/kb_documents.jsonl `
  --version v20260325.1
```

## 本地检索问答

仅 BM25 检索：

```powershell
python RAG_data/rag_answer.py `
  --question "案件类型：劳动争议。农民工工伤后急需赔偿，多次协商无果，风险如何判断？" `
  --model "qwen3.5:4b" `
  --pretty
```

启用 embedding 重排：

```powershell
python RAG_data/rag_answer.py `
  --question "案件类型：环境污染。居民持续投诉油烟扰民并围堵商户，风险如何判断？" `
  --model "qwen3.5:4b" `
  --embed_model "nomic-embed-text" `
  --pretty
```

## 推荐实践

- 优先用 `risk_data/train.jsonl` 构建 KB，保持评估独立
- 先用 RAG 找依据，再结合 `risk_scoring.py` 规则分给出结论
- 微调阶段优先用于固化输出格式，不替代知识更新
