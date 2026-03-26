from __future__ import annotations

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "RAG_data" / "kb_documents.jsonl"
OLLAMA_BASE = "http://127.0.0.1:11434"


def tokenize(text: str) -> List[str]:
    text = text.lower()
    chunks = re.findall(r"[\u4e00-\u9fff]{1,}|[a-z0-9_]+", text)
    tokens: List[str] = []
    for chunk in chunks:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(chunk[i : i + 2] for i in range(max(1, len(chunk) - 1)))
            tokens.extend(list(chunk))
        else:
            tokens.append(chunk)
    return tokens


def load_kb(path: Path) -> List[Dict[str, object]]:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def build_bm25(docs: Sequence[Dict[str, object]]) -> Tuple[List[Counter], Dict[str, float], float]:
    doc_tokens: List[Counter] = []
    doc_freq = defaultdict(int)
    lengths: List[int] = []

    for doc in docs:
        tokens = tokenize(str(doc.get("content", "")))
        counts = Counter(tokens)
        doc_tokens.append(counts)
        lengths.append(sum(counts.values()))
        for token in counts:
            doc_freq[token] += 1

    avg_len = sum(lengths) / len(lengths) if lengths else 1.0
    idf = {}
    total_docs = len(docs)
    for token, freq in doc_freq.items():
        idf[token] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
    return doc_tokens, idf, avg_len


def bm25_search(
    query: str,
    docs: Sequence[Dict[str, object]],
    doc_tokens: Sequence[Counter],
    idf: Dict[str, float],
    avg_len: float,
    top_k: int,
) -> List[Dict[str, object]]:
    query_tokens = tokenize(query)
    k1 = 1.5
    b = 0.75
    scored = []
    for doc, counts in zip(docs, doc_tokens):
        score = 0.0
        doc_len = sum(counts.values()) or 1
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            token_idf = idf.get(token, 0.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += token_idf * numerator / denominator
        if score > 0:
            payload = dict(doc)
            payload["_score"] = round(score, 6)
            scored.append(payload)
    scored.sort(key=lambda item: item["_score"], reverse=True)
    return scored[:top_k]


def ollama_embed(model: str, text: str) -> List[float]:
    body = json.dumps({"model": model, "prompt": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    embedding = payload.get("embedding")
    if not embedding:
        raise RuntimeError("Ollama embeddings response missing embedding.")
    return embedding


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dense_rerank(query: str, docs: Sequence[Dict[str, object]], embed_model: str, top_k: int) -> List[Dict[str, object]]:
    query_embedding = ollama_embed(embed_model, query)
    reranked = []
    for doc in docs:
        embedding = ollama_embed(embed_model, str(doc.get("content", "")))
        payload = dict(doc)
        payload["_dense_score"] = round(cosine_similarity(query_embedding, embedding), 6)
        reranked.append(payload)
    reranked.sort(key=lambda item: item["_dense_score"], reverse=True)
    return reranked[:top_k]


def build_prompt(question: str, retrieved_docs: Sequence[Dict[str, object]]) -> str:
    context_blocks = []
    for index, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"[文档{index}] 类型={doc.get('doc_type')} 案件类型={doc.get('case_type') or '无'}\n{doc.get('content')}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "你是一个基层风险研判助手。请严格基于检索到的知识库内容回答。\n"
        "如果知识库证据不足，要明确说明“知识库证据不足”。\n"
        "回答格式：\n"
        "1. 结论\n"
        "2. 依据\n"
        "3. 建议\n\n"
        f"知识库上下文：\n{context}\n\n"
        f"用户问题：\n{question}"
    )


def ollama_generate(model: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("response", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RAG query for Ollama qwen3.5-4b.")
    parser.add_argument("--question", required=True, help="Question or case text for retrieval.")
    parser.add_argument("--kb_path", default=str(KB_PATH))
    parser.add_argument("--model", default="qwen3.5:4b", help="Local Ollama generation model.")
    parser.add_argument(
        "--embed_model",
        default="",
        help="Optional Ollama embedding model such as nomic-embed-text or bge-m3. Empty means lexical retrieval only.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    docs = load_kb(Path(args.kb_path))
    doc_tokens, idf, avg_len = build_bm25(docs)
    lexical_docs = bm25_search(args.question, docs, doc_tokens, idf, avg_len, top_k=max(args.top_k * 3, args.top_k))
    if not lexical_docs:
        raise RuntimeError("No relevant knowledge documents were retrieved.")

    retrieval_mode = "bm25"
    final_docs = lexical_docs[: args.top_k]
    if args.embed_model:
        try:
            final_docs = dense_rerank(args.question, lexical_docs, args.embed_model, args.top_k)
            retrieval_mode = f"bm25+dense({args.embed_model})"
        except (urllib.error.URLError, RuntimeError) as exc:
            retrieval_mode = f"bm25 (dense fallback failed: {exc})"

    prompt = build_prompt(args.question, final_docs)
    answer = ollama_generate(args.model, prompt)
    payload = {
        "retrieval_mode": retrieval_mode,
        "model": args.model,
        "question": args.question,
        "retrieved_docs": [
            {
                "id": doc.get("id"),
                "doc_type": doc.get("doc_type"),
                "case_type": doc.get("case_type"),
                "risk_level": doc.get("risk_level"),
                "risk_score": doc.get("risk_score"),
                "score": doc.get("_dense_score", doc.get("_score")),
            }
            for doc in final_docs
        ],
        "answer": answer,
    }
    indent = 2 if args.pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
