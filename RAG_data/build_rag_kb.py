from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
RISK_DATA_DIR = ROOT / "risk_data"
OUTPUT_DIR = ROOT / "RAG_data"
COMBINED_OUTPUT = OUTPUT_DIR / "kb_documents.jsonl"
STATS_OUTPUT = OUTPUT_DIR / "kb_stats.json"
MANIFEST_OUTPUT = OUTPUT_DIR / "kb_manifest.json"

RISK_LEVEL_RE = re.compile(r"风险等级[:：]\s*(L\d)")
RISK_SCORE_RE = re.compile(r"风险分值[:：]\s*(\d+)")
EMOTION_RE = re.compile(r"情感状态[:：]\s*([^\n，。,；;]+)")
CASE_TYPE_RE = re.compile(r"(?:案件类型|类型)[:：]\s*([^\n]+)")
CASE_TITLE_RE = re.compile(r"(?:案件标题|案件)[:：]\s*([^\n]+)")

TASK_RISK = "risk_assessment"
TASK_EMOTION = "emotion_analysis"
TASK_DISPOSITION = "disposition_advice"
TASK_GENERIC = "generic"

DOC_RISK = "risk_case"
DOC_EMOTION = "emotion_case"
DOC_DISPOSITION = "disposition_case"
DOC_GENERIC = "generic_case"
DOC_SUMMARY = "case_type_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild, integrate, and classify RAG knowledge assets.")
    parser.add_argument("--train_file", default=str(RISK_DATA_DIR / "train.jsonl"))
    parser.add_argument("--val_file", default=str(RISK_DATA_DIR / "val.jsonl"))
    parser.add_argument(
        "--include_val",
        action="store_true",
        help="Include validation data in KB. Default is false to avoid evaluation leakage.",
    )
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--output", default=str(COMBINED_OUTPUT))
    parser.add_argument("--version", default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, str]]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line:
            yield line_no, json.loads(line)


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def infer_task_type(instruction: str, output_text: str) -> str:
    if "风险等级" in output_text or "风险分值" in output_text:
        return TASK_RISK
    if "情感状态" in instruction or "情感状态" in output_text or "情感分析" in output_text:
        return TASK_EMOTION
    if "处置建议" in instruction or "建议处置方案" in output_text or "处置方案" in output_text:
        return TASK_DISPOSITION
    return TASK_GENERIC


def task_to_doc_type(task_type: str) -> str:
    if task_type == TASK_RISK:
        return DOC_RISK
    if task_type == TASK_EMOTION:
        return DOC_EMOTION
    if task_type == TASK_DISPOSITION:
        return DOC_DISPOSITION
    return DOC_GENERIC


def extract_case_type(text: str) -> str:
    match = CASE_TYPE_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_case_title(text: str) -> str:
    match = CASE_TITLE_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_risk_level(output_text: str) -> str:
    match = RISK_LEVEL_RE.search(output_text)
    return match.group(1) if match else ""


def extract_risk_score(output_text: str) -> int | None:
    match = RISK_SCORE_RE.search(output_text)
    return int(match.group(1)) if match else None


def extract_emotion(output_text: str) -> str:
    match = EMOTION_RE.search(output_text)
    return match.group(1).strip() if match else ""


def to_keywords(*items: str) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = normalize_space(item)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_case_content(doc: Dict[str, object]) -> str:
    lines = [
        f"文档类型：{doc['doc_type']}",
        f"任务类型：{doc['task_type']}",
        f"来源拆分：{doc['source_split']}",
        f"来源文件：{doc['source_file']}",
        f"案件类型：{doc['case_type'] or '未标注'}",
    ]
    if doc.get("case_title"):
        lines.append(f"案件标题：{doc['case_title']}")
    if doc.get("risk_level"):
        lines.append(f"风险等级：{doc['risk_level']}")
    if doc.get("risk_score") is not None:
        lines.append(f"风险分值：{doc['risk_score']}")
    if doc.get("emotion_label"):
        lines.append(f"情感状态：{doc['emotion_label']}")
    lines.extend(
        [
            f"指令：{doc['instruction']}",
            f"输入：{doc['input']}",
            f"参考输出：{doc['output']}",
        ]
    )
    return "\n".join(lines)


def to_case_doc(
    source_file: str,
    source_split: str,
    source_line_no: int,
    row: Dict[str, str],
    version: str,
) -> Dict[str, object]:
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    output_text = row.get("output", "").strip()
    task_type = infer_task_type(instruction, output_text)
    doc_type = task_to_doc_type(task_type)
    case_type = extract_case_type(input_text)
    case_title = extract_case_title(input_text)
    risk_level = extract_risk_level(output_text) if task_type == TASK_RISK else ""
    risk_score = extract_risk_score(output_text) if task_type == TASK_RISK else None
    emotion_label = extract_emotion(output_text) if task_type == TASK_EMOTION else ""

    base_doc = {
        "doc_type": doc_type,
        "task_type": task_type,
        "task_scope": [task_type],
        "source_file": source_file,
        "source_split": source_split,
        "source_line_no": source_line_no,
        "case_type": case_type,
        "case_title": case_title,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "emotion_label": emotion_label,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "keywords": to_keywords(case_type, task_type, risk_level, emotion_label),
        "data_version": version,
    }
    content = build_case_content(base_doc)
    base_doc["id"] = stable_id("doc", f"{doc_type}|{source_file}|{source_line_no}|{content}")
    base_doc["content"] = content
    return base_doc


def build_reference_docs(version: str) -> List[Dict[str, object]]:
    docs: List[Dict[str, object]] = []
    candidates = [
        (RISK_DATA_DIR / "risk_scoring_method.md", "methodology", [TASK_RISK]),
        (RISK_DATA_DIR / "README.md", "dataset_readme", [TASK_RISK, TASK_EMOTION, TASK_DISPOSITION]),
        (ROOT / "output" / "风险项目数据标准化技术方案.md", "data_standard_plan", [TASK_RISK, TASK_EMOTION]),
        (ROOT / "output" / "风险项目数据标准化小白说明.md", "data_standard_explainer", [TASK_RISK, TASK_EMOTION]),
    ]

    for path, doc_type, task_scope in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative_path = str(path.relative_to(ROOT))
        docs.append(
            {
                "id": stable_id("ref", f"{doc_type}|{text}"),
                "doc_type": doc_type,
                "task_type": "reference",
                "task_scope": task_scope,
                "source_file": relative_path,
                "source_split": "reference",
                "source_line_no": 0,
                "case_type": "",
                "case_title": "",
                "risk_level": "",
                "risk_score": None,
                "emotion_label": "",
                "instruction": "",
                "input": "",
                "output": "",
                "keywords": to_keywords(doc_type, *task_scope),
                "content": text,
                "data_version": version,
            }
        )
    return docs


def build_case_summaries(case_docs: Sequence[Dict[str, object]], version: str, source_tag: str) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for doc in case_docs:
        case_type = str(doc.get("case_type") or "未标注")
        grouped[(str(doc["task_type"]), case_type)].append(doc)

    summaries: List[Dict[str, object]] = []
    for (task_type, case_type), items in sorted(grouped.items()):
        if task_type == TASK_RISK:
            levels = Counter(str(item.get("risk_level") or "未标注") for item in items)
            scores = [int(item["risk_score"]) for item in items if isinstance(item.get("risk_score"), int)]
            dist_text = "；".join(f"{level}={count}" for level, count in sorted(levels.items()))
            if scores:
                score_text = f"平均分值：{round(sum(scores) / len(scores), 2)}，最低分：{min(scores)}，最高分：{max(scores)}"
            else:
                score_text = "平均分值：无精确分值"
            summary_body = (
                f"文档类型：案件类型摘要\n"
                f"任务类型：{task_type}\n"
                f"案件类型：{case_type}\n"
                f"样本数量：{len(items)}\n"
                f"等级分布：{dist_text}\n"
                f"{score_text}\n"
                f"用途：给风险评估任务提供该类型历史样本分布。"
            )
        elif task_type == TASK_EMOTION:
            emotions = Counter(str(item.get("emotion_label") or "未标注") for item in items)
            dist_text = "；".join(f"{emotion}={count}" for emotion, count in sorted(emotions.items()))
            summary_body = (
                f"文档类型：案件类型摘要\n"
                f"任务类型：{task_type}\n"
                f"案件类型：{case_type}\n"
                f"样本数量：{len(items)}\n"
                f"情绪分布：{dist_text}\n"
                f"用途：给情绪分析任务提供该类型历史样本分布。"
            )
        else:
            summary_body = (
                f"文档类型：案件类型摘要\n"
                f"任务类型：{task_type}\n"
                f"案件类型：{case_type}\n"
                f"样本数量：{len(items)}\n"
                f"用途：给该任务提供案件类型经验样本。"
            )

        summaries.append(
            {
                "id": stable_id("sum", f"{task_type}|{case_type}|{summary_body}"),
                "doc_type": DOC_SUMMARY,
                "task_type": "summary",
                "task_scope": [task_type],
                "source_file": source_tag,
                "source_split": "summary",
                "source_line_no": 0,
                "case_type": case_type,
                "case_title": "",
                "risk_level": "",
                "risk_score": None,
                "emotion_label": "",
                "instruction": "",
                "input": "",
                "output": "",
                "keywords": to_keywords(task_type, case_type, "summary"),
                "content": summary_body,
                "data_version": version,
            }
        )
    return summaries


def deduplicate_docs(docs: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    unique_docs: List[Dict[str, object]] = []
    seen = set()
    dropped = 0
    for doc in docs:
        fingerprint = stable_id(
            "dup",
            f"{doc.get('doc_type')}|{doc.get('task_type')}|{normalize_space(str(doc.get('content', '')))}",
        )
        if fingerprint in seen:
            dropped += 1
            continue
        seen.add(fingerprint)
        unique_docs.append(doc)
    return unique_docs, dropped


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    combined_output = Path(args.output)
    version = args.version or datetime.now(timezone.utc).strftime("v%Y%m%d.1")

    source_paths = [(Path(args.train_file), "train")]
    if args.include_val:
        source_paths.append((Path(args.val_file), "val"))

    case_docs: List[Dict[str, object]] = []
    source_rows = {}
    for path, split in source_paths:
        rows = list(load_jsonl(path))
        source_rows[split] = len(rows)
        source_file = str(path.relative_to(ROOT))
        for line_no, row in rows:
            case_docs.append(to_case_doc(source_file, split, line_no, row, version))

    source_tag = "+".join(str(path.relative_to(ROOT)) for path, _ in source_paths)
    summary_docs = build_case_summaries(case_docs, version, source_tag)
    reference_docs = build_reference_docs(version)
    all_docs, dedup_dropped = deduplicate_docs(case_docs + summary_docs + reference_docs)
    all_docs = sorted(all_docs, key=lambda doc: (str(doc.get("doc_type")), str(doc.get("id"))))

    by_doc_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for doc in all_docs:
        by_doc_type[str(doc["doc_type"])].append(doc)

    category_files = {
        DOC_RISK: output_dir / "kb_risk_cases.jsonl",
        DOC_EMOTION: output_dir / "kb_emotion_cases.jsonl",
        DOC_DISPOSITION: output_dir / "kb_disposition_cases.jsonl",
        DOC_GENERIC: output_dir / "kb_generic_cases.jsonl",
        DOC_SUMMARY: output_dir / "kb_case_summaries.jsonl",
        "reference": output_dir / "kb_reference_docs.jsonl",
    }

    write_jsonl(combined_output, all_docs)
    write_jsonl(category_files[DOC_RISK], by_doc_type.get(DOC_RISK, []))
    write_jsonl(category_files[DOC_EMOTION], by_doc_type.get(DOC_EMOTION, []))
    write_jsonl(category_files[DOC_DISPOSITION], by_doc_type.get(DOC_DISPOSITION, []))
    write_jsonl(category_files[DOC_GENERIC], by_doc_type.get(DOC_GENERIC, []))
    write_jsonl(category_files[DOC_SUMMARY], by_doc_type.get(DOC_SUMMARY, []))

    reference_rows: List[Dict[str, object]] = []
    for reference_type in ("methodology", "dataset_readme", "data_standard_plan", "data_standard_explainer"):
        reference_rows.extend(by_doc_type.get(reference_type, []))
    write_jsonl(category_files["reference"], reference_rows)

    doc_type_counts = {key: len(value) for key, value in sorted(by_doc_type.items())}
    task_counts = Counter(str(doc.get("task_type")) for doc in all_docs)
    stats = {
        "version": version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_val": bool(args.include_val),
        "source_rows": source_rows,
        "source_files": [str(path.relative_to(ROOT)) for path, _ in source_paths],
        "doc_type_counts": doc_type_counts,
        "task_type_counts": dict(sorted(task_counts.items())),
        "deduplicated_docs": dedup_dropped,
        "total_docs": len(all_docs),
        "outputs": {
            "combined": str(combined_output),
            **{name: str(path) for name, path in category_files.items()},
        },
    }
    manifest = {
        "name": "risk-rag-kb",
        "version": version,
        "description": "Classified RAG assets for risk assessment project",
        "build_args": {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "include_val": bool(args.include_val),
            "output_dir": str(output_dir),
            "output": str(combined_output),
        },
        "stats": stats,
    }

    STATS_OUTPUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
