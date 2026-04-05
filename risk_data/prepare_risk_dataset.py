from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from risk_config import LEVEL_ACTIONS, LEVEL_THRESHOLDS
from risk_engine import assess_case
from risk_matching import _normalize_for_match, parse_case_text


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILES = ("train.jsonl", "val.jsonl")
TARGET_FILES = {
    "master": "risk_cases_master.jsonl",
    "train": "risk_train_clean.jsonl",
    "val": "risk_val_clean.jsonl",
    "report": "risk_dataset_report.json",
}

CANONICAL_INSTRUCTION = "请根据案件类型和案件描述，预测风险等级并给出处置建议。"
CHAT_SYSTEM_PROMPT = "你是基层矛盾纠纷风险预测助手，需要基于案件信息输出风险等级和处置建议。"
LABEL_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}


def extract_risk_label(output: str) -> str | None:
    match = re.search(r"L[1-4]", output or "")
    return match.group(0) if match else None


def threshold_range(level: str) -> Tuple[int, int]:
    for name, low, high in LEVEL_THRESHOLDS:
        if name == level:
            return low, high
    raise KeyError(level)


def canonical_case_text(case_type: str, title: str, description: str) -> str:
    parts: List[str] = []
    if title:
        parts.append(f"案件标题：{title}")
    if case_type:
        parts.append(f"案件类型：{case_type}")
    if description:
        parts.append(f"案情：{description}")
    return "\n".join(parts).strip()


def case_key(case_type: str, title: str, description: str) -> str:
    packed = "||".join([_normalize_for_match(case_type), _normalize_for_match(title), _normalize_for_match(description)])
    return hashlib.md5(packed.encode("utf-8")).hexdigest()


def resolve_label(case_text: str, labels: List[str]) -> Tuple[str, str, float, Dict[str, int]]:
    votes = Counter(labels)
    if len(votes) == 1:
        only = next(iter(votes))
        strategy = "single_source" if sum(votes.values()) == 1 else "unanimous"
        return only, strategy, 1.0, dict(votes)

    top_vote = max(votes.values())
    top_labels = sorted([label for label, count in votes.items() if count == top_vote], key=lambda item: LABEL_ORDER[item])
    confidence = round(top_vote / sum(votes.values()), 4)

    if len(top_labels) == 1:
        return top_labels[0], "majority_vote", confidence, dict(votes)

    heuristic_label = assess_case(case_text).risk_level
    if heuristic_label in top_labels:
        return heuristic_label, "heuristic_tiebreak", confidence, dict(votes)

    conservative = max(top_labels, key=lambda item: LABEL_ORDER[item])
    return conservative, "conservative_tiebreak", confidence, dict(votes)


def load_risk_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source_name in SOURCE_FILES:
        source_path = BASE_DIR / source_name
        with source_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                record = json.loads(line)
                label = extract_risk_label(record.get("output", ""))
                if not label:
                    continue
                parsed = parse_case_text(record.get("input", ""))
                rows.append(
                    {
                        "source_file": source_name,
                        "line_no": line_no,
                        "case_type": parsed["case_type"].strip(),
                        "title": parsed["title"].strip(),
                        "description": parsed["description"].strip(),
                        "raw_input": record.get("input", "").strip(),
                        "raw_output": record.get("output", "").strip(),
                        "label": label,
                    }
                )
    return rows


def load_master_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def build_master_records(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = case_key(row["case_type"], row["title"], row["description"])  # type: ignore[arg-type]
        grouped[key].append(row)

    master_records: List[Dict[str, object]] = []
    for key, records in grouped.items():
        exemplar = records[0]
        case_text = canonical_case_text(
            exemplar["case_type"],  # type: ignore[arg-type]
            exemplar["title"],  # type: ignore[arg-type]
            exemplar["description"],  # type: ignore[arg-type]
        )
        labels = [record["label"] for record in records]  # type: ignore[index]
        resolved_label, strategy, confidence, votes = resolve_label(case_text, labels)
        score_min, score_max = threshold_range(resolved_label)
        source_refs = [f"{record['source_file']}:{record['line_no']}" for record in records]
        assistant_output = json.dumps(
            {
                "risk_level": resolved_label,
                "risk_score_range": [score_min, score_max],
                "recommendation": LEVEL_ACTIONS[resolved_label],
            },
            ensure_ascii=False,
        )

        sample_id = f"risk-{len(master_records) + 1:04d}"
        master_records.append(
            {
                "sample_id": sample_id,
                "task": "risk_prediction",
                "case_type": exemplar["case_type"],
                "title": exemplar["title"],
                "description": exemplar["description"],
                "case_text": case_text,
                "risk_level": resolved_label,
                "risk_score_min": score_min,
                "risk_score_max": score_max,
                "recommendation": LEVEL_ACTIONS[resolved_label],
                "label_confidence": confidence,
                "label_votes": votes,
                "label_conflict": len(votes) > 1,
                "resolution_strategy": strategy,
                "source_count": len(records),
                "source_refs": source_refs,
                "instruction": CANONICAL_INSTRUCTION,
                "input": case_text,
                "output": assistant_output,
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{CANONICAL_INSTRUCTION}\n\n{case_text}"},
                    {"role": "assistant", "content": assistant_output},
                ],
                "sft_instruction": CANONICAL_INSTRUCTION,
                "sft_input": case_text,
                "sft_output": assistant_output,
            }
        )

    master_records.sort(key=lambda item: item["sample_id"])
    return master_records


def stable_label_split(records: List[Dict[str, object]], train_ratio: float = 0.8) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_records: List[Dict[str, object]] = []
    val_records: List[Dict[str, object]] = []

    by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_label[record["risk_level"]].append(record)

    for label, items in by_label.items():
        items.sort(key=lambda item: hashlib.md5(item["case_text"].encode("utf-8")).hexdigest())
        split_index = max(1, round(len(items) * train_ratio)) if len(items) > 1 else len(items)
        train_records.extend({**item, "split": "train"} for item in items[:split_index])
        val_records.extend({**item, "split": "val"} for item in items[split_index:])

    train_records.sort(key=lambda item: item["sample_id"])
    val_records.sort(key=lambda item: item["sample_id"])
    return train_records, val_records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(master: List[Dict[str, object]], train: List[Dict[str, object]], val: List[Dict[str, object]]) -> Dict[str, object]:
    conflict_cases = [record for record in master if record["label_conflict"]]
    return {
        "source_files": list(SOURCE_FILES),
        "raw_risk_rows": sum(record["source_count"] for record in master),
        "unique_risk_cases": len(master),
        "conflict_case_count": len(conflict_cases),
        "conflict_resolution_breakdown": Counter(record["resolution_strategy"] for record in master),
        "master_label_distribution": Counter(record["risk_level"] for record in master),
        "train_size": len(train),
        "val_size": len(val),
        "train_label_distribution": Counter(record["risk_level"] for record in train),
        "val_label_distribution": Counter(record["risk_level"] for record in val),
    }


def main() -> int:
    source_paths = [BASE_DIR / name for name in SOURCE_FILES]
    master_path = BASE_DIR / TARGET_FILES["master"]

    if all(path.exists() for path in source_paths):
        rows = load_risk_rows()
        master = build_master_records(rows)
    elif master_path.exists():
        master = load_master_records(master_path)
    else:
        raise FileNotFoundError("Neither raw source files nor risk_cases_master.jsonl are available.")

    train, val = stable_label_split(master)
    report = build_report(master, train, val)

    write_jsonl(BASE_DIR / TARGET_FILES["master"], master)
    write_jsonl(BASE_DIR / TARGET_FILES["train"], train)
    write_jsonl(BASE_DIR / TARGET_FILES["val"], val)
    with (BASE_DIR / TARGET_FILES["report"]).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
