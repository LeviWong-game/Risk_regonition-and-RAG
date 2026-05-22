#!/usr/bin/env python3
"""Debug keyword, case-type prior, and indicator scoring signals."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_DATA_DIR = PROJECT_ROOT / "risk_data"
sys.path.insert(0, str(RISK_DATA_DIR))

from risk_scoring import (  # noqa: E402
    INDICATOR_LABELS,
    PATTERN_RULES,
    _apply_case_type_prior,
    _collect_indicator_signal,
    _find_keywords,
    _matched_case_type_names,
    _normalize_for_match,
    normalize_indicator,
    parse_case_text,
)


TEST_TEXT = """案件类型：劳动争议
案情：农民工工伤后急需赔偿，多次协商无果，并扬言聚集维权。"""


def debug_matching() -> None:
    parsed = parse_case_text(TEST_TEXT)
    raw_text = parsed["raw_text"]
    case_type = parsed["case_type"]
    searchable_text = "\n".join(
        part for part in (parsed["title"], parsed["description"]) if part
    ).strip() or raw_text
    matched_case_types = _matched_case_type_names(case_type)

    print("=" * 70)
    print("解析后的文本")
    print(f"  案件类型：{case_type}")
    print(f"  命中案件类型先验：{matched_case_types}")
    print(f"  检索文本：{searchable_text}")
    print("=" * 70)
    print()

    print("逐个指标检查匹配情况")
    print("=" * 70)
    for indicator_key, indicator_label in INDICATOR_LABELS.items():
        text_score, evidence, evidence_count = _collect_indicator_signal(
            indicator_key,
            searchable_text,
        )
        final_score, final_evidence = _apply_case_type_prior(
            matched_case_types,
            indicator_key,
            text_score,
            evidence_count,
            evidence,
        )
        normalized = normalize_indicator(final_score)

        if text_score > 1 or final_score > text_score:
            print(f"\n{indicator_label} ({indicator_key}):")
            print(f"  文本证据分：{text_score:.4f}")
            print(f"  文本证据数量：{evidence_count}")
            print(f"  文本证据：{evidence}")
            print(f"  先验融合后分：{final_score:.4f}")
            print(f"  融合后证据：{final_evidence}")
            print(f"  标准化分：{normalized}")

    print()
    print("=" * 70)
    print("PATTERN_RULES 实际命中关键词")
    print("=" * 70)
    normalized_text = _normalize_for_match(searchable_text)
    for indicator_key, rules in PATTERN_RULES.items():
        printed_indicator = False
        for score, keywords in rules.items():
            hits = _find_keywords(searchable_text, keywords, normalized_text)
            if not hits:
                continue
            if not printed_indicator:
                print(f"\n{indicator_key}:")
                printed_indicator = True
            print(f"  {score} 分：命中 {hits}")


if __name__ == "__main__":
    debug_matching()
