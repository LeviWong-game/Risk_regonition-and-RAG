#!/usr/bin/env python3
"""调试关键词匹配"""

import sys
sys.path.insert(0, r'e:\风险预测和评估\risk_data')

from risk_scoring import (
    parse_case_text, 
    _collect_indicator_score, 
    _apply_case_type_hints,
    _find_keywords,
    normalize_indicator,
    PATTERN_RULES,
    INDICATOR_LABELS
)

TEST_TEXT = """案件类型：劳动争议
案情：农民工工伤后急需赔偿，多次协商无果，并扬言聚集维权。"""

def debug_matching():
    parsed = parse_case_text(TEST_TEXT)
    raw_text = parsed["raw_text"]
    case_type = parsed["case_type"]
    
    print("=" * 70)
    print(f"解析后的文本:")
    print(f"  案件类型：{case_type}")
    print(f"  原始文本：{raw_text}")
    print("=" * 70)
    print()
    
    print("逐个指标检查匹配情况:")
    print("=" * 70)
    
    for indicator_key in INDICATOR_LABELS:
        raw_score, evidence = _collect_indicator_score(indicator_key, raw_text)
        raw_score_after, evidence_after = _apply_case_type_hints(case_type, indicator_key, raw_score, evidence)
        normalized = normalize_indicator(raw_score_after)
        
        # 只显示有匹配或分数被提升的指标
        if raw_score > 1 or raw_score_after > raw_score:
            print(f"\n{indicator_key}:")
            print(f"  关键词匹配分数：{raw_score}")
            print(f"  匹配证据：{evidence}")
            print(f"  案件类型修正后分数：{raw_score_after}")
            print(f"  修正证据：{evidence_after}")
            print(f"  标准化分数：{normalized}")
    
    print()
    print("=" * 70)
    print("所有指标的 PATTERN_RULES 关键词:")
    print("=" * 70)
    
    # 检查文本中实际包含哪些关键词
    for indicator_key, rules in PATTERN_RULES.items():
        print(f"\n{indicator_key}:")
        for score, keywords in rules.items():
            hits = _find_keywords(raw_text, keywords)
            if hits:
                print(f"  {score}分：命中 {hits}")

if __name__ == "__main__":
    debug_matching()
