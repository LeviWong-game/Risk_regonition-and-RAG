#!/usr/bin/env python3
"""完整评估调试"""

import sys
sys.path.insert(0, r'e:\风险预测和评估\risk_data')

from risk_scoring import assess_case

TEST_TEXT = """案件类型：劳动争议
案情：农民工工伤后急需赔偿，多次协商无果，并扬言聚集维权。"""

def debug_full():
    result = assess_case(TEST_TEXT)
    
    print("=" * 70)
    print("风险评估结果")
    print("=" * 70)
    print(f"风险总分：{result.risk_score}")
    print(f"风险等级：{result.risk_level}")
    print(f"P (冲突性): {result.P:.4f}")
    print(f"I (影响性): {result.I:.4f}")
    print(f"D (处置难度): {result.D:.4f}")
    print(f"触发规则：{result.trigger_flags}")
    print(f"处置建议：{result.recommendation}")
    print()
    
    print("=" * 70)
    print("12 个指标详情")
    print("=" * 70)
    for key, indicator in result.indicators.items():
        print(f"{indicator.label}:")
        print(f"  原始分={indicator.raw_score}, 标准化={indicator.normalized_score}")
        print(f"  证据={indicator.evidence}")
    print()
    
    print("=" * 70)
    print("解释文本")
    print("=" * 70)
    print(result.reason_text)

if __name__ == "__main__":
    debug_full()
