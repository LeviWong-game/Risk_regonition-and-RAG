from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

try:
    from .risk_config import (
        ANCHOR_TEXT,
        INDICATOR_DIMENSIONS,
        INDICATOR_LABELS,
        LEVEL_ACTIONS,
        LEVEL_THRESHOLDS,
        PRIMARY_WEIGHTS,
        SECONDARY_WEIGHTS,
    )
    from .risk_matching import (
        CASE_TYPE_ALIASES,
        CASE_TYPE_HINTS,
        FIELD_ALIASES,
        KEYWORD_ALIASES,
        PATTERN_RULES,
        RED_FLAG_L3,
        RED_FLAG_L4,
        TRIGGER_PATTERNS,
        _find_keywords,
        _matched_case_type_names,
        _normalize_for_match,
        parse_case_text,
    )
except ImportError:
    from risk_config import (  # type: ignore
        ANCHOR_TEXT,
        INDICATOR_DIMENSIONS,
        INDICATOR_LABELS,
        LEVEL_ACTIONS,
        LEVEL_THRESHOLDS,
        PRIMARY_WEIGHTS,
        SECONDARY_WEIGHTS,
    )
    from risk_matching import (  # type: ignore
        CASE_TYPE_ALIASES,
        CASE_TYPE_HINTS,
        FIELD_ALIASES,
        KEYWORD_ALIASES,
        PATTERN_RULES,
        RED_FLAG_L3,
        RED_FLAG_L4,
        TRIGGER_PATTERNS,
        _find_keywords,
        _matched_case_type_names,
        _normalize_for_match,
        parse_case_text,
    )


@dataclass(frozen=True)
class RiskIndicatorResult:
    key: str
    label: str
    dimension: str
    raw_score: int
    normalized_score: float
    anchor_text: str
    evidence: List[str]


@dataclass(frozen=True)
class RiskAssessment:
    case_type: str
    title: str
    description: str
    risk_score: int
    risk_level: str
    P: float
    I: float
    D: float
    trigger_flags: List[str]
    recommendation: str
    reason_text: str
    indicators: Dict[str, RiskIndicatorResult]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["indicators"] = {k: asdict(v) for k, v in self.indicators.items()}
        payload["views"] = {
            "resident": _build_resident_view(self),
            "management": _build_management_view(self),
        }
        return payload


def normalize_indicator(raw_score: int) -> float:
    clipped = min(5, max(1, raw_score))
    return round((clipped - 1) / 4, 4)


def classify_level(score: int) -> str:
    bounded = min(100, max(0, int(score)))
    for level, low, high in LEVEL_THRESHOLDS:
        if low <= bounded <= high:
            return level
    return "L4"


def _collect_indicator_score(indicator_key: str, text: str) -> Tuple[int, List[str]]:
    best_score = 1
    evidence: List[str] = []
    normalized_text = _normalize_for_match(text)
    for score, keywords in PATTERN_RULES[indicator_key].items():
        hits = _find_keywords(text, keywords, normalized_text)
        if not hits:
            continue
        if score > best_score:
            best_score = score
            evidence = hits
        elif score == best_score:
            for item in hits:
                if item not in evidence:
                    evidence.append(item)
    return best_score, evidence


def _apply_case_type_hints(case_type: str, indicator_key: str, score: int, evidence: List[str]) -> Tuple[int, List[str]]:
    for name in _matched_case_type_names(case_type):
        hints = CASE_TYPE_HINTS[name]
        if indicator_key not in hints:
            continue
        hinted = hints[indicator_key]
        if hinted > score:
            score = hinted
            evidence = evidence + [f"案件类型提示:{name}"]
    return score, evidence


def _dimension_score(indicator_keys: Sequence[str], indicator_results: Dict[str, RiskIndicatorResult]) -> float:
    dimension = INDICATOR_DIMENSIONS[indicator_keys[0]]
    total = 0.0
    for key in indicator_keys:
        total += SECONDARY_WEIGHTS[dimension][key] * indicator_results[key].normalized_score
    return round(total, 4)


def _trigger_flags(text: str) -> List[str]:
    flags: List[str] = []
    normalized_text = _normalize_for_match(text)
    for flag, keywords in TRIGGER_PATTERNS.items():
        if _find_keywords(text, keywords, normalized_text):
            flags.append(flag)
    return flags


def _apply_level_floor(score: int, trigger_flags: Sequence[str]) -> int:
    red_flag_hits = sum(1 for flag in trigger_flags if flag in RED_FLAG_L3)
    has_l4_flag = any(flag in RED_FLAG_L4 for flag in trigger_flags)
    bounded = min(100, max(0, score))
    if has_l4_flag or red_flag_hits >= 2:
        return max(bounded, 76)
    if red_flag_hits >= 1:
        return max(bounded, 51)
    return bounded


def _top_indicators_for_dimension(
    dimension: str,
    indicator_results: Dict[str, RiskIndicatorResult],
) -> List[RiskIndicatorResult]:
    items = [item for item in indicator_results.values() if item.dimension == dimension]
    items.sort(key=lambda item: (item.raw_score, item.normalized_score, len(item.evidence)), reverse=True)
    tops = [item for item in items if item.raw_score >= 4][:2]
    if not tops:
        tops = items[:2]
    return tops


def _reason_segment(dimension: str, indicator_results: Dict[str, RiskIndicatorResult]) -> str:
    segments: List[str] = []
    for item in _top_indicators_for_dimension(dimension, indicator_results):
        if item.evidence:
            evidence = "、".join(item.evidence[:3])
            segments.append(f"{item.label}={item.raw_score}({evidence})")
        else:
            segments.append(f"{item.label}={item.raw_score}({item.anchor_text})")
    return "；".join(segments)


def _recommendation(level: str) -> str:
    return LEVEL_ACTIONS[level]


def _level_label(level: str) -> str:
    labels = {
        "L1": "低风险",
        "L2": "一般风险",
        "L3": "较高风险",
        "L4": "高风险",
    }
    return labels.get(level, level)


def _pid_score_100(assessment: RiskAssessment) -> Dict[str, int]:
    keys = ("P", "I", "D")
    values = [max(0.0, float(assessment.P)), max(0.0, float(assessment.I)), max(0.0, float(assessment.D))]
    total = sum(values)
    if total <= 0:
        return {"P": 34, "I": 33, "D": 33}

    raw = [value * 100 / total for value in values]
    floors = [int(value) for value in raw]
    remain = 100 - sum(floors)
    # 用最大余数法分配余量，确保三项加总严格等于 100。
    fractions = sorted(
        ((raw[idx] - floors[idx], idx) for idx in range(len(keys))),
        key=lambda item: item[0],
        reverse=True,
    )
    for i in range(remain):
        floors[fractions[i][1]] += 1
    return {keys[idx]: floors[idx] for idx in range(len(keys))}


def _pid_score_text(assessment: RiskAssessment) -> str:
    scores = _pid_score_100(assessment)
    return f"P：{scores['P']}，I：{scores['I']}，D：{scores['D']}"


def _build_resident_view(assessment: RiskAssessment) -> Dict[str, object]:
    trigger_text = "无"
    if assessment.trigger_flags:
        trigger_text = "、".join(assessment.trigger_flags)
    pid_scores = _pid_score_100(assessment)
    return {
        "结论": f"{_level_label(assessment.risk_level)}（{assessment.risk_level}）",
        "风险分数": assessment.risk_score,
        "P_I_D_分数": pid_scores,
        "P_I_D_文本": _pid_score_text(assessment),
        "建议": assessment.recommendation,
        "主要原因": assessment.reason_text,
        "触发规则": trigger_text,
    }


def _build_management_view(assessment: RiskAssessment) -> Dict[str, object]:
    dimension_scores = {
        "P": round(assessment.P, 4),
        "I": round(assessment.I, 4),
        "D": round(assessment.D, 4),
    }
    formula_result = round(100 * (0.75 * assessment.P * assessment.I + 0.25 * assessment.D))
    indicator_breakdown: Dict[str, List[Dict[str, object]]] = {"P": [], "I": [], "D": []}
    for indicator_key, indicator in assessment.indicators.items():
        dimension = indicator.dimension
        weight = SECONDARY_WEIGHTS[dimension][indicator_key]
        indicator_breakdown[dimension].append(
            {
                "key": indicator.key,
                "label": indicator.label,
                "weight": weight,
                "raw_score": indicator.raw_score,
                "normalized_score": indicator.normalized_score,
                "weighted_component": round(weight * indicator.normalized_score, 4),
                "evidence": indicator.evidence,
            }
        )
    for dimension in indicator_breakdown:
        indicator_breakdown[dimension].sort(key=lambda item: item["weighted_component"], reverse=True)
    return {
        "formula": "risk_score = round(100 * (0.75 * P * I + 0.25 * D))",
        "weights": {
            "primary": PRIMARY_WEIGHTS,
            "secondary": SECONDARY_WEIGHTS,
        },
        "dimension_scores": dimension_scores,
        "dimension_scores_100": _pid_score_100(assessment),
        "dimension_scores_100_text": _pid_score_text(assessment),
        "raw_formula_score_before_floor": formula_result,
        "trigger_flags": assessment.trigger_flags,
        "indicator_breakdown": indicator_breakdown,
    }


def _trigger_flag_text(flag: str) -> str:
    labels = {
        "injury_or_violence": "出现人身伤害或暴力风险",
        "public_safety_hazard": "存在公共安全隐患",
        "gathering_or_public_opinion": "存在聚集或舆情扩散风险",
        "judicial_or_police": "已涉及公安/司法介入",
    }
    return labels.get(flag, flag)


def _render_resident_result_text(assessment: RiskAssessment) -> str:
    trigger_text = "未触发"
    if assessment.trigger_flags:
        trigger_text = "；".join(_trigger_flag_text(flag) for flag in assessment.trigger_flags)
    return (
        f"【风险结论】{_level_label(assessment.risk_level)}（{assessment.risk_level}），综合风险分 {assessment.risk_score} 分。"
        f"【指标分布】{_pid_score_text(assessment)}。"
        f"【主要原因】{assessment.reason_text}"
        f"【建议处置】{assessment.recommendation}。"
        f"【风险提示】{trigger_text}。"
    )


def _write_programmer_log(text: str, assessment: RiskAssessment, pretty: bool) -> str:
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(logs_dir, f"risk_assessment_{stamp}.json")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_text": text,
        "result": assessment.to_dict(),
    }
    indent = 2 if pretty else None
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    return file_path


def combine_weights(subjective_weights: Dict[str, float], entropy_weights: Dict[str, float]) -> Dict[str, float]:
    common_keys = [key for key in subjective_weights if key in entropy_weights]
    if not common_keys:
        raise ValueError("No overlapping keys between subjective and entropy weights.")
    merged = {key: math.sqrt(subjective_weights[key] * entropy_weights[key]) for key in common_keys}
    total = sum(merged.values())
    if total == 0:
        raise ValueError("Combined weights sum to zero.")
    return {key: value / total for key, value in merged.items()}


def entropy_weight(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    keys = list(records[0].keys())
    columns: Dict[str, List[float]] = {key: [] for key in keys}
    for record in records:
        for key in keys:
            columns[key].append(float(record[key]))

    normalized: Dict[str, List[float]] = {}
    for key, values in columns.items():
        total = sum(values)
        if total == 0:
            normalized[key] = [1 / len(values)] * len(values)
        else:
            normalized[key] = [value / total for value in values]

    n = len(records)
    if n == 1:
        return {key: 1 / len(keys) for key in keys}

    entropy: Dict[str, float] = {}
    k = 1 / math.log(n)
    for key, values in normalized.items():
        entropy[key] = -k * sum(value * math.log(value) for value in values if value > 0)

    divergence = {key: 1 - value for key, value in entropy.items()}
    total_divergence = sum(divergence.values())
    if total_divergence == 0:
        return {key: 1 / len(keys) for key in keys}
    return {key: value / total_divergence for key, value in divergence.items()}


def assess_case(text: str) -> RiskAssessment:
    parsed = parse_case_text(text)
    case_type = parsed["case_type"]
    raw_text = parsed["raw_text"]
    indicator_results: Dict[str, RiskIndicatorResult] = {}

    for indicator_key in INDICATOR_LABELS:
        raw_score, evidence = _collect_indicator_score(indicator_key, raw_text)
        raw_score, evidence = _apply_case_type_hints(case_type, indicator_key, raw_score, evidence)
        normalized = normalize_indicator(raw_score)
        indicator_results[indicator_key] = RiskIndicatorResult(
            key=indicator_key,
            label=INDICATOR_LABELS[indicator_key],
            dimension=INDICATOR_DIMENSIONS[indicator_key],
            raw_score=raw_score,
            normalized_score=normalized,
            anchor_text=ANCHOR_TEXT[indicator_key][raw_score],
            evidence=evidence,
        )

    p_keys = tuple(SECONDARY_WEIGHTS["P"].keys())
    i_keys = tuple(SECONDARY_WEIGHTS["I"].keys())
    d_keys = tuple(SECONDARY_WEIGHTS["D"].keys())
    p_score = _dimension_score(p_keys, indicator_results)
    i_score = _dimension_score(i_keys, indicator_results)
    d_score = _dimension_score(d_keys, indicator_results)

    raw_score = round(100 * (0.75 * p_score * i_score + 0.25 * d_score))
    trigger_flags = _trigger_flags(raw_text)
    final_score = _apply_level_floor(raw_score, trigger_flags)
    level = classify_level(final_score)

    trigger_text = "无"
    if trigger_flags:
        trigger_text = "、".join(trigger_flags)
    reason_text = (
        f"P高在{_reason_segment('P', indicator_results)}；"
        f"I高在{_reason_segment('I', indicator_results)}；"
        f"D高在{_reason_segment('D', indicator_results)}；"
        f"触发规则: {trigger_text}。"
    )

    return RiskAssessment(
        case_type=case_type,
        title=parsed["title"],
        description=parsed["description"],
        risk_score=final_score,
        risk_level=level,
        P=p_score,
        I=i_score,
        D=d_score,
        trigger_flags=trigger_flags,
        recommendation=_recommendation(level),
        reason_text=reason_text,
        indicators=indicator_results,
    )


def _setup_windows_console_utf8() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _cli() -> int:
    _setup_windows_console_utf8()
    parser = argparse.ArgumentParser(description="Rule-based, explainable risk scoring for dispute cases.")
    parser.add_argument("--text", help="Full case text with optional labels such as 案件类型/案情.")
    parser.add_argument("--input-file", help="Path to a UTF-8 text file containing one case.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    text = args.text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    elif not text:
        print("请输入案件文本（可多行，空行结束）：")
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)
        text = "\n".join(lines)

    assessment = assess_case(text or "")
    log_path = _write_programmer_log(text or "", assessment, args.pretty)
    print(_render_resident_result_text(assessment))
    print(f"（详细日志已保存：{log_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
