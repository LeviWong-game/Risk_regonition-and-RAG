from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from .risk_config import (
        ANCHOR_TEXT,
        FORMULA_WEIGHTS,
        INDICATOR_DIMENSIONS,
        INDICATOR_LABELS,
        LEVEL_ACTIONS,
        LEVEL_THRESHOLDS,
        PRIOR_BLEND_WEIGHTS,
        SCORE_CALIBRATION,
        SECONDARY_WEIGHTS,
    )
    from .risk_matching import (
        CASE_TYPE_HINTS,
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
        FORMULA_WEIGHTS,
        INDICATOR_DIMENSIONS,
        INDICATOR_LABELS,
        LEVEL_ACTIONS,
        LEVEL_THRESHOLDS,
        PRIOR_BLEND_WEIGHTS,
        SCORE_CALIBRATION,
        SECONDARY_WEIGHTS,
    )
    from risk_matching import (  # type: ignore
        CASE_TYPE_HINTS,
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
    continuous_score: float
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
        payload["indicators"] = {key: asdict(value) for key, value in self.indicators.items()}
        payload["views"] = {
            "resident": _build_resident_view(self),
            "management": _build_management_view(self),
        }
        return payload


def normalize_indicator(score: float) -> float:
    clipped = min(5.0, max(1.0, float(score)))
    return round((clipped - 1.0) / 4.0, 4)


def band_indicator(score: float) -> int:
    return min(5, max(1, int(round(score))))


def classify_level(score: int, thresholds: Sequence[Tuple[str, int, int]] = LEVEL_THRESHOLDS) -> str:
    bounded = min(100, max(0, int(score)))
    for level, low, high in thresholds:
        if low <= bounded <= high:
            return level
    return thresholds[-1][0]


def score_formula(
    p_score: float,
    i_score: float,
    d_score: float,
    weights: Dict[str, float] = FORMULA_WEIGHTS,
    calibration: Dict[str, float] = SCORE_CALIBRATION,
) -> int:
    raw_total = (
        weights["PI"] * p_score * i_score
        + weights["P"] * p_score
        + weights["I"] * i_score
        + weights["D"] * d_score
    )
    raw_score = 100 * raw_total
    calibrated = raw_score * calibration["scale"] + calibration["bias"]
    return min(100, max(0, round(calibrated)))


def _score_from_hits(level_hits: Dict[int, List[str]]) -> Tuple[float, List[str], int]:
    if not level_hits:
        return 1.0, [], 0

    strongest_level = max(level_hits)
    strongest_hits = level_hits[strongest_level]
    total_hits = sum(len(items) for items in level_hits.values())
    level_diversity = len(level_hits)

    bonus = min(0.45, 0.10 * max(0, total_hits - 1) + 0.05 * max(0, level_diversity - 1))
    continuous_score = min(5.0, strongest_level + bonus)
    evidence = strongest_hits[:]
    return continuous_score, evidence, total_hits


def _collect_indicator_signal(indicator_key: str, text: str) -> Tuple[float, List[str], int]:
    normalized_text = _normalize_for_match(text)
    level_hits: Dict[int, List[str]] = {}
    for level, keywords in PATTERN_RULES[indicator_key].items():
        hits = _find_keywords(text, keywords, normalized_text)
        if hits:
            level_hits[level] = list(dict.fromkeys(hits))
    return _score_from_hits(level_hits)


def _prior_weight(evidence_count: int) -> float:
    if evidence_count >= 2:
        return PRIOR_BLEND_WEIGHTS[2]
    return PRIOR_BLEND_WEIGHTS[evidence_count]


def _apply_case_type_prior(
    case_type_names: Sequence[str],
    indicator_key: str,
    text_score: float,
    evidence_count: int,
    evidence: List[str],
) -> Tuple[float, List[str]]:
    prior_candidates = [(name, CASE_TYPE_HINTS[name][indicator_key]) for name in case_type_names if indicator_key in CASE_TYPE_HINTS[name]]
    if not prior_candidates:
        return text_score, evidence

    prior_name, prior_score = max(prior_candidates, key=lambda item: item[1])
    weight = _prior_weight(evidence_count)
    blended = round((1 - weight) * text_score + weight * float(prior_score), 4)

    if not evidence or prior_score > text_score:
        evidence = evidence + [f"案件类型先验:{prior_name}"]
    return blended, evidence


def _dimension_score(dimension: str, indicator_results: Dict[str, RiskIndicatorResult]) -> float:
    total = 0.0
    for key, weight in SECONDARY_WEIGHTS[dimension].items():
        total += weight * indicator_results[key].normalized_score
    return round(total, 4)


def _trigger_flags(text: str) -> List[str]:
    normalized_text = _normalize_for_match(text)
    flags = [flag for flag, keywords in TRIGGER_PATTERNS.items() if _find_keywords(text, keywords, normalized_text)]
    return flags


def _apply_level_floor(score: int, trigger_flags: Sequence[str]) -> int:
    red_flag_hits = sum(1 for flag in trigger_flags if flag in RED_FLAG_L3)
    has_l4_flag = any(flag in RED_FLAG_L4 for flag in trigger_flags)
    bounded = min(100, max(0, score))
    if has_l4_flag or red_flag_hits >= 2:
        return max(bounded, 81)
    if red_flag_hits >= 1:
        return max(bounded, 56)
    return bounded


def _top_indicators_for_dimension(
    dimension: str,
    indicator_results: Dict[str, RiskIndicatorResult],
) -> List[RiskIndicatorResult]:
    items = [value for value in indicator_results.values() if value.dimension == dimension]
    items.sort(key=lambda item: (item.continuous_score, item.normalized_score, len(item.evidence)), reverse=True)
    top_items = [item for item in items if item.raw_score >= 4][:2]
    if not top_items:
        top_items = items[:2]
    return top_items


def _reason_segment(dimension: str, indicator_results: Dict[str, RiskIndicatorResult]) -> str:
    segments: List[str] = []
    for item in _top_indicators_for_dimension(dimension, indicator_results):
        details = "、".join(item.evidence[:3]) if item.evidence else item.anchor_text
        segments.append(f"{item.label}={item.raw_score}({details})")
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
    return f"P={scores['P']}，I={scores['I']}，D={scores['D']}"


def _build_resident_view(assessment: RiskAssessment) -> Dict[str, object]:
    trigger_text = "无"
    if assessment.trigger_flags:
        trigger_text = "、".join(_trigger_flag_text(flag) for flag in assessment.trigger_flags)
    return {
        "结论": f"{_level_label(assessment.risk_level)}（{assessment.risk_level}）",
        "风险分数": assessment.risk_score,
        "P_I_D_分数": _pid_score_100(assessment),
        "P_I_D_文本": _pid_score_text(assessment),
        "建议": assessment.recommendation,
        "主要原因": assessment.reason_text,
        "触发规则": trigger_text,
    }


def _build_management_view(assessment: RiskAssessment) -> Dict[str, object]:
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
                "continuous_score": indicator.continuous_score,
                "normalized_score": indicator.normalized_score,
                "weighted_component": round(weight * indicator.normalized_score, 4),
                "evidence": indicator.evidence,
            }
        )
    for dimension in indicator_breakdown:
        indicator_breakdown[dimension].sort(key=lambda item: item["weighted_component"], reverse=True)

    return {
        "formula": "score = clamp(round((100 * (0.10 * P * I + 0.30 * P + 0.40 * I + 0.20 * D)) * 2.0 + 20))",
        "formula_weights": FORMULA_WEIGHTS,
        "score_calibration": SCORE_CALIBRATION,
        "thresholds": LEVEL_THRESHOLDS,
        "dimension_scores": {"P": assessment.P, "I": assessment.I, "D": assessment.D},
        "dimension_scores_100": _pid_score_100(assessment),
        "trigger_flags": assessment.trigger_flags,
        "indicator_breakdown": indicator_breakdown,
    }


def _trigger_flag_text(flag: str) -> str:
    labels = {
        "injury_or_violence": "存在人身伤害或暴力风险",
        "public_safety_hazard": "存在公共安全隐患",
        "gathering_or_public_opinion": "存在聚集或舆情扩散风险",
        "judicial_or_police": "已涉及公安或司法介入",
    }
    return labels.get(flag, flag)


def _render_resident_result_text(assessment: RiskAssessment) -> str:
    trigger_text = "未触发"
    if assessment.trigger_flags:
        trigger_text = "；".join(_trigger_flag_text(flag) for flag in assessment.trigger_flags)
    return (
        f"【风险结论】{_level_label(assessment.risk_level)}（{assessment.risk_level}），综合风险分 {assessment.risk_score} 分。\n"
        f"【指标分布】{_pid_score_text(assessment)}。\n"
        f"【主要原因】{assessment.reason_text}\n"
        f"【建议处置】{assessment.recommendation}\n"
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
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None)
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
        normalized[key] = [1 / len(values)] * len(values) if total == 0 else [value / total for value in values]

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
    searchable_text = "\n".join(part for part in (parsed["title"], parsed["description"]) if part).strip() or parsed["raw_text"]
    matched_case_types = _matched_case_type_names(case_type)
    indicator_results: Dict[str, RiskIndicatorResult] = {}

    for indicator_key in INDICATOR_LABELS:
        text_score, evidence, evidence_count = _collect_indicator_signal(indicator_key, searchable_text)
        final_score, evidence = _apply_case_type_prior(matched_case_types, indicator_key, text_score, evidence_count, evidence)
        raw_score = band_indicator(final_score)
        indicator_results[indicator_key] = RiskIndicatorResult(
            key=indicator_key,
            label=INDICATOR_LABELS[indicator_key],
            dimension=INDICATOR_DIMENSIONS[indicator_key],
            raw_score=raw_score,
            continuous_score=round(final_score, 4),
            normalized_score=normalize_indicator(final_score),
            anchor_text=ANCHOR_TEXT[indicator_key][raw_score],
            evidence=evidence,
        )

    p_score = _dimension_score("P", indicator_results)
    i_score = _dimension_score("I", indicator_results)
    d_score = _dimension_score("D", indicator_results)

    trigger_flags = _trigger_flags(searchable_text)
    final_score = _apply_level_floor(score_formula(p_score, i_score, d_score), trigger_flags)
    level = classify_level(final_score)

    trigger_text = "无" if not trigger_flags else "、".join(_trigger_flag_text(flag) for flag in trigger_flags)
    reason_text = (
        f"P高在{_reason_segment('P', indicator_results)}；"
        f"I高在{_reason_segment('I', indicator_results)}；"
        f"D高在{_reason_segment('D', indicator_results)}；"
        f"触发规则:{trigger_text}。"
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
    parser.add_argument("--text", help="Full case text with optional labels such as 案件类型/案情。")
    parser.add_argument("--input-file", help="Path to a UTF-8 text file containing one case.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    text = args.text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    elif not text:
        print("请输入案件文本，可多行输入，空行结束：")
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
