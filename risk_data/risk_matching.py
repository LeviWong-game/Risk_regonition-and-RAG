from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CASE_TYPE_HINTS: Dict[str, Dict[str, int]] = {
    "劳动争议": {
        "party_complexity": 2,
        "economic_livelihood_loss": 4,
        "cross_department_difficulty": 3,
        "vulnerable_urgency": 3,
    },
    "环境污染": {
        "impact_scope": 3,
        "safety_health_harm": 3,
        "public_order_opinion": 3,
        "cross_department_difficulty": 3,
    },
    "婚姻家庭": {"party_complexity": 2, "vulnerable_urgency": 3},
    "婚姻家庭纠纷": {"party_complexity": 2, "vulnerable_urgency": 3},
    "物业纠纷": {"impact_scope": 3, "public_order_opinion": 2},
    "物业管理纠纷": {"impact_scope": 3, "public_order_opinion": 2},
    "邻里纠纷": {"impact_scope": 2, "public_order_opinion": 2},
    "房屋租赁": {"legal_factual_complexity": 3},
    "房屋租赁纠纷": {"legal_factual_complexity": 3},
    "土地权属": {"legal_factual_complexity": 4, "party_complexity": 3},
    "土地权属纠纷": {"legal_factual_complexity": 4, "party_complexity": 3},
    "消费维权": {"economic_livelihood_loss": 2},
    "安全隐患": {"safety_health_harm": 4, "public_order_opinion": 3},
    "人身损害": {"safety_health_harm": 4, "economic_livelihood_loss": 3},
    "人身损害纠纷": {"safety_health_harm": 4, "economic_livelihood_loss": 3},
}

PATTERN_RULES: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "conflict_intensity": {
        2: ("纠纷", "矛盾", "投诉", "不满", "争议"),
        3: ("争执", "对峙", "拒绝", "推诿", "僵持", "强烈不满", "苦不堪言"),
        4: ("激烈争执", "威胁", "围堵", "报警", "过激", "拉横幅", "泼水"),
        5: ("肢体冲突", "殴打", "持刀", "砍伤", "报复", "自杀", "爆炸"),
    },
    "persistence_recurrence": {
        2: ("再次", "反复", "持续", "迟迟"),
        3: ("多次反映", "长期", "屡次", "一直", "反复投诉", "持续投诉", "多日"),
        4: ("多轮调解失败", "久拖未决", "长期未解决", "屡调未果", "长期积累", "多次催讨无果"),
        5: ("多年", "持续升级", "反复升级", "群体性积累"),
    },
    "party_complexity": {
        2: ("双方", "业主与物业", "房东与租客", "夫妻双方"),
        3: ("多方", "多户", "村民", "业主", "商户", "住户"),
        4: ("群体", "多名", "多人", "村集体", "工友", "居民代表"),
        5: ("数十人", "上百人", "大批", "涉众", "集体"),
    },
    "escalation_signals": {
        2: ("情绪激动", "强烈不满", "拒不配合"),
        3: ("报警", "发帖", "曝光", "维权", "拉横幅"),
        4: ("聚集", "围堵", "上访", "网传", "舆情", "直播"),
        5: ("扬言", "报复", "火灾", "爆炸", "群体性事件", "极端"),
    },
    "impact_scope": {
        2: ("家庭", "邻里", "两户", "双方"),
        3: ("楼栋", "小区", "单位", "商铺", "村民", "住户", "居民"),
        4: ("社区", "村集体", "多户", "多名", "多位"),
        5: ("跨区域", "全村", "全小区", "大范围", "大批"),
    },
    "safety_health_harm": {
        2: ("扰民", "噪音", "异味", "轻微受伤"),
        3: ("污染", "受伤", "骨折", "消防隐患", "停电", "停水", "油烟", "污水"),
        4: ("工伤", "起火", "火灾隐患", "严重污染", "住院", "踩踏", "坠落"),
        5: ("爆炸", "重伤", "死亡", "中毒", "坍塌"),
    },
    "economic_livelihood_loss": {
        2: ("退款", "押金", "租金", "停车费", "物业费", "赔偿"),
        3: ("误工费", "护理费", "停业", "停产", "生活受影响", "收益分配"),
        4: ("工资", "欠薪", "医疗费", "急需赔偿", "家庭困难", "抚养费"),
        5: ("生计", "无法生活", "全部积蓄", "重大损失", "断供"),
    },
    "public_order_opinion": {
        2: ("社区和谐", "邻里关系", "扰民"),
        3: ("持续投诉", "多人投诉", "公共区域", "楼道", "占道", "苦不堪言"),
        4: ("围堵", "聚集", "舆情", "上访", "网络传播"),
        5: ("大规模舆情", "群体性事件", "秩序失控", "交通阻断"),
    },
    "legal_factual_complexity": {
        2: ("责任", "争议", "协商"),
        3: ("合同", "程序", "证据", "条款", "收费依据", "认定"),
        4: ("权属", "继承", "工伤认定", "征地", "宅基地", "合法性"),
        5: ("司法鉴定", "刑民交叉", "行政争议", "多重合同链"),
    },
    "cross_department_difficulty": {
        2: ("物业", "网格员", "社区"),
        3: ("调解员", "街道", "居委会"),
        4: ("联合处理", "多部门", "政府部门", "消防", "住建"),
        5: ("公安", "法院", "应急", "司法强制", "刑事"),
    },
    "mediation_pressure": {
        2: ("尽快", "尽早", "催促"),
        3: ("急需", "限期", "马上", "立即", "苦不堪言"),
        4: ("多次催讨", "多次催办", "久拖未决", "调解失败"),
        5: ("随时可能", "一触即发", "立即处置", "失控"),
    },
    "vulnerable_urgency": {
        2: ("压力大", "情绪焦虑", "困难"),
        3: ("老人", "儿童", "未成年人", "孕妇", "残疾"),
        4: ("农民工", "家庭困难", "急需", "病人", "独居老人"),
        5: ("重病", "危重", "孤寡", "未成年受伤", "急救"),
    },
}

TRIGGER_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "injury_or_violence": ("肢体冲突", "殴打", "砍伤", "重伤", "死亡", "持刀", "报复", "工伤", "骨折", "住院", "坠落"),
    "public_safety_hazard": ("起火", "火灾", "爆炸", "消防隐患", "坍塌", "中毒", "严重污染", "私拉电线", "明火"),
    "gathering_or_public_opinion": ("聚集", "围堵", "上访", "舆情", "网络传播", "直播", "群体性事件"),
    "judicial_or_police": ("公安", "法院", "刑事", "司法强制", "报警"),
}

RED_FLAG_L3 = ("injury_or_violence", "public_safety_hazard", "gathering_or_public_opinion")
RED_FLAG_L4 = ("judicial_or_police",)

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "case_type": ("案件类型", "类型", "纠纷类型", "案件类别", "案由"),
    "title": ("案件标题", "案件"),
    "description": ("案情", "描述", "案情简介", "基本情况"),
}

KEYWORD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "协商无果": ("协商未果",),
    "屡调未果": ("多次调解未果", "调解不成"),
    "久拖未决": ("久拖不决",),
    "多次催讨无果": ("多次催讨未果",),
    "群体性事件": ("群体事件",),
    "聚集": ("聚众",),
    "上访": ("信访", "赴京上访"),
    "舆情": ("网络舆论", "网络发酵"),
    "欠薪": ("拖欠工资", "拖欠薪资"),
    "工伤": ("工伤事故",),
    "急需赔偿": ("急需补偿", "迫切索赔"),
    "报警": ("报了警", "已报警"),
    "维权": ("讨薪", "讨说法"),
}

CASE_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "劳动争议": ("劳动纠纷", "劳务纠纷", "劳动合同纠纷", "欠薪纠纷"),
    "环境污染": ("污染纠纷", "生态环境纠纷", "环保纠纷"),
    "物业纠纷": ("物业管理纠纷", "业主物业纠纷"),
    "房屋租赁": ("租赁纠纷", "房屋租赁纠纷"),
    "人身损害": ("人身伤害", "人身损害纠纷"),
}

_PUNCT_OR_SPACE_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_FUZZY_MIN_KEYWORD_LEN = 4


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _PUNCT_OR_SPACE_RE.sub("", normalized)


@lru_cache(maxsize=4096)
def _normalize_keyword_cached(text: str) -> str:
    return _normalize_for_match(text)


def _similarity_threshold(keyword_len: int) -> float:
    if keyword_len <= 4:
        return 0.92
    if keyword_len <= 6:
        return 0.88
    return 0.84


def _fuzzy_contains(normalized_text: str, normalized_keyword: str, keyword_chars: Optional[set] = None) -> bool:
    keyword_len = len(normalized_keyword)
    if keyword_len < _FUZZY_MIN_KEYWORD_LEN:
        return False
    if len(normalized_text) < max(2, keyword_len - 1):
        return False

    threshold = _similarity_threshold(keyword_len)
    if keyword_chars is None:
        keyword_chars = set(normalized_keyword)
    min_overlap = 0.6 if keyword_len <= 6 else 0.5
    min_window = max(2, keyword_len - 1)
    max_window = min(len(normalized_text), keyword_len + 1)
    for window_len in range(min_window, max_window + 1):
        for start in range(len(normalized_text) - window_len + 1):
            window = normalized_text[start:start + window_len]
            overlap_ratio = len(keyword_chars & set(window)) / max(1, len(keyword_chars))
            if overlap_ratio < min_overlap:
                continue
            if SequenceMatcher(None, window, normalized_keyword).ratio() >= threshold:
                return True
    return False


@lru_cache(maxsize=4096)
def _expand_keyword_variants(keyword: str) -> Tuple[str, ...]:
    variants = [keyword]
    variants.extend(KEYWORD_ALIASES.get(keyword, ()))
    deduplicated = dict.fromkeys(item for item in variants if item)
    return tuple(deduplicated)


@lru_cache(maxsize=4096)
def _keyword_variant_profiles(keyword: str) -> Tuple[Tuple[str, str, frozenset], ...]:
    profiles: List[Tuple[str, str, frozenset]] = []
    for variant in _expand_keyword_variants(keyword):
        normalized_variant = _normalize_keyword_cached(variant)
        if not normalized_variant:
            continue
        profiles.append((variant, normalized_variant, frozenset(normalized_variant)))
    return tuple(profiles)


def _keyword_matches(
    text: str,
    keyword: str,
    normalized_text: Optional[str] = None,
    normalized_text_chars: Optional[set] = None,
) -> bool:
    if not keyword:
        return False
    if normalized_text is None:
        normalized_text = _normalize_for_match(text)
    if normalized_text_chars is None:
        normalized_text_chars = set(normalized_text)

    for variant, normalized_variant, variant_chars in _keyword_variant_profiles(keyword):
        if variant in text:
            return True
        if normalized_variant in normalized_text:
            return True
        if not (variant_chars & normalized_text_chars):
            continue
        if _fuzzy_contains(normalized_text, normalized_variant, set(variant_chars)):
            return True
    return False


def _case_type_matches(case_type: str, case_type_hint: str, normalized_case_type: Optional[str] = None) -> bool:
    if not case_type_hint:
        return False
    if normalized_case_type is None:
        normalized_case_type = _normalize_for_match(case_type)
    if _keyword_matches(case_type, case_type_hint, normalized_case_type):
        return True
    for alias in CASE_TYPE_ALIASES.get(case_type_hint, ()):
        if _keyword_matches(case_type, alias, normalized_case_type):
            return True
    return False


@lru_cache(maxsize=1024)
def _matched_case_type_names(case_type: str) -> Tuple[str, ...]:
    normalized_case_type = _normalize_for_match(case_type)
    matched: List[str] = []
    for name in CASE_TYPE_HINTS:
        if _case_type_matches(case_type, name, normalized_case_type):
            matched.append(name)
    return tuple(matched)


def _find_keywords(text: str, keywords: Sequence[str], normalized_text: Optional[str] = None) -> List[str]:
    if normalized_text is None:
        normalized_text = _normalize_for_match(text)
    normalized_text_chars = set(normalized_text)
    return [kw for kw in keywords if _keyword_matches(text, kw, normalized_text, normalized_text_chars)]


def _extract_field(text: str, labels: Iterable[str]) -> str:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[:：]\s*(.+)", text)
        if match:
            return match.group(1).strip()
    return ""


def parse_case_text(text: str) -> Dict[str, str]:
    case_type = _extract_field(text, FIELD_ALIASES["case_type"])
    title = _extract_field(text, FIELD_ALIASES["title"])
    description = _extract_field(text, FIELD_ALIASES["description"])
    if not description:
        description = text.strip()
    return {
        "case_type": case_type,
        "title": title,
        "description": description,
        "raw_text": text.strip(),
    }
