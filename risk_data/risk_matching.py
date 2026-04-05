from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CASE_TYPE_HINTS: Dict[str, Dict[str, int]] = {
    "劳动争议": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "economic_livelihood_loss": 4,
        "cross_department_difficulty": 3,
        "vulnerable_urgency": 3,
    },
    "工伤赔偿纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "safety_health_harm": 4,
        "economic_livelihood_loss": 4,
        "vulnerable_urgency": 4,
    },
    "邻里纠纷": {
        "conflict_intensity": 2,
        "impact_scope": 2,
        "public_order_opinion": 2,
    },
    "物业纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "impact_scope": 3,
        "public_order_opinion": 3,
        "cross_department_difficulty": 3,
    },
    "环境污染": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "impact_scope": 3,
        "safety_health_harm": 4,
        "public_order_opinion": 3,
        "cross_department_difficulty": 3,
    },
    "婚姻家庭": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "impact_scope": 2,
        "vulnerable_urgency": 3,
    },
    "土地权属纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "legal_factual_complexity": 4,
        "party_complexity": 3,
        "impact_scope": 3,
        "cross_department_difficulty": 4,
    },
    "征地拆迁": {
        "conflict_intensity": 3,
        "persistence_recurrence": 3,
        "impact_scope": 4,
        "party_complexity": 4,
        "public_order_opinion": 4,
        "cross_department_difficulty": 4,
    },
    "合同纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "legal_factual_complexity": 3,
        "economic_livelihood_loss": 3,
    },
    "房屋租赁纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 2,
        "legal_factual_complexity": 3,
        "economic_livelihood_loss": 3,
    },
    "欠薪纠纷": {
        "conflict_intensity": 2,
        "persistence_recurrence": 3,
        "economic_livelihood_loss": 4,
        "vulnerable_urgency": 3,
        "mediation_pressure": 3,
    },
}

PATTERN_RULES: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "conflict_intensity": {
        2: ("不满", "投诉", "争议", "纠纷", "口角", "矛盾", "摩擦"),
        3: ("争吵", "对峙", "冲突", "拒绝配合", "情绪激动", "争执", "逼迫"),
        4: ("激烈争执", "威胁", "围堵", "堵门", "报警", "拉横幅", "泼水", "停电措施"),
        5: ("殴打", "动手", "持刀", "砍伤", "打伤", "暴力", "极端行为"),
    },
    "persistence_recurrence": {
        2: ("再次", "又来", "反复", "重复", "多日", "多次"),
        3: ("多次反映", "持续", "长期", "一直", "多次投诉", "久拖未决", "无果", "催讨", "未支付"),
        4: ("多轮调解失败", "屡次调解无果", "长期未解决", "反复升级", "多次协商无果"),
        5: ("积案", "多年未决", "持续恶化", "长期积累"),
    },
    "party_complexity": {
        2: ("双方", "业主与物业", "夫妻双方", "邻居双方"),
        3: ("多方", "村民", "业主", "商户", "施工方", "开发商", "包工头"),
        4: ("多户", "村集体", "业主委员会", "单位与居民", "组织参与", "多人参与"),
        5: ("群体", "大批", "数十人", "社会人员参与", "多人聚集"),
    },
    "escalation_signals": {
        2: ("扬言", "拒不接受", "言语过激", "录视频维权"),
        3: ("报警", "发帖", "投诉上访", "拉横幅", "媒体曝光"),
        4: ("聚集", "围堵", "堵门", "堵路", "舆情扩散", "网传"),
        5: ("冲击", "打砸", "自杀威胁", "报复", "群体性事件"),
    },
    "impact_scope": {
        2: ("家庭", "邻里", "两户", "楼上楼下", "住户"),
        3: ("小区", "楼栋", "村组", "单位", "多户居民", "商户"),
        4: ("社区", "村民", "整栋楼", "周边商户", "多人受影响"),
        5: ("跨区域", "大范围", "全小区", "全村", "全网关注"),
    },
    "safety_health_harm": {
        2: ("扰民", "异味", "噪音", "轻微受伤", "轻伤", "惊吓"),
        3: ("污染", "受伤", "骨折", "住院", "健康风险", "中毒风险", "苦不堪言"),
        4: ("工伤", "流血", "重大隐患", "火灾隐患", "人身伤害", "摔落"),
        5: ("重伤", "死亡", "爆炸", "火灾", "群死群伤", "生命危险"),
    },
    "economic_livelihood_loss": {
        2: ("赔偿争议", "费用争议", "损失", "补偿", "抚养费"),
        3: ("工资", "货款", "赔偿款", "停工", "收入受影响", "经济损失", "涨租", "租金"),
        4: ("欠薪", "拖欠工资", "无法生活", "生计困难", "急需赔偿", "无力支付"),
        5: ("巨额损失", "破产", "断供", "失去生活来源", "重大财产损失"),
    },
    "public_order_opinion": {
        2: ("邻里关系紧张", "多次投诉", "社区和谐"),
        3: ("社区秩序", "持续投诉", "舆情", "网络传播", "围观", "住户"),
        4: ("聚集围观", "围堵", "堵路", "媒体关注", "舆情扩散"),
        5: ("冲击秩序", "群体性事件", "广泛传播", "持续发酵"),
    },
    "legal_factual_complexity": {
        2: ("责任认定", "证据不清", "存在争议", "抚养费"),
        3: ("合同", "协议", "程序", "举证", "鉴定", "赔偿标准", "租赁"),
        4: ("权属", "继承", "工伤认定", "土地流转", "历史遗留", "拆迁补偿"),
        5: ("诉讼程序复杂", "多份证据冲突", "多法律关系交织"),
    },
    "cross_department_difficulty": {
        2: ("社区协调", "物业处理", "网格员处理"),
        3: ("街道介入", "调解员介入", "多方协调", "环保投诉", "人社协调"),
        4: ("联合处置", "多部门", "住建", "环保", "人社", "信访", "部门联合处理"),
        5: ("公安介入", "法院介入", "应急处置", "司法程序"),
    },
    "mediation_pressure": {
        2: ("尽快处理", "尽快解决", "催促"),
        3: ("急需", "限期", "短期内处理", "连续投诉", "申请调解"),
        4: ("多次催办", "调解失败", "久拖不决", "反复催办"),
        5: ("立即处理", "情况紧急", "随时失控", "拖延风险极高"),
    },
    "vulnerable_urgency": {
        2: ("生活困难", "情绪压力", "家庭压力"),
        3: ("老人", "儿童", "孕妇", "残疾", "学生", "孩子"),
        4: ("工伤", "重病", "低保", "家庭困难", "失业", "急需治疗", "独自抚养"),
        5: ("紧急救助", "未成年人受伤", "老人独居", "孕妇受伤", "危及基本生存"),
    },
}

TRIGGER_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "injury_or_violence": ("殴打", "动手", "打伤", "砍伤", "流血", "重伤", "暴力", "生命危险"),
    "public_safety_hazard": ("火灾", "爆炸", "燃气泄漏", "重大隐患", "坍塌", "中毒"),
    "gathering_or_public_opinion": ("聚集", "围堵", "堵路", "拉横幅", "舆情", "媒体关注", "群体性事件"),
    "judicial_or_police": ("公安介入", "警方介入", "法院受理", "立案", "司法程序"),
}

RED_FLAG_L3 = ("injury_or_violence", "public_safety_hazard", "gathering_or_public_opinion")
RED_FLAG_L4 = ("judicial_or_police",)

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "case_type": ("案件类型", "类型", "案由"),
    "title": ("案件标题", "标题", "案件名称", "案件"),
    "description": ("案件描述", "案情", "描述", "基本情况", "情况", "案情简介"),
}

KEYWORD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "报警": ("报案", "报警求助"),
    "拉横幅": ("打横幅",),
    "聚集": ("聚众", "多人聚集"),
    "围堵": ("围堵单位", "围堵大门"),
    "欠薪": ("拖欠工资", "讨薪"),
    "工伤": ("工地受伤", "施工受伤"),
    "舆情": ("网络舆情", "网上传播"),
    "法院介入": ("法院受理", "起诉到法院"),
    "公安介入": ("警方介入", "派出所介入"),
}

CASE_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "劳动争议": ("劳资纠纷", "劳动纠纷"),
    "工伤赔偿纠纷": ("工伤纠纷", "工伤赔偿"),
    "邻里纠纷": ("邻居纠纷", "邻里矛盾"),
    "物业纠纷": ("物业费纠纷", "业主物业纠纷"),
    "环境污染": ("环境纠纷", "污染纠纷"),
    "婚姻家庭": ("婚姻家庭纠纷", "家庭纠纷"),
    "土地权属纠纷": ("土地纠纷", "权属纠纷"),
    "征地拆迁": ("拆迁纠纷", "征迁纠纷"),
    "合同纠纷": ("合同争议",),
    "房屋租赁纠纷": ("租赁纠纷", "商铺租赁纠纷", "房屋租赁"),
    "欠薪纠纷": ("讨薪纠纷", "工资纠纷"),
}

NEGATION_WORDS: Tuple[str, ...] = ("未", "没有", "无", "并未", "尚未", "未曾", "未发生", "未出现")

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


def _is_negated_match(text: str, keyword: str) -> bool:
    position = text.find(keyword)
    if position < 0:
        return False
    window = text[max(0, position - 6):position]
    return any(flag in window for flag in NEGATION_WORDS)


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
    return tuple(dict.fromkeys(item for item in variants if item))


@lru_cache(maxsize=4096)
def _keyword_variant_profiles(keyword: str) -> Tuple[Tuple[str, str, frozenset], ...]:
    profiles: List[Tuple[str, str, frozenset]] = []
    for variant in _expand_keyword_variants(keyword):
        normalized_variant = _normalize_keyword_cached(variant)
        if normalized_variant:
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
        if variant in text and not _is_negated_match(text, variant):
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
    return any(_keyword_matches(case_type, alias, normalized_case_type) for alias in CASE_TYPE_ALIASES.get(case_type_hint, ()))


@lru_cache(maxsize=1024)
def _matched_case_type_names(case_type: str) -> Tuple[str, ...]:
    normalized_case_type = _normalize_for_match(case_type)
    matched = [name for name in CASE_TYPE_HINTS if _case_type_matches(case_type, name, normalized_case_type)]
    return tuple(matched)


def _find_keywords(text: str, keywords: Sequence[str], normalized_text: Optional[str] = None) -> List[str]:
    if normalized_text is None:
        normalized_text = _normalize_for_match(text)
    normalized_text_chars = set(normalized_text)
    return [keyword for keyword in keywords if _keyword_matches(text, keyword, normalized_text, normalized_text_chars)]


def _extract_field(text: str, labels: Iterable[str]) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for label in labels:
            for separator in ("：", ":"):
                prefix = f"{label}{separator}"
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
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
