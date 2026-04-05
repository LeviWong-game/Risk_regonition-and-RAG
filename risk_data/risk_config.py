from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


LEVEL_THRESHOLDS: Tuple[Tuple[str, int, int], ...] = (
    ("L1", 0, 26),
    ("L2", 27, 52),
    ("L3", 53, 80),
    ("L4", 81, 100),
)

LEVEL_ACTIONS: Dict[str, str] = {
    "L1": "网格员或物业自行处置，保留回访即可。",
    "L2": "由调解员介入，尽快组织协商并设置跟踪节点。",
    "L3": "启动街道或相关部门联合处置，提前做好稳控与预案。",
    "L4": "立即升级至公安、法院或应急体系联动处置。",
}

# 使用“乘法 + 加法”混合公式，既保留 P 和 I 的耦合关系，又降低纯乘法过于保守的问题。
FORMULA_WEIGHTS: Dict[str, float] = {
    "PI": 0.10,
    "P": 0.30,
    "I": 0.40,
    "D": 0.20,
}

SCORE_CALIBRATION: Dict[str, float] = {
    "scale": 2.0,
    "bias": 20.0,
}

# 文本证据越弱，案由先验权重越高；文本证据越强，先验只做微调。
PRIOR_BLEND_WEIGHTS: Dict[int, float] = {
    0: 0.55,
    1: 0.30,
    2: 0.18,
}

INDICATOR_SPECS: Dict[str, Dict[str, object]] = {
    "conflict_intensity": {
        "dimension": "P",
        "weight": 0.35,
        "label": "冲突强度",
        "anchors": {
            1: "普通投诉或一般分歧。",
            2: "存在明显不满，但仍以协商为主。",
            3: "已出现争执、对峙或一方持续拒绝配合。",
            4: "已出现激烈争吵、威胁、围堵或报警。",
            5: "已出现肢体冲突、暴力威胁或极端行为。",
        },
    },
    "persistence_recurrence": {
        "dimension": "P",
        "weight": 0.25,
        "label": "持续性/复发性",
        "anchors": {
            1: "首次发生或短时纠纷。",
            2: "短期内反复出现。",
            3: "已多次反映或持续一段时间。",
            4: "多轮调解失败或长期未解决。",
            5: "长期积累且持续升级。",
        },
    },
    "party_complexity": {
        "dimension": "P",
        "weight": 0.20,
        "label": "涉事主体复杂度",
        "anchors": {
            1: "单一主体内部事务。",
            2: "双方主体清晰。",
            3: "多方主体参与。",
            4: "涉众或组织化主体参与。",
            5: "大范围群体、机构与第三方交织。",
        },
    },
    "escalation_signals": {
        "dimension": "P",
        "weight": 0.20,
        "label": "升级触发信号",
        "anchors": {
            1: "无明显升级信号。",
            2: "情绪明显但仍可控。",
            3: "存在报警、拉横幅、网络发帖等苗头。",
            4: "出现聚集、围堵、威胁或舆情扩散。",
            5: "现实暴力、高危隐患或群体性升级已出现。",
        },
    },
    "impact_scope": {
        "dimension": "I",
        "weight": 0.20,
        "label": "影响范围",
        "anchors": {
            1: "影响局限于个体。",
            2: "影响家庭、邻里或相邻双方。",
            3: "影响楼栋、小区、单位或村组。",
            4: "影响社区、村集体或较多人群。",
            5: "影响跨区域或大范围公众。",
        },
    },
    "safety_health_harm": {
        "dimension": "I",
        "weight": 0.35,
        "label": "安全/健康损害",
        "anchors": {
            1: "无现实人身或环境风险。",
            2: "有轻微扰民或健康风险。",
            3: "存在明显健康、污染或轻伤风险。",
            4: "已发生伤害或存在重大安全隐患。",
            5: "涉及重伤、火灾、爆炸或严重公共安全风险。",
        },
    },
    "economic_livelihood_loss": {
        "dimension": "I",
        "weight": 0.20,
        "label": "经济/民生损失",
        "anchors": {
            1: "损失较小，对生活影响有限。",
            2: "存在一般经济争议。",
            3: "损失较明显或影响正常生活。",
            4: "赔偿、工资、生计压力明显。",
            5: "重大经济损失或核心生计中断。",
        },
    },
    "public_order_opinion": {
        "dimension": "I",
        "weight": 0.25,
        "label": "公共秩序/舆情影响",
        "anchors": {
            1: "对公共秩序无明显影响。",
            2: "可能影响邻里关系或局部秩序。",
            3: "影响社区秩序或引发持续投诉。",
            4: "涉及聚集、扰民、围堵或舆情扩散。",
            5: "对公共秩序造成明显冲击或持续性舆情事件。",
        },
    },
    "legal_factual_complexity": {
        "dimension": "D",
        "weight": 0.25,
        "label": "法律与事实争议度",
        "anchors": {
            1: "事实简单、责任清晰。",
            2: "存在一般争议点。",
            3: "涉及合同、程序、证据等复杂事实。",
            4: "涉及权属、继承、工伤认定等高复杂议题。",
            5: "高度复杂，单靠常规调解难以厘清责任。",
        },
    },
    "cross_department_difficulty": {
        "dimension": "D",
        "weight": 0.35,
        "label": "跨部门协调难度",
        "anchors": {
            1: "单一主体即可处置。",
            2: "社区、物业或网格员可闭环处理。",
            3: "需要调解员或街道协同。",
            4: "需要多部门联合处置。",
            5: "需要公安、法院、应急等强制或专业介入。",
        },
    },
    "mediation_pressure": {
        "dimension": "D",
        "weight": 0.20,
        "label": "调解资源与时效压力",
        "anchors": {
            1: "无明显时效压力。",
            2: "需要尽快沟通处理。",
            3: "存在短期处置压力。",
            4: "已多次催办或调解失败。",
            5: "处置时效极强，拖延易失控。",
        },
    },
    "vulnerable_urgency": {
        "dimension": "D",
        "weight": 0.20,
        "label": "弱势群体/紧迫性",
        "anchors": {
            1: "无弱势群体或紧迫情形。",
            2: "存在一般情绪或生活压力。",
            3: "涉及老人、儿童、孕妇等弱势主体之一。",
            4: "涉及工伤、家庭困难、失业等明显紧迫情形。",
            5: "弱势群体叠加且存在紧急救助需求。",
        },
    },
}

INDICATOR_LABELS: Dict[str, str] = {
    key: spec["label"]  # type: ignore[index]
    for key, spec in INDICATOR_SPECS.items()
}

INDICATOR_DIMENSIONS: Dict[str, str] = {
    key: spec["dimension"]  # type: ignore[index]
    for key, spec in INDICATOR_SPECS.items()
}

ANCHOR_TEXT: Dict[str, Dict[int, str]] = {
    key: spec["anchors"]  # type: ignore[index]
    for key, spec in INDICATOR_SPECS.items()
}

SECONDARY_WEIGHTS: Dict[str, Dict[str, float]] = defaultdict(dict)
for indicator_key, spec in INDICATOR_SPECS.items():
    dimension = spec["dimension"]  # type: ignore[index]
    weight = spec["weight"]  # type: ignore[index]
    SECONDARY_WEIGHTS[dimension][indicator_key] = weight
SECONDARY_WEIGHTS = dict(SECONDARY_WEIGHTS)

DIMENSION_INDICATORS: Dict[str, Tuple[str, ...]] = {
    dimension: tuple(weights.keys())
    for dimension, weights in SECONDARY_WEIGHTS.items()
}
