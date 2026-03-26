from __future__ import annotations

from typing import Dict, Tuple


LEVEL_THRESHOLDS: Tuple[Tuple[str, int, int], ...] = (
    ("L1", 0, 25),
    ("L2", 26, 50),
    ("L3", 51, 75),
    ("L4", 76, 100),
)

LEVEL_ACTIONS: Dict[str, str] = {
    "L1": "网格员/物业自行处理",
    "L2": "调解员介入调解",
    "L3": "政府部门联合处置",
    "L4": "法院/公安或应急联动处置",
}

# 一级指标（3 个维度） 分别对应：冲突性（P）、影响性（I）、复杂性（D）
PRIMARY_WEIGHTS: Dict[str, float] = {"P": 0.75, "I": 0.75, "D": 0.25}
# 二级指标（12 个具体指标）
SECONDARY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "P": {
        "conflict_intensity": 0.35,
        "persistence_recurrence": 0.25,
        "party_complexity": 0.20,
        "escalation_signals": 0.20,
    },
    "I": {
        "impact_scope": 0.20,
        "safety_health_harm": 0.35,
        "economic_livelihood_loss": 0.20,
        "public_order_opinion": 0.25,
    },
    "D": {
        "legal_factual_complexity": 0.25,
        "cross_department_difficulty": 0.35,
        "mediation_pressure": 0.20,
        "vulnerable_urgency": 0.20,
    },
}

INDICATOR_LABELS: Dict[str, str] = {
    "conflict_intensity": "冲突强度",
    "persistence_recurrence": "持续性/复发性",
    "party_complexity": "涉事主体复杂度",
    "escalation_signals": "升级触发信号",
    "impact_scope": "涉及范围",
    "safety_health_harm": "安全/健康损害",
    "economic_livelihood_loss": "经济/民生损失",
    "public_order_opinion": "公共秩序/舆情影响",
    "legal_factual_complexity": "法律与事实争议度",
    "cross_department_difficulty": "跨部门协调难度",
    "mediation_pressure": "调解资源与时效压力",
    "vulnerable_urgency": "脆弱群体/紧迫性",
}

INDICATOR_DIMENSIONS: Dict[str, str] = {
    "conflict_intensity": "P",
    "persistence_recurrence": "P",
    "party_complexity": "P",
    "escalation_signals": "P",
    "impact_scope": "I",
    "safety_health_harm": "I",
    "economic_livelihood_loss": "I",
    "public_order_opinion": "I",
    "legal_factual_complexity": "D",
    "cross_department_difficulty": "D",
    "mediation_pressure": "D",
    "vulnerable_urgency": "D",
}

ANCHOR_TEXT: Dict[str, Dict[int, str]] = {
    "conflict_intensity": {
        1: "普通投诉或一般分歧",
        2: "明显不满但仍以协商为主",
        3: "争执、对峙或一方持续拒绝配合",
        4: "激烈争执、威胁、围堵或报警",
        5: "肢体冲突、暴力威胁或极端行为",
    },
    "persistence_recurrence": {
        1: "首次发生或短时纠纷",
        2: "短期反复出现",
        3: "已多次反映或持续一段时间",
        4: "多轮调解失败或长期未解",
        5: "长期积累且持续升级",
    },
    "party_complexity": {
        1: "单一主体内部事务",
        2: "双方主体清晰",
        3: "多方主体参与",
        4: "涉众或组织化主体参与",
        5: "大范围群体、机构与第三方交织",
    },
    "escalation_signals": {
        1: "无升级信号",
        2: "情绪明显但可控",
        3: "存在报警、拉横幅、网络发帖等苗头",
        4: "存在聚集、围堵、威胁或舆情扩散",
        5: "现实暴力、高危隐患或群体性升级已出现",
    },
    "impact_scope": {
        1: "影响局限于个体",
        2: "影响家庭或相邻双方",
        3: "影响小区、楼栋、单位",
        4: "影响社区、村集体或较大人群",
        5: "影响跨区域或大范围涉众",
    },
    "safety_health_harm": {
        1: "无现实人身或环境风险",
        2: "有轻微扰民或轻微健康风险",
        3: "存在明显健康、污染或轻伤风险",
        4: "已发生伤害或存在重大安全隐患",
        5: "重伤、火灾、爆炸或严重公共安全风险",
    },
    "economic_livelihood_loss": {
        1: "损失较小，民生影响有限",
        2: "有一般经济争议",
        3: "损失较明显或影响正常生活",
        4: "赔偿、工资、生计压力明显",
        5: "重大经济损失或核心生计中断",
    },
    "public_order_opinion": {
        1: "对公共秩序无明显影响",
        2: "可能影响邻里关系",
        3: "影响社区秩序或引发持续投诉",
        4: "涉及聚集、扰民、围堵或舆情扩散",
        5: "对公共秩序造成明显冲击或持续舆情事件",
    },
    "legal_factual_complexity": {
        1: "事实简单、责任清晰",
        2: "存在一般争议点",
        3: "涉及合同、程序、证据等复杂事实",
        4: "涉及权属、继承、工伤认定等高复杂议题",
        5: "高度复杂，单一调解难以判明责任",
    },
    "cross_department_difficulty": {
        1: "单一主体可处理",
        2: "社区/物业/网格员可闭环",
        3: "需调解员或街道协同",
        4: "需多部门联合处置",
        5: "需公安、法院、应急等强制或专业介入",
    },
    "mediation_pressure": {
        1: "无明显时效压力",
        2: "需尽快沟通",
        3: "存在短期处置压力",
        4: "已出现多次催办或调解失败",
        5: "即刻处置压力大，拖延易失控",
    },
    "vulnerable_urgency": {
        1: "无脆弱群体或紧迫情况",
        2: "存在一般情绪或生活压力",
        3: "涉及老人、儿童、孕妇等脆弱主体之一",
        4: "涉及工伤、家庭困难、失业等明显紧迫情形",
        5: "脆弱群体叠加急迫救助需求",
    },
}
