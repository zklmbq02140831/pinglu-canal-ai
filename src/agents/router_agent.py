# -*- coding: utf-8 -*-
"""
router_agent.py · 意图路由（两阶段：关键词快路径 → LLM 慢路径兜底）
============================================================
route(question: str) -> {"intent": "spatial"|"policy"|"unknown",
                        "confidence": float, "reason": str}

快路径（零 LLM 调用）：
  spatial 信号词命中 → intent=spatial, confidence=0.95
  policy  信号词命中 → intent=policy,  confidence=0.95
  同时命中两类 → policy 优先（稳定策略：政策库覆盖面更广）
慢路径（关键词全空）：
  调智谱 GLM-4-Flash 做分类, temperature=0, max_tokens=10
  超时/异常/解析失败 → 兜底 intent="policy"

无新依赖，复用 langchain_openai.ChatOpenAI（与 index_agent.py 同 base_url）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# =========================================================================
# 1. 关键词库
# =========================================================================
SPATIAL_KEYWORDS = [
    # —— 坐标/几何类 ——
    "坐标", "经纬度", "距离", "半径", "缓冲区", "点位",
    "桩号", "沿", "沿线", "沿线5公里", "沿线10公里",
    # —— GIS/QGIS 工具类 ——
    "QGIS", "GIS", "h3", "hex", "网格", "网格化",
    # —— 指数/排名/冷点类（IndexAgent 核心能力）——
    "指数", "综合指数", "composite", "排名", "冷点", "凹槽", "活力",
    "top", "grid", "poi密度", "路网密度", "坡度",
    # —— 纯事实查询词（追加，锚定 IndexAgent canal_facts 查询通道）——
    "起点", "终点", "全长", "开工", "建成", "枢纽", "马道", "企石", "青年",
    "什么时候", "何时", "开工时间", "建成时间", "动工", "完工",
]

POLICY_KEYWORDS = [
    # —— 政策文档类 ——
    "政策", "规划", "意见稿", "文件", "印发", "实施方案", "实施办法",
    # —— 扶持/补贴类 ——
    "补贴", "退税", "扶持", "支持政策", "优惠", "税收",
    # —— 规划期类 ——
    "十五五", "十四五", "十三五", "陆海新通道", "先进制造业",
    # —— 经济类 ——
    "带动作用", "高质量发展", "向海经济", "产业支持",
    # —— 影响/作用类（Day20追加：防止"建成后最大改变"类问题被spatial快路径截胡）——
    "改变", "影响", "作用", "意义", "带动", "航运", "出海", "物流", "经济",
]

# 大小写不敏感处理
_SPATIAL_LOWER = [k.lower() for k in SPATIAL_KEYWORDS]
_POLICY_LOWER = [k.lower() for k in POLICY_KEYWORDS]


# =========================================================================
# 2. Key 读取（多路径兜底，与项目其它 Agent 风格一致）
# =========================================================================
def _read_zhipu_key() -> str:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / ".env",
        Path(r"D:\yunhe\.env"),
        Path(r"d:\112\yunhe\.env"),
    ]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("ZHIPU_API_KEY", "ZHIPUAI_API_KEY") and v.strip():
                    return v.strip().strip("'\"")
    return os.environ.get("ZHIPU_API_KEY") or os.environ.get("ZHIPUAI_API_KEY") or ""


# =========================================================================
# 3. LLM 分类（慢路径）
# =========================================================================
LLM_PROMPT = """你是意图路由器。根据用户问题判断属于哪一类，只输出以下三选一的单词：
- spatial: 空间/地理/网格/指数/距离/坐标类问题
- policy: 政策/法规/规划/补贴/扶持类问题
- unknown: 与平陆运河无关的问题

只输出一个单词，不要加任何其他文字、标点或解释。"""

_ALLOWED_INTENTS = {"spatial", "policy", "unknown"}


def _llm_classify(question: str) -> Optional[str]:
    """调 GLM-4-Flash 做意图分类，失败返回 None。"""
    try:
        from langchain_openai import ChatOpenAI
        zhipu_key = _read_zhipu_key()
        if not zhipu_key:
            return None
        llm = ChatOpenAI(
            model="glm-4-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=zhipu_key,
            temperature=0,
            max_tokens=10,
        )
        resp = llm.invoke([
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": question},
        ])
        text = (resp.content or "").strip().lower()
        # 取第一个匹配的关键词（容忍 LLM 多输出文字的情况）
        for intent in _ALLOWED_INTENTS:
            if intent in text:
                return intent
        return None
    except Exception:
        return None


# =========================================================================
# 4. 对外主入口：route()
# =========================================================================
def route(question: str) -> dict:
    """
    两阶段意图路由。
    Returns:
        {"intent": "spatial"|"policy"|"unknown",
         "confidence": float, "reason": str}
    """
    q = question.strip() if question else ""
    q_lower = q.lower()

    # —— 阶段 1：关键词快路径 ——
    spatial_hits = [k for k in _SPATIAL_LOWER if k in q_lower]
    policy_hits = [k for k in _POLICY_LOWER if k in q_lower]

    if spatial_hits and not policy_hits:
        return {
            "intent": "spatial",
            "confidence": 0.95,
            "reason": f"关键词快路径命中 spatial: {spatial_hits}",
        }

    if policy_hits and not spatial_hits:
        return {
            "intent": "policy",
            "confidence": 0.95,
            "reason": f"关键词快路径命中 policy: {policy_hits}",
        }

    if spatial_hits and policy_hits:
        # 混合命中 → policy 优先（稳定策略：政策库覆盖面更广）
        return {
            "intent": "policy",
            "confidence": 0.90,
            "reason": (
                f"关键词同时命中 spatial={spatial_hits} 和 policy={policy_hits}，"
                f"按稳定优先级策略 → policy"
            ),
        }

    # —— 阶段 2：LLM 慢路径（关键词全空）——
    llm_result = _llm_classify(q)
    if llm_result is not None:
        return {
            "intent": llm_result,
            "confidence": 0.70,
            "reason": f"关键词未命中，LLM(GLM-4-Flash) 判定 → {llm_result}",
        }

    # —— 兜底：LLM 也不可用 → policy（政策库更完整）——
    return {
        "intent": "policy",
        "confidence": 0.50,
        "reason": "关键词未命中 + LLM 不可用/解析失败，兜底返回 policy（政策库更完整）",
    }


# =========================================================================
# 5. CLI
# =========================================================================
if __name__ == "__main__":
    import sys, json
    q = sys.argv[1] if len(sys.argv) > 1 else "运河沿线的产业扶持政策有哪些"
    print(f"\n🗣️  问题：{q}\n")
    r = route(q)
    print(json.dumps(r, ensure_ascii=False, indent=2))
