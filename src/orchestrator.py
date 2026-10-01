# -*- coding: utf-8 -*-
"""
orchestrator.py · 统一入口（Router + Agent 调度）
================================================
run(question: str) -> dict

流程：
  ① router_agent.route(question) → intent / confidence / reason
  ② spatial → index_agent.run()
  ③ policy  → policy_agent.run()
  ④ unknown → 固定提示（不调下游 Agent）

统一输出结构：
  {answer, sources, hex_ids, intent, router_reason, agent_flow, snapshot}

注：IndexAgent 返回 {answer, hex_ids, agent_flow, snapshot} 无 sources → 补 sources=[]
   PolicyAgent  返回 {answer, sources, agent_flow, snapshot} 无 hex_ids → 补 hex_ids=[]
   agent_flow 重写为 ["Router", "IndexAgent"|"PolicyAgent"]
"""
from __future__ import annotations

import json
from typing import Any

# —— 懒加载单例（模块级，避免每次 run 都 rebuild LangGraph agent）——
_index_agent = None
_policy_agent = None


def _get_index_agent():
    global _index_agent
    if _index_agent is None:
        from src.agents.index_agent import build_agent
        _index_agent = build_agent()
    return _index_agent


def _get_policy_agent():
    global _policy_agent
    if _policy_agent is None:
        from src.agents.policy_agent import build_agent
        _policy_agent = build_agent()
    return _policy_agent


# =========================================================================
# 主入口
# =========================================================================
def run(question: str) -> dict[str, Any]:
    """
    统一问答入口：路由 → 分发 → 返回标准化 JSON。
    """
    from src.agents.router_agent import route

    # ① 路由
    route_result = route(question)
    intent = route_result["intent"]
    router_reason = route_result["reason"]

    base: dict[str, Any] = {
        "intent": intent,
        "router_reason": router_reason,
    }

    # ② 分发
    if intent == "spatial":
        from src.agents import index_agent
        downstream = index_agent.run(question, agent=_get_index_agent())
        # 字段对齐：IndexAgent 无 sources → 补空列表
        base.update({
            "answer": downstream.get("answer", ""),
            "sources": [],
            "hex_ids": downstream.get("hex_ids", []),
            "coords": downstream.get("coords", []),
            "agent_flow": ["Router", "IndexAgent"],
            "snapshot": downstream.get("snapshot", "v2026-09"),
        })

    elif intent == "policy":
        from src.agents import policy_agent
        downstream = policy_agent.run(question, agent=_get_policy_agent())
        # 字段对齐：PolicyAgent 无 hex_ids → 补空列表
        base.update({
            "answer": downstream.get("answer", ""),
            "sources": downstream.get("sources", []),
            "hex_ids": [],
            "coords": [],
            "agent_flow": ["Router", "PolicyAgent"],
            "snapshot": downstream.get("snapshot", "v2026-09"),
        })

    else:  # unknown
        base.update({
            "answer": "该问题超出本系统范围,请咨询空间或政策相关问题",
            "sources": [],
            "hex_ids": [],
            "coords": [],
            "agent_flow": ["Router"],
            "snapshot": "v2026-09",
        })

    return base


# =========================================================================
# 资源清理（消除 QdrantClient.__del__ 噪音 + 释放 Agent 引用）
# =========================================================================
def cleanup():
    """显式关闭所有懒加载的底层 client。可多次调用。"""
    global _index_agent, _policy_agent
    # —— 关闭 policy_agent 模块级的 QdrantClient 单例 ——
    try:
        from src.agents import policy_agent
        if getattr(policy_agent, "_qdrant_client", None) is not None:
            policy_agent._qdrant_client.close()
            policy_agent._qdrant_client = None
    except Exception:
        pass
    # —— 释放 Agent 引用（让 GC 回收 LangGraph / LLM client）——
    _index_agent = None
    _policy_agent = None


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "运河沿线的产业扶持政策有哪些"
    print(f"\n🗣️  问题：{q}\n")
    r = run(q)
    print(json.dumps(r, ensure_ascii=False, indent=2))
