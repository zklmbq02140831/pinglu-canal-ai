# -*- coding: utf-8 -*-
"""
frontend/app.py · 平陆运河多 Agent 智能问答系统 · Streamlit 演示前端
=======================================================================
直连 src/orchestrator.run(question)，无额外依赖。
"""
from __future__ import annotations

import os
import sys

# —— 关键：把项目根目录加到 sys.path，否则 import src.* 会 ModuleNotFoundError ——
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402


# =========================================================================
# 资源缓存（避免每次 rerun 都重建 Qdrant / Agent 实例）
# orchestrator 内部已用模块级懒加载单例，这里的 cache_resource 防止
# Streamlit rerun 时重新 import 链的开销与潜在副作用。
# =========================================================================
@st.cache_resource
def _get_orchestrator():
    import src.orchestrator as orch
    return orch


# =========================================================================
# 页面配置
# =========================================================================
st.set_page_config(
    page_title="平陆运河多Agent智能问答系统",
    page_icon="🚢",
    layout="wide",
)


# =========================================================================
# 侧边栏
# =========================================================================
with st.sidebar:
    st.title("🚢 平陆运河")
    st.subheader("多 Agent 智能问答系统")

    st.divider()

    # snapshot 标识
    st.caption(f"📦 数据快照：**snapshot = v2026-09**")

    # 架构说明
    st.markdown(
        """
        **架构**：

        ```
        RouterAgent ─┬─→ IndexAgent（空间分析）
                     └─→ PolicyAgent（政策 RAG）
        ```
        """
    )

    st.divider()

    # 演示预置按钮区
    st.markdown("### 🎯 演示预置场景")

    _PRESETS = [
        ("空间题 · 坐标查询", "平陆运河起点坐标是什么"),
        ("空间题 · 乡镇检索", "检索沿线5公里内的乡镇"),
        ("政策题 · 经济带动", "平陆运河对广西经济的带动作用"),
        ("负样本 · 领域外", "跨境电商退税"),
        ("兜底题 · 无关", "今天天气怎么样"),
    ]

    for label, q in _PRESETS:
        if st.button(label, key=f"preset_{label}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()


# =========================================================================
# 主区 · Chat 交互
# =========================================================================
st.markdown(
    "<div style='text-align:center;color:#888'>您可以询问运河空间信息或政策相关内容</div>",
    unsafe_allow_html=True,
)

question = st.chat_input("请输入您的问题…")

# —— 预置按钮触发：从 session_state 取待处理问题 ——
if question is None and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    # 用户消息
    with st.chat_message("user"):
        st.markdown(question)

    # 路由 + 分析
    orch = _get_orchestrator()
    with st.spinner("正在路由并分析…"):
        result = orch.run(question)

    intent = result.get("intent", "unknown")
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    agent_flow = result.get("agent_flow", [])
    router_reason = result.get("router_reason", "")

    with st.chat_message("assistant"):
        # —— agent_flow 徽章 + router_reason ——
        if agent_flow:
            badges = " → ".join(
                f"**[{name}]**" if name == "Router" else f"[{name}]"
                for name in agent_flow
            )
            st.markdown(f"`{badges}`")
        if router_reason:
            st.markdown(
                f"<span style='color:#888;font-size:12px'>路由判定：{router_reason}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")  # 间距

        # —— 答案正文 ——
        st.markdown(answer)

        # —— intent=unknown 友好提示已在 answer 中给出；额外加强文案 ——
        if intent == "unknown":
            st.info("💡 该问题超出本系统范围，您可询问运河空间信息或政策相关内容")

        # —— sources 区（仅 policy intent 有值）——
        if sources:
            with st.expander(f"📌 政策依据（{len(sources)} 条）", expanded=False):
                for i, src in enumerate(sources, 1):
                    title = src.get("title", "未知文档")
                    score = src.get("score", 0)
                    snippet = src.get("snippet", src.get("content", ""))
                    st.markdown(
                        f"**{i}. {title}**  \n"
                        f"<span style='color:#888;font-size:12px'>相似度得分：{score:.4f}</span>",
                        unsafe_allow_html=True,
                    )
                    if snippet:
                        st.markdown(f"> {snippet[:300]}")
                    st.divider()
        elif intent == "policy":
            st.warning("⚠️ 未检索到政策依据——系统拒绝编造回答")
        # spatial intent 天然无 sources，不显示任何 sources 区
