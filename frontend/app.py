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
    initial_sidebar_state="expanded",
)


# =========================================================================
# CUSTOM_CSS —— 全部自定义样式集中于此，不散落
# =========================================================================
CUSTOM_CSS = """
<style>
/* ===== 顶部横幅 ===== */
.hero-banner {
    background: linear-gradient(135deg, #1E5A8A 0%, #2563EB 55%, #3B82F6 100%);
    padding: 36px 48px;
    border-radius: 14px;
    color: #FFFFFF;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(30, 90, 138, 0.25);
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -40px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 32px; font-weight: 700; letter-spacing: 2px;
    margin: 0 0 10px 0;
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.hero-subtitle {
    font-size: 15px; font-weight: 400; opacity: 0.92;
    margin: 0 0 14px 0;
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.hero-arch {
    font-size: 13px; opacity: 0.85; line-height: 1.6;
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.hero-snapshot {
    position: absolute; top: 24px; right: 32px;
    background: rgba(255,255,255,0.22);
    border: 1px solid rgba(255,255,255,0.45);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px; font-weight: 500;
    backdrop-filter: blur(6px);
}

/* ===== Agent Flow 彩色药丸 ===== */
.agent-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
    margin: 0 3px;
    letter-spacing: 0.5px;
    vertical-align: middle;
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.pill-router  { background: #7C3AED; }
.pill-index   { background: #2563EB; }
.pill-policy  { background: #059669; }
.pill-arrow {
    display: inline-block; color: #9CA3AF;
    font-size: 14px; margin: 0 2px; vertical-align: middle;
}

/* ===== 路由判定小字 ===== */
.router-reason {
    color: #9CA3AF; font-size: 12px;
    margin-top: 4px; font-style: italic;
}

/* ===== Sources 卡片 ===== */
.source-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.source-title {
    font-size: 14px; font-weight: 700; color: #1F2937;
    margin-bottom: 8px;
}
.source-score {
    display: inline-block;
    background: #059669; color: #FFFFFF;
    border-radius: 12px; padding: 2px 10px;
    font-size: 11px; font-weight: 600;
    margin-left: 8px; vertical-align: middle;
}
.source-snippet {
    background: #F9FAFB; color: #6B7280;
    font-size: 13px; line-height: 1.7;
    padding: 8px 12px; border-radius: 6px;
    border-left: 3px solid #D1D5DB;
    margin-top: 6px;
}

/* ===== 空 Sources 警示卡片 ===== */
.empty-warning {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border: 1px solid #F59E0B;
    border-radius: 10px;
    padding: 14px 20px;
    color: #92400E;
    font-size: 14px; font-weight: 500;
}

/* ===== Unknown 蓝框提示 ===== */
.unknown-box {
    background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
    border: 1px solid #3B82F6;
    border-radius: 10px;
    padding: 14px 20px;
    color: #1E40AF;
    font-size: 14px; font-weight: 500;
}

/* ===== 侧边栏系统信息小卡 ===== */
.sys-card {
    background: linear-gradient(135deg, #1E5A8A 0%, #2563EB 100%);
    color: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(30, 90, 138, 0.25);
}
.sys-card-title {
    font-size: 15px; font-weight: 700; margin-bottom: 6px;
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.sys-card-item {
    font-size: 12px; opacity: 0.9; line-height: 1.8;
}

/* ===== Chat 头像 emoji（隐藏默认字母头像后） ===== */
.stChatMessage [data-testid="avatar"] {
    background: transparent !important;
}

/* ===== 数据口径小字 ===== */
.sidebar-footnote {
    color: #9CA3AF; font-size: 11px;
    border-top: 1px solid #E5E7EB;
    padding-top: 10px; margin-top: 14px;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# 侧边栏
# =========================================================================
with st.sidebar:
    # 系统信息小卡
    st.markdown(
        """
        <div class="sys-card">
            <div class="sys-card-title">🚢 平陆运河</div>
            <div class="sys-card-item">多 Agent 智能问答系统</div>
            <div class="sys-card-item">📦 snapshot = <b>v2026-09</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 架构说明
    st.markdown(
        """
        <div style="font-size:12px;color:#6B7280;margin-bottom:12px">
            <b>三层架构</b><br>
            RouterAgent ─┬─→ IndexAgent（空间分析）<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─→ PolicyAgent（政策 RAG）
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # 演示预置按钮区（带 emoji）
    st.markdown("### 🎯 演示预置场景")

    # key 保持稳定（用 label 的简短 key），显示 label 带 emoji
    _PRESETS = [
        ("📍 起点坐标", "preset_coord",     "平陆运河起点坐标是什么"),
        ("🏘️ 沿线乡镇", "preset_town",      "检索沿线5公里内的乡镇"),
        ("📈 经济带动", "preset_economy",   "平陆运河对广西经济的带动作用"),
        ("⚠️ 负样本",   "preset_negative",  "跨境电商退税"),
        ("🌤️ 兜底题",   "preset_fallback", "今天天气怎么样"),
    ]

    for label, key, q in _PRESETS:
        if st.button(label, key=key, use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

    # 底部数据口径
    st.markdown(
        '<div class="sidebar-footnote">📚 13 份政策文档 · 232 chunks<br>'
        '阈值 0.45 · 离线只读</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# 主区 · 顶部横幅
# =========================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-snapshot">📦 snapshot = v2026-09</div>
        <div class="hero-title">平陆运河多Agent智能问答系统</div>
        <div class="hero-subtitle">基于 LangGraph + 智谱 GLM-4-Flash 的空间-政策协同问答</div>
        <div class="hero-arch">
            🧭 RouterAgent（意图路由） → 📍 IndexAgent（空间分析）<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ 📜 PolicyAgent（政策 RAG · Qdrant 向量检索）
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# 主区 · Chat 交互
# =========================================================================
question = st.chat_input("请输入您的问题…")

# —— 预置按钮触发：从 session_state 取待处理问题 ——
if question is None and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    # 用户消息（加 emoji 前缀作为头像文字）
    with st.chat_message("user"):
        st.markdown(f"🧭 {question}")

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
        # —— agent_flow 彩色药丸（消灭 markdown 星号 bug）——
        if agent_flow:
            pill_html_parts = []
            for i, name in enumerate(agent_flow):
                arrow = '<span class="pill-arrow">→</span>' if i > 0 else ""
                if name == "Router":
                    pill_html_parts.append(f'{arrow}<span class="agent-pill pill-router">Router</span>')
                elif name == "IndexAgent":
                    pill_html_parts.append(f'{arrow}<span class="agent-pill pill-index">IndexAgent</span>')
                elif name == "PolicyAgent":
                    pill_html_parts.append(f'{arrow}<span class="agent-pill pill-policy">PolicyAgent</span>')
                else:
                    pill_html_parts.append(f'{arrow}<span class="agent-pill">{name}</span>')
            st.markdown(" ".join(pill_html_parts), unsafe_allow_html=True)

        # —— router_reason 灰色小字 ——
        if router_reason:
            st.markdown(
                f'<div class="router-reason">🔍 路由判定：{router_reason}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")  # 间距

        # —— 答案正文（unknown 场景去重：兜底文案与蓝框重复，跳过）——
        if intent != "unknown":
            st.markdown(answer)

        # —— intent=unknown 蓝框提示 ——
        if intent == "unknown":
            st.markdown(
                '<div class="unknown-box">💡 该问题超出本系统范围，您可询问运河空间信息或政策相关内容</div>',
                unsafe_allow_html=True,
            )

        # —— sources 卡片区（仅 policy intent 有值）——
        if sources:
            st.markdown(f"📌 **政策依据（{len(sources)} 条）**")
            for i, src in enumerate(sources, 1):
                title = src.get("title", "未知文档")
                score = src.get("score", 0)
                snippet = src.get("snippet", src.get("content", ""))
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">
                            {i}. {title}
                            <span class="source-score">{score:.3f}</span>
                        </div>
                        {f'<div class="source-snippet">{snippet[:300]}</div>' if snippet else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif intent == "policy":
            # 负样本 / 空召回：橙黄警示卡片
            st.markdown(
                '<div class="empty-warning">⚠️ 未检索到政策依据——系统拒绝编造回答</div>',
                unsafe_allow_html=True,
            )
        # spatial intent 天然无 sources，不显示任何 sources 区
