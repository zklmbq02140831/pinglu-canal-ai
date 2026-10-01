# -*- coding: utf-8 -*-
"""
frontend/app.py · 平陆运河多 Agent 智能问答系统 · Streamlit 演示前端
=======================================================================
直连 src/orchestrator.run(question)，无额外依赖。
"""
from __future__ import annotations

import os
import re
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
    page_title="智汇运河 · 平陆运河多智能体决策支持平台",
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

/* ===== 侧边栏轻量按钮覆盖 ===== */
button[kind="secondary"] {
    border: 1px solid #D1D5DB !important;
    background: #F9FAFB !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"]:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
    color: #1E5A8A !important;
}

/* ===== 事实锚定徽章（绿色） ===== */
.anchor-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #D1FAE5;
    border: 1px solid #10B981;
    color: #047857;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
}

/* ===== 兜底原文徽章（黄色） ===== */
.fallback-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #FEF3C7;
    border: 1px solid #F59E0B;
    color: #92400E;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
}

/* ===== 兜底提示横幅（黄色） ===== */
.fallback-alert {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border: 1px solid #F59E0B;
    border-radius: 10px;
    padding: 10px 16px;
    color: #92400E;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 12px;
}

/* ===== 兜底分条 chunk 卡片 ===== */
.fallback-chunk {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.fallback-chunk-title {
    font-size: 13px; font-weight: 700; color: #92400E; margin-bottom: 6px;
}
.fallback-chunk-text {
    font-size: 12px; color: #78350F; line-height: 1.7;
    white-space: pre-wrap;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# 侧边栏
# =========================================================================
# =========================================================================
# 底图源配置（仅标准 / 卫星可用；天地图需 TIANDITU_KEY 才启用）
# pydeck 渲染自定义 XYZ 瓦片：将瓦片 URL 转为 mapbox raster 样式 JSON
# 字符串，通过 map_style 传给 Deck —— 这是 deck.gl 9.x 唯一能正确渲染
# 自定义 XYZ 瓦片的方式（TileLayer 在此版本的 pydeck+Streamlit 组合
# 中会被错误地序列化为 GeoJsonLayer）。
# =========================================================================
_BASEMAP_OPTIONS = {
    "标准地图": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "卫星影像": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

# 天地图（条件启用）
_TIANDITU_KEY = os.environ.get("TIANDITU_KEY", "")
_HAS_TIANDITU = bool(_TIANDITU_KEY)
if _HAS_TIANDITU:
    _BASEMAP_OPTIONS["天地图影像"] = (
        f"https://t{sorted(list('01234567'))[0]}.tianditu.gov.cn/img_w/wmts?"
        f"SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&"
        f"TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={{z}}&TILEROW={{y}}&TILECOL={{x}}&"
        f"tk={_TIANDITU_KEY}"
    )


def _tile_url_to_mapstyle(tile_url: str) -> str:
    """将 XYZ 瓦片模板 URL 转为 data:application/json;base64, 形式的 mapbox raster 样式。

    为什么：deck.gl 9.x 的 DeckGL 组件要求 mapStyle 为字符串；
    普通 mapbox 样式 JSON 字符串若包含 http:// 会被误判为 URL 去 fetch，
    dict 又会在 JS 端触发 e.mapStyle?.includes is not a function；
    data URI 是唯一能让 deck.gl 正确识别为内联样式的方式。
    """
    import base64 as _b64
    import json as _json
    style = {
        "version": 8,
        "sources": {
            "custom-tiles": {
                "type": "raster",
                "tiles": [tile_url],
                "tileSize": 256,
            }
        },
        "layers": [
            {
                "id": "custom-tiles",
                "type": "raster",
                "source": "custom-tiles",
                "minzoom": 0,
                "maxzoom": 19,
            }
        ],
    }
    json_str = _json.dumps(style)
    b64 = _b64.b64encode(json_str.encode("utf-8")).decode("ascii")
    return f"data:application/json;base64,{b64}"


with st.sidebar:
    # 系统信息小卡
    st.markdown(
        """
        <div class="sys-card">
            <div class="sys-card-title">🚢 智汇运河</div>
            <div class="sys-card-item">多智能体决策支持与空间智能平台</div>
            <div class="sys-card-item">📦 snapshot = <b>v2026-09</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # —— 底图选择器 ——
    st.markdown("🗺️ **底图切换**")
    _bm_labels = list(_BASEMAP_OPTIONS.keys())
    _default_idx = 0
    _selected_basemap_label = st.radio(
        "底图",
        _bm_labels,
        index=_default_idx,
        horizontal=True,
        key="basemap_selector",
        label_visibility="collapsed",
    )
    st.session_state["selected_basemap_url"] = _BASEMAP_OPTIONS[_selected_basemap_label]

    st.divider()

    # —— 陆海物流路径演示 ——
    with st.expander("🚢 陆海物流路径演示", expanded=False):
        st.markdown("勾选下列路线，地图将叠加示意线路（WGS84 坐标）")
        st.markdown("")

        # 4 条路线复选框（key 稳定，与 geojson 顺序对齐）
        _ROUTE_CHECKBOXES = [
            ("钦州港 → 南宁 → 贵阳（陆路·蓝）",   "route_guiyang",   "#2563EB"),
            ("钦州港 → 南宁 → 成都（陆路·橙）",   "route_chengdu",  "#EA580C"),
            ("钦州港 → 南宁 → 昆明（陆路·紫）",   "route_kunming",   "#7C3AED"),
            ("钦州港 → 海上 → 新加坡港（海运·绿）", "route_singapore", "#059669"),
        ]
        _checked_routes = {}
        for label, key, color in _ROUTE_CHECKBOXES:
            _checked_routes[key] = st.checkbox(label, key=key, value=False)

        # 把勾选状态缓存到 session_state 供主渲染块读取
        st.session_state["route_selection"] = _checked_routes

    st.divider()

    # 快捷提问按钮区（轻量样式）
    st.markdown("### 💡 快捷提问")

    # key 保持稳定（用 label 的简短 key），显示 label 带 emoji
    _PRESETS = [
        ("📍 起点坐标", "preset_coord",     "平陆运河起点坐标是什么"),
        ("📊 网格指标", "preset_town",      "平陆运河沿线网格的空间指标概况"),
        ("📈 经济带动", "preset_economy",   "平陆运河对广西经济的带动作用"),
        ("⚠️ 负样本",   "preset_negative",  "跨境电商退税"),
        ("🌤️ 兜底题",   "preset_fallback", "今天天气怎么样"),
    ]

    for label, key, q in _PRESETS:
        if st.button(label, key=key, use_container_width=True, type="secondary"):
            st.session_state["pending_question"] = q
            st.rerun()

    st.divider()

    # 关于本系统折叠器（默认收起）
    with st.expander("ℹ️ 关于本系统", expanded=False):
        st.markdown(
            """
            **系统架构**

            RouterAgent 意图路由 → IndexAgent 空间分析 / PolicyAgent 政策 RAG

            ---

            **数据口径**
            - 📦 snapshot = `v2026-09`（离线只读）
            - 📚 13 份政策文档 · 232 chunks
            - 🎯 向量检索阈值 `0.45`（Cosine）
            """
        )


# =========================================================================
# 主区 · 顶部横幅
# =========================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-snapshot">📦 snapshot = v2026-09</div>
        <div class="hero-title">智汇运河</div>
        <div class="hero-subtitle">面向中国-东盟陆海联动的多智能体决策支持与空间智能平台</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# 主区 · 陆海物流路径演示地图（独立于问答流，勾选即显示）
# =========================================================================
_route_selection = st.session_state.get("route_selection", {})
_any_route_checked = any(_route_selection.values())

if _any_route_checked:
    import json as _json
    import pydeck as pdk
    from pathlib import Path as _P

    # geojson 路径：frontend/data/routes.geojson
    _geojson_path = _P(__file__).resolve().parent / "data" / "routes.geojson"

    # key 与侧边栏 _ROUTE_CHECKBOXES 顺序严格对齐 → feature 序号
    _KEY_TO_FIDX = {
        "route_guiyang": 0,
        "route_chengdu": 1,
        "route_kunming": 2,
        "route_singapore": 3,
    }

    try:
        with open(_geojson_path, "r", encoding="utf-8") as _f:
            _fc = _json.load(_f)

        _sel_features = []
        for _ck, _idx in _KEY_TO_FIDX.items():
            if _route_selection.get(_ck):
                _sel_features.append(_fc["features"][_idx])

        if _sel_features:
            # —— 准备 PathLayer 数据（[[lon,lat]...] 序列）——
            _path_rows = []
            _endpoint_rows = []  # 端点（起终点 + 路径末端）
            for _feat in _sel_features:
                _coords = _feat["geometry"]["coordinates"]  # [[lon,lat], ...]
                _col_hex = _feat["properties"]["color"]
                _col_rgb = [
                    int(_col_hex[1:3], 16),
                    int(_col_hex[3:5], 16),
                    int(_col_hex[5:7], 16),
                ]
                # PathLayer 字段：path（序列）+ color
                _path_rows.append({
                    "path": _coords,
                    "name": _feat["properties"]["name"],
                    "mode": _feat["properties"]["mode"],
                    "color": _col_rgb,
                })
                # 端点：每个路径的起点 + 终点
                _start = _coords[0]
                _end = _coords[-1]
                _endpoint_rows.append({
                    "lon": _start[0], "lat": _start[1],
                    "label": f"{_feat['properties']['name']}（起点）",
                    "color": _col_rgb,
                })
                _endpoint_rows.append({
                    "lon": _end[0], "lat": _end[1],
                    "label": f"{_feat['properties']['name']}（终点）",
                    "color": _col_rgb,
                })

            # —— 地图视野：覆盖所有路径的经纬度边界框 ——
            _all_lons = []
            _all_lats = []
            for _feat in _sel_features:
                for _c in _feat["geometry"]["coordinates"]:
                    _all_lons.append(_c[0])
                    _all_lats.append(_c[1])
            _view_lon = (min(_all_lons) + max(_all_lons)) / 2
            _view_lat = (min(_all_lats) + max(_all_lats)) / 2
            # 中国-东南亚范围足够大，zoom=4 即可覆盖
            _view_zoom = 4

            # —— 共享底图：mapbox raster 样式 dict（map_provider='mapbox' 配合使用）——
            _bm_url = st.session_state.get("selected_basemap_url", _BASEMAP_OPTIONS["标准地图"])
            _map_style = _tile_url_to_mapstyle(_bm_url)

            # —— PathLayer ——
            _path_layer = pdk.Layer(
                "PathLayer",
                data=_path_rows,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                width_max_pixels=8,
                get_width=5,
                rounded=True,
                pickable=True,
            )

            # —— 端点 ScatterplotLayer ——
            _ep_layer = pdk.Layer(
                "ScatterplotLayer",
                data=_endpoint_rows,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=20000,
                radius_min_pixels=5,
                radius_max_pixels=12,
                pickable=True,
                filled=True,
                stroked=True,
                line_width_min_pixels=1,
            )

            _route_deck = pdk.Deck(
                map_provider="mapbox",
                map_style=_map_style,
                initial_view_state=pdk.ViewState(
                    latitude=_view_lat,
                    longitude=_view_lon,
                    zoom=_view_zoom,
                    pitch=0,
                ),
                layers=[_path_layer, _ep_layer],
                tooltip={"text": "{label}"},
            )

            st.markdown("🚢 **陆海物流路径演示（示意）**")
            st.pydeck_chart(_route_deck, height=420)
            st.caption("示意线路，基于公开通道规划，非精确导航路径")

    except FileNotFoundError:
        st.warning(f"未找到路径数据文件：{_geojson_path}")
    except Exception as _e:
        st.warning(f"路径地图渲染失败：{_e}")

# —— 聊天输入分隔 ——
st.divider()


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
            original_answer = answer

            # —— 检测兜底原文标记 ——
            fallback_match = re.match(
                r"^\[政策原文兜底·本地快照\]\s*\n?(.*)", original_answer, re.DOTALL
            )
            if fallback_match:
                # 渲染黄色徽章 + 兜底提示横幅
                st.markdown(
                    '<div class="fallback-badge">📜 兜底原文</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="fallback-alert">⚠️ 在线合成暂不可用,以下为检索原文</div>',
                    unsafe_allow_html=True,
                )
                # 解析 JSON chunks
                import json as _json
                raw_json = fallback_match.group(1).strip()
                try:
                    chunks = _json.loads(raw_json)
                    if isinstance(chunks, list):
                        for i, c in enumerate(chunks, 1):
                            st.markdown(
                                f"""
                                <div class="fallback-chunk">
                                    <div class="fallback-chunk-title">
                                        {i}. 《{c.get('title', '未知')}》
                                        {c.get('level', '')} · {c.get('issue_date', '')}
                                    </div>
                                    <div class="fallback-chunk-text">{c.get('text', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                except Exception:
                    # JSON 解析失败 → 退化为直接显示
                    st.markdown(raw_json)
            else:
                # 正则剥离 [事实锚定·本地快照] 前缀（可能出现多次）
                cleaned_answer = re.sub(r"\[事实锚定·本地快照\]\s*", "", answer)
                has_anchor_tag = cleaned_answer != original_answer
                # 渲染绿色徽章（独立标签，不混正文）
                if has_anchor_tag:
                    st.markdown('<div class="anchor-badge">📌 事实锚定</div>', unsafe_allow_html=True)
                st.markdown(cleaned_answer)

        # —— intent=unknown 蓝框提示 ——
        if intent == "unknown":
            st.markdown(
                '<div class="unknown-box">💡 该问题超出本系统范围，您可询问运河空间信息或政策相关内容</div>',
                unsafe_allow_html=True,
            )

        # —— 地图渲染（IndexAgent 命中运河事实时 coords 非空）——
        coords_list = result.get("coords", [])
        if coords_list:
            import pandas as pd
            import pydeck as pdk

            map_df = pd.DataFrame([
                {"lat": c["lat"], "lon": c["lon"], "label": c.get("name", "")}
                for c in coords_list
            ])
            st.markdown("🗺️ **平陆运河关键点位（真实坐标）**")

            # pydeck ScatterplotLayer：起点绿 #059669 / 终点红 #DC2626 / 其他蓝
            colors = []
            for lbl in map_df["label"]:
                if "起点" in str(lbl):
                    colors.append([5, 150, 105])      # #059669
                elif "终点" in str(lbl):
                    colors.append([220, 38, 38])      # #DC2626
                else:
                    colors.append([37, 99, 235])      # #2563EB（其他）
            map_df["color"] = colors

            # 地图视野：以起终点中点为中心，zoom=8
            view_lat = map_df["lat"].mean()
            view_lon = map_df["lon"].mean()

            # —— 共享底图：mapbox raster 样式 dict（map_provider='mapbox' 配合使用）——
            _bm_url = st.session_state.get("selected_basemap_url", _BASEMAP_OPTIONS["标准地图"])
            _map_style = _tile_url_to_mapstyle(_bm_url)

            deck = pdk.Deck(
                map_provider="mapbox",
                map_style=_map_style,
                initial_view_state=pdk.ViewState(
                    latitude=view_lat,
                    longitude=view_lon,
                    zoom=8,
                    pitch=30,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lon", "lat"],
                        get_color="color",
                        get_radius=8000,
                        radius_min_pixels=6,
                        radius_max_pixels=12,
                        pickable=True,
                        filled=True,
                        stroked=True,
                        line_width_min_pixels=2,
                    ),
                ],
                tooltip={"text": "{label}\n经度: {lon}\n纬度: {lat}"},
            )
            st.pydeck_chart(deck, height=380)

            # 图例（起终点颜色标识）
            legend_items = []
            for _, row in map_df.iterrows():
                lbl = str(row["label"])
                if "起点" in lbl:
                    legend_items.append(f"🟢 {lbl}")
                elif "终点" in lbl:
                    legend_items.append(f"🔴 {lbl}")
                else:
                    legend_items.append(f"🔵 {lbl}")
            st.caption(" · ".join(legend_items))

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
                            <span class="source-score" title="向量余弦相似度;阈值0.45,低于则拒答">相关度 {score:.3f}</span>
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
