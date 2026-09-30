# -*- coding: utf-8 -*-
"""
index_agent.py · 平陆运河空间分析 Agent（LangGraph）
=====================================================
功能：产业指数 / 网格排名 / 凹槽冷点三类问题的自动化应答。
数据源：全部只读本地快照 v2026-09，零网络请求（运行时）。
    - data/processed/land_sea_index_v1.csv  ← 五维指数主表（542格）
    - data/processed/grid_county_map.csv   ← 网格 → 县区映射
    - data/raw/canal_centerline.geojson     ← 运河中心线（算dist_canal）
    - src/agents/index_agent.py 内 STRIP_13 ← 13冷点 hex_id → 街道（Day13锁定）
LLM：智谱 GLM-4-Flash（langchain_openai.ChatOpenAI，base_url 指向 bigmodel）。
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, List, Optional

# —— 路径锚定：以项目根为基准，无论从哪里 import ——
_ROOT = Path(__file__).resolve().parents[2]
_INDEX_CSV = _ROOT / "data" / "processed" / "land_sea_index_v1.csv"
_COUNTY_CSV = _ROOT / "data" / "processed" / "grid_county_map.csv"
_CANAL_GEOJSON = _ROOT / "data" / "raw" / "canal_centerline.geojson"

SNAPSHOT = "v2026-09"

# —— 13 冷点格黄金清单（Day13 案例验证硬锁）——
# 来源：day13_case_top20.py STRIP_URBAN + STRIP_PORT
# 总数 13 = 9 市区段 + 4 港区段
STRIP_13: dict[str, str] = {
    # —— 凹槽条带 · 市区段（9 格）——
    "87415018bffffff": "向阳街道",
    "874150189ffffff": "水东街道",
    "87415018effffff": "沙埠镇",
    "87415018cffffff": "沙埠镇",
    "87415018dffffff": "沙埠镇",
    "87415018affffff": "水东街道",
    "874150188ffffff": "水东街道",
    "8741500a4ffffff": "子材街道",
    "874150016ffffff": "南珠街道",
    # —— 凹槽条带 · 港区段（4 格）——
    "874150123ffffff": "犀牛脚镇",
    "8741508d2ffffff": "钦州港经济技术开发区",
    "8741508d6ffffff": "钦州港经济技术开发区",
    "8741508d4ffffff": "广西钦州保税港区",
}

# —— 懒加载缓存（避免每次调工具都重读 CSV / GeoJSON）——
_df_index = None
_df_county = None
_canal_cache = None  # (shapely_multilinestring, transformer)


# =========================================================================
# 0. 数据层：CSV / GeoJSON 懒加载 + 纯本地几何计算
# =========================================================================
def _load_index():
    import pandas as pd
    global _df_index
    if _df_index is None:
        _df_index = pd.read_csv(_INDEX_CSV, encoding="utf-8-sig")
    return _df_index


def _load_county():
    import pandas as pd
    global _df_county
    if _df_county is None:
        _df_county = pd.read_csv(_COUNTY_CSV, encoding="utf-8-sig")
    return _df_county


def _load_canal():
    """返回 (shapely unary_union 运河中心线, pyproj Transformer WGS84→UTM48N)。"""
    global _canal_cache
    if _canal_cache is not None:
        return _canal_cache
    from pyproj import Transformer
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    with open(_CANAL_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    lines = []
    for ft in feats:
        g = ft.get("geometry") or ft
        if g.get("type") == "LineString":
            lines.append(g["coordinates"])
        elif g.get("type") == "MultiLineString":
            lines.extend(g["coordinates"])

    tf = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    proj_lines = []
    for line in lines:
        pts = [tf.transform(x, y) for x, y in line]
        if len(pts) >= 2:
            proj_lines.append(LineString(pts))
    canal = unary_union(proj_lines)
    _canal_cache = (canal, tf)
    return _canal_cache


def _dist_canal_km(lng: float, lat: float) -> float:
    """WGS84 坐标 → 投影后到运河中心线的最短距离 (km)。"""
    from shapely.geometry import Point
    canal, tf = _load_canal()
    p = tf.transform(lng, lat)
    return round(Point(p).distance(canal) / 1000.0, 2)


def _hex_center(h: str):
    """H3 res7 hex_id → (lat, lng)。"""
    import h3
    lat, lng = h3.cell_to_latlng(h)
    return lat, lng


# =========================================================================
# 1. 三个工具：全只读本地文件，返回 JSON 必须带 snapshot 字段
# =========================================================================
def get_index(hex_id: str) -> dict[str, Any]:
    """
    查询指定网格(hex_id)的五维指标 + 综合指数。
    Args:
        hex_id: H3 res7 网格 id，例如 "874150188ffffff"。
    Returns:
        {hex_id, county, poi_density, road_density, slope_mean, gdp, port_dist,
         composite_index, snapshot}
    """
    df = _load_index()
    row = df[df["hex_id"] == hex_id]
    if row.empty:
        return {"hex_id": hex_id, "error": "not_found", "snapshot": SNAPSHOT}

    county_df = _load_county()
    county_row = county_df[county_df["hex_id"] == hex_id]
    county = county_row["county"].iloc[0] if not county_row.empty else None

    r = row.iloc[0]
    return {
        "hex_id": hex_id,
        "county": county,
        "poi_density": round(float(r["poi_density"]), 4),
        "road_density": round(float(r["road_density"]), 4),
        "slope_mean": round(float(r["slope_mean"]), 4),
        "gdp": round(float(r["gdp"]), 2),
        "port_dist": round(float(r["port_dist"]), 2),
        "composite_index": round(float(r["composite_index"]), 4),
        "snapshot": SNAPSHOT,
    }


def get_top_grids(n: int = 10, region: Optional[str] = None) -> dict[str, Any]:
    """
    返回 composite_index 前 n 的网格及区县归属。
    Args:
        n: 返回前几名，默认 10。
        region: 可选过滤——县区名匹配（钦南区 / 钦北区 / 灵山县 / 横州市）或 '条带'/'非条带'。
    Returns:
        {top_n: [{hex_id, county, composite_index, rank}], snapshot}
    """
    import pandas as pd
    df = _load_index()
    county_df = _load_county()

    merged = df.merge(county_df, on="hex_id", how="left")

    if region:
        if region in ("条带", "凹槽", "冷点", "凹槽条带"):
            merged = merged[merged["hex_id"].isin(STRIP_13)]
        elif region in ("非条带", "非凹槽"):
            merged = merged[~merged["hex_id"].isin(STRIP_13)]
        else:
            merged = merged[merged["county"].str.contains(region, na=False)]

    top = merged.sort_values("composite_index", ascending=False).head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    result = top[["rank", "hex_id", "county", "composite_index"]].apply(
        lambda r: {
            "rank": int(r["rank"]),
            "hex_id": r["hex_id"],
            "county": r["county"] if pd.notna(r["county"]) else None,
            "composite_index": round(float(r["composite_index"]), 4),
        },
        axis=1,
    ).tolist()

    return {
        "top_n": result,
        "total_after_filter": int(len(merged)),
        "requested_n": n,
        "snapshot": SNAPSHOT,
    }


def get_coldspots() -> dict[str, Any]:
    """
    返回凹槽条带 13 格的黄金清单：hex_id / 所在街道 / composite_index / 距运河距离 km。
    （Day13 案例验证硬锁，数字与 dataset_card.md 一致。）
    Returns:
        {grids: [{hex_id, street, composite_index, dist_canal_km}], count: 13, snapshot}
    """
    df = _load_index()
    rows = []
    for hex_id, street in STRIP_13.items():
        row = df[df["hex_id"] == hex_id]
        if row.empty:
            continue
        r = row.iloc[0]
        lat, lng = _hex_center(hex_id)
        rows.append({
            "hex_id": hex_id,
            "street": street,
            "composite_index": round(float(r["composite_index"]), 4),
            "dist_canal_km": _dist_canal_km(lng, lat),
        })

    rows.sort(key=lambda x: x["composite_index"], reverse=True)
    return {
        "grids": rows,
        "count": len(rows),  # 必须 = 13
        "snapshot": SNAPSHOT,
    }


# =========================================================================
# 2. LangGraph Agent 构建
# =========================================================================
def _read_zhipu_key() -> str:
    """
    读取智谱 API Key：.env 的 ZHIPU_API_KEY → ZHIPUAI_API_KEY → 系统环境变量。
    与项目现有外部服务 key 双名兼容策略保持一致风格。
    """
    from pathlib import Path as _Path
    fallback = ""
    candidates = [
        _Path(__file__).resolve().parent / ".env",
        _Path(__file__).resolve().parents[2] / ".env",
        _Path(r"D:\yunhe\.env"),          # 与项目历史脚本一致的外部 .env 位置
        _Path(r"d:\112\yunhe\.env"),       # 额外兜底
    ]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if k.strip() == "ZHIPU_API_KEY":
                    return v
                if k.strip() == "ZHIPUAI_API_KEY" and not fallback:
                    fallback = v
    return (os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY")
            or fallback)


SYSTEM_PROMPT = """你是平陆运河经济带空间分析专员，只负责回答产业指数、网格排名类问题。
回答简洁专业，数字带单位，末尾提示"已为您定位到地图"。

背景格局（数字以工具查询结果为准，不背诵不扩展）：
- 指数呈港-城双核、中间塌陷结构；
- 运河城区段存在贴河低值凹槽条带（Gi*显著冷点，13格）；
- 条带以生活岸线为主、工业底色淡，含少量仓储锚点，通航后填空潜力大。

**铁律：涉及任何具体数字、hex_id、排名、密度、范围的问题，必须先调用相应工具查询，禁止凭背景知识或记忆直接作答。** 只有定性/方向性问题（如"哪里有潜力"）可直接引用上述背景。

只陈述事实，不推断因果。
"""


def build_agent():
    """
    构建 LangGraph ReAct Agent，绑定三个本地工具。
    使用 langgraph.prebuilt.create_react_agent → 自动处理 tool-call 循环。
    """
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    zhipu_key = _read_zhipu_key()
    if not zhipu_key:
        raise RuntimeError(
            "未找到 ZHIPU_API_KEY（或 ZHIPUAI_API_KEY），请在项目根 .env 配置"
        )

    llm = ChatOpenAI(
        model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=zhipu_key,
        temperature=0.2,
    )

    # —— 三个 LangChain Tool（用 @tool 装饰，自动生成 schema 给 LLM）——
    @tool
    def tool_get_index(hex_id: str) -> str:
        """查询指定 H3 res7 网格（hex_id，形如 '874150188ffffff'）的五维指标和综合指数。"""
        return json.dumps(get_index(hex_id), ensure_ascii=False)

    @tool
    def tool_get_top_grids(n: int = 10, region: Optional[str] = None) -> str:
        """返回 composite_index 前 n 的网格及区县归属。region 可选过滤：县区名（钦南区/钦北区/灵山县/横州市）、'条带'、'非条带'。"""
        return json.dumps(get_top_grids(n=n, region=region), ensure_ascii=False)

    @tool
    def tool_get_coldspots() -> str:
        """返回凹槽条带 13 格的黄金清单：hex_id / 街道 / composite_index / 距运河距离 km。共13格，Day13锁定。"""
        return json.dumps(get_coldspots(), ensure_ascii=False)

    tools = [tool_get_index, tool_get_top_grids, tool_get_coldspots]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return agent


# =========================================================================
# 3.5 兜底：检测数字问题后主动拉取真实工具数据注入 answer
# =========================================================================
def _proactive_fetch(question: str) -> Optional[tuple[str, list[str]]]:
    """
    当 LLM 未调工具但问题明显是数字查询时，主动调相应工具，
    用真实数据修正 answer 并收集 hex_ids。
    返回 (修正后的 answer, 新增 hex_ids 列表)；无需兜底则返回 None。
    """
    new_parts: list[str] = []
    extra_hex: list[str] = []

    # —— 场景 1：问题涉及 "冷点 / 凹槽 / 条带 / 13格" → 注入 get_coldspots() ——
    cold_keywords = ("冷点", "凹槽", "条带", "13格", "13 个")
    if any(k in question for k in cold_keywords):
        cs = get_coldspots()
        extras = [g["hex_id"] for g in cs["grids"]]
        extra_hex.extend(extras)
        range_str = (
            f"composite_index 范围 {cs['grids'][-1]['composite_index']:.4f}"
            f" ~ {cs['grids'][0]['composite_index']:.4f}"
        )
        mid = statistics.median(g["dist_canal_km"] for g in cs["grids"])
        new_parts.append(
            f"[数据兜底·本地快照] {cs['count']} 格凹槽条带，{range_str}，"
            f"距运河中位 {mid:.2f} km。"
        )

    # —— 场景 2：问题涉及 "活力最高 / 最高 / 排名" → 注入 get_top_grids() ——
    rank_keywords = ("活力最高", "最高", "排名", "前", "第一")
    if any(k in question for k in rank_keywords):
        top = get_top_grids(n=5)
        extras = [g["hex_id"] for g in top["top_n"]]
        extra_hex.extend(extras)
        top1 = top["top_n"][0]
        new_parts.append(
            f"[数据兜底·本地快照] composite_index 第1格 hex={top1['hex_id'][:12]}... "
            f"({top1['county']})，composite={top1['composite_index']:.4f}。"
        )

    if not new_parts:
        return None

    # 去重 hex
    seen = set()
    extra_hex = [h for h in extra_hex if not (h in seen or seen.add(h))]

    # 拼接：先放原 answer，再接兜底数据块
    return (" ".join(new_parts), extra_hex)


# =========================================================================
# 4. 对外主入口：run() 接收问题 → 返回标准 JSON 响应
# =========================================================================
def run(question: str, agent=None) -> dict[str, Any]:
    """
    执行一次 Agent 问答，返回标准化 JSON：
    {answer, hex_ids: [...], agent_flow: ["IndexAgent"], snapshot: "v2026-09"}
    """
    if agent is None:
        agent = build_agent()

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # 提取 LLM 最终回答
    messages = result.get("messages", [])
    answer = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            answer = msg.content
            break
    if not answer and messages:
        answer = str(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])

    # 从工具调用中提取涉及的 hex_ids（用于前端地图联动高亮）
    hex_ids: List[str] = []
    for msg in messages:
        # tool_call 参数里可能含 hex_id
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                args = tc.get("args", {}) if isinstance(tc, dict) else tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if "hex_id" in args:
                    hex_ids.append(args["hex_id"])
        # ToolMessage 内容里解析出 hex_id
        if getattr(msg, "type", None) == "tool":
            try:
                content = msg.content if hasattr(msg, "content") else str(msg)
                data = json.loads(content) if isinstance(content, str) else content
                if isinstance(data, dict):
                    if "hex_id" in data:
                        hex_ids.append(data["hex_id"])
                    if "grids" in data and isinstance(data["grids"], list):
                        for g in data["grids"]:
                            if "hex_id" in g:
                                hex_ids.append(g["hex_id"])
                    if "top_n" in data and isinstance(data["top_n"], list):
                        for g in data["top_n"]:
                            if "hex_id" in g:
                                hex_ids.append(g["hex_id"])
            except Exception:
                pass

    # 去重保序
    seen = set()
    hex_ids = [h for h in hex_ids if not (h in seen or seen.add(h))]

    # —— 兜底：LLM 没调工具但问题含数字关键词 → 主动调工具注入真实数据 ——
    numeric_keywords = ("范围", "多少", "数值", "composite", "密度",
                        "排名", "前几", "几个", "总数", "有多少", "比例")
    no_tool_but_numeric = (
        not hex_ids
        and any(k.lower() in question.lower() for k in numeric_keywords)
    )
    if no_tool_but_numeric:
        injected = _proactive_fetch(question)
        if injected:
            injected_text, extra_hex = injected
            answer = answer.rstrip("。") + "；" + injected_text
            for h in extra_hex:
                if h not in hex_ids:
                    hex_ids.append(h)

    return {
        "answer": answer,
        "hex_ids": hex_ids,
        "agent_flow": ["IndexAgent"],
        "snapshot": SNAPSHOT,
    }


# =========================================================================
# 4. CLI：python -m src.agents.index_agent "哪个网格产业活力最高"
# =========================================================================
if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "哪个网格产业活力最高"
    print(f"\n🗣️  问题：{q}\n")
    result = run(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
