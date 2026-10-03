# -*- coding: utf-8 -*-
"""
canal_facts.py · 平陆运河确定性事实锚定表（只读常量表 + 查询函数）
======================================================================
来源：维基百科"平陆运河"词条 + 交通运输部通航新闻（2026-09-16 首航）。
用途：IndexAgent 的 get_canal_fact 工具数据源，消灭 LLM 事实幻觉。
零外部依赖（仅 stdlib），运行时零网络请求。

对外接口：
    CANAL_FACTS: dict   全量事实（key → value）
    query_fact(keyword) → {matched_key, value, snapshot}
    get_all_coords()    → [{name, lon, lat}, ...]  所有含坐标的事实点
"""
from __future__ import annotations

import json
from typing import Any, Optional

SNAPSHOT = "v2026-09"

# =========================================================================
# 确定性常量表
# =========================================================================
CANAL_FACTS: dict[str, Any] = {
    # —— 起止点坐标（权威来源：维基词条 + 交通部）——
    "起点": {
        "name": "起点：横州市新福镇平塘江口（郁江西津水库）",
        "location": "广西壮族自治区南宁市横州市新福镇平塘江口，郁江西津水库库区",
        "lon": 109.06806,
        "lat": 22.64444,
        "lon_rounded": 109.068,
        "lat_rounded": 22.644,
    },
    "终点": {
        "name": "终点：北部湾钦州入海口",
        "location": "广西壮族自治区钦州市钦南区北部湾入海口",
        "lon": 108.58361,
        "lat": 21.69806,
        "lon_rounded": 108.584,
        "lat_rounded": 21.698,
    },

    # —— 工程参数 ——
    "全长": {
        "value_km": 134.2,
        "unit": "km",
        "text": "全长 134.2 公里",
    },
    "落差": {
        "value_m": 65,
        "unit": "m",
        "text": "总落差 65 米",
    },
    "通航等级": {
        "text": "内河Ⅰ级航道，可通航 5000 吨级船舶",
        "grade": "内河Ⅰ级",
        "tonnage": 5000,
        "unit": "吨级",
        "source": "交通运输部 · 平陆运河工程可行性研究报告",
    },
    "总投资": {
        "value_yi": 727,
        "unit": "亿元",
        "text": "总投资约 727 亿元人民币",
        "source": "广西壮族自治区人民政府 · 平陆运河项目公告",
    },
    "通过能力": {
        "value_wt": 8900,
        "unit": "万吨/年",
        "text": "设计年单向通过能力约 8900 万吨",
        "source": "交通运输部 · 平陆运河工程可行性研究报告",
    },
    "缩短里程": {
        "value_km": 560,
        "unit": "公里",
        "text": "较经广州出海缩短约 560 公里入海里程",
        "source": "广西壮族自治区人民政府 · 平陆运河新闻发布会",
    },

    # —— 三大枢纽 ——
    "枢纽": {
        "list": [
            {"name": "马道枢纽", "desc": "运河起点第一级枢纽，位于横州市平塘江口上游"},
            {"name": "企石枢纽", "desc": "运河中部枢纽，位于钦州市钦北区境内"},
            {"name": "青年枢纽", "desc": "运河终点枢纽，位于北部湾入海口附近"},
        ],
        "count": 3,
        "source": "维基百科 · 平陆运河词条",
    },

    # —— 时间节点 ——
    "开工": {
        "date": "2022-08-28",
        "text": "2022 年 8 月 28 日正式开工",
    },
    "首航": {
        "date": "2026-09-16",
        "text": "2026 年 9 月 16 日实现首次通航",
    },
    "建成": {
        "date": "2026-09",
        "text": "2026 年 9 月建成通航",
    },
    "建设目标": {
        "date": "2026-12",
        "text": "2026 年底主体建成",
        "source": "广西壮族自治区交通运输厅 · 平陆运河建设目标公告",
    },
    "生态措施": {
        "text": "生态护岸、鱼类增殖放流站、动物通道、临时生态补水设施等",
        "measures": [
            "生态护岸",
            "鱼类增殖放流站",
            "动物通道",
            "临时生态补水设施",
        ],
        "source": "平陆运河工程环境影响评价报告 · 生态保护专章",
    },

    # —— 运河概况（一段式描述，LLM 可直接引用）——
    "概况": {
        "text": (
            "平陆运河是中国西部陆海新通道骨干工程，北起广西南宁横州市新福镇平塘江口（郁江西津水库），"
            "南至钦州北部湾入海口，全长 134.2 公里，总落差 65 米，设置马道、企石、青年三大枢纽，"
            "可通航 5000 吨级船舶，总投资约 727 亿元。"
            "2022 年 8 月 28 日开工建设，2026 年 9 月 16 日实现首次通航。"
        ),
    },
}

# —— 关键词 → CANAL_FACTS key 的多别名映射（提升 query_fact 命中率）——
_KEYWORD_ALIASES: dict[str, str] = {
    # 起点别名
    "起点": "起点", "发源地": "起点", "源头": "起点", "平塘江口": "起点",
    "横州": "起点", "新福镇": "起点", "西津": "起点", "西津水库": "起点",
    # 终点别名
    "终点": "终点", "入海口": "终点", "出海口": "终点", "北部湾": "终点",
    "钦州港": "终点", "入港": "终点",
    # 坐标别名
    "坐标": None, "经度": None, "纬度": None, "经纬度": None,
    # 工程参数别名
    "全长": "全长", "长度": "全长", "多少公里": "全长", "距离": "全长",
    "落差": "落差", "水位差": "落差",
    "通航": "通航等级", "吨位": "通航等级", "吨级": "通航等级", "船舶": "通航等级",
    "航道": "通航等级", "航道等级": "通航等级", "内河": "通航等级", "内河一级": "通航等级", "内河Ⅰ级": "通航等级",
    "投资": "总投资", "多少钱": "总投资", "造价": "总投资",
    # 通过能力别名
    "通过能力": "通过能力", "运力": "通过能力", "吞吐": "通过能力", "年通过": "通过能力", "单向": "通过能力",
    # 缩短里程别名
    "缩短": "缩短里程", "缩短里程": "缩短里程", "入海里程": "缩短里程", "广州": "缩短里程", "出海": "缩短里程",
    # 枢纽别名
    "枢纽": "枢纽", "船闸": "枢纽", "马道": "枢纽", "企石": "枢纽", "青年": "枢纽",
    "三个枢纽": "枢纽", "三大枢纽": "枢纽",
    # 建设目标别名
    "建设目标": "建设目标", "主体": "建设目标", "主体建成": "建设目标",
    # 生态措施别名
    "生态": "生态措施", "生态保护": "生态措施", "环保": "生态措施", "生态措施": "生态措施",
    "护岸": "生态措施", "鱼类": "生态措施", "增殖": "生态措施", "动物通道": "生态措施",
    # 时间别名
    "开工": "开工", "开工时间": "开工", "动工": "开工",
    "首航": "首航", "通航时间": "首航", "开通": "首航",  # 注："通航"已在工程参数区→通航等级
    "建成": "建成", "完工": "建成",
    # 概况别名
    "概况": "概况", "介绍": "概况", "是什么": "概况", "基本信息": "概况",
}


# =========================================================================
# 查询函数（模糊匹配 + 返回结构化数据）
# =========================================================================
def query_fact(keyword: str) -> dict[str, Any]:
    """
    根据关键词查询运河事实。
    Args:
        keyword: 自然语言关键词（如"起点坐标"、"全长多少"、"开工时间"）。
    Returns:
        {matched_key, value, snapshot, coords:[{name,lon,lat}] (可选)}
    """
    if not keyword:
        return {"matched_key": None, "value": None, "snapshot": SNAPSHOT, "coords": []}

    text = keyword.strip()
    matched_key: Optional[str] = None

    # —— 特殊处理：问题含"坐标"/"经纬度" → 同时返回起点+终点坐标 ——
    if any(k in text for k in ("坐标", "经度", "纬度", "经纬度")):
        start = CANAL_FACTS["起点"]
        end = CANAL_FACTS["终点"]
        coords = [
            {"name": start["name"], "lon": start["lon"], "lat": start["lat"]},
            {"name": end["name"],   "lon": end["lon"],   "lat": end["lat"]},
        ]
        return {
            "matched_key": "坐标",
            "value": {
                "起点": start,
                "终点": end,
            },
            "coords": coords,
            "snapshot": SNAPSHOT,
        }

    # —— 多关键字迭代匹配（长 alias 优先，避免短词抢匹配）——
    sorted_aliases = sorted(_KEYWORD_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
    for alias, fact_key in sorted_aliases:
        if alias in text:
            matched_key = fact_key or alias
            break

    if matched_key is None:
        # 兜底：直接在 CANAL_FACTS 的 key 里做包含检测
        for k in CANAL_FACTS:
            if k in text or text in k:
                matched_key = k
                break

    if matched_key is None or matched_key not in CANAL_FACTS:
        return {"matched_key": None, "value": None, "snapshot": SNAPSHOT, "coords": []}

    value = CANAL_FACTS[matched_key]

    # —— 组装 coords（如果 value 本身含 lon/lat）——
    coords: list[dict[str, Any]] = []
    if isinstance(value, dict) and "lon" in value and "lat" in value:
        coords.append({
            "name": value.get("name", matched_key),
            "lon": value["lon"],
            "lat": value["lat"],
        })
    # 起点/终点分别命中时，也要同时放两个点（方便前端地图完整展示）
    if matched_key in ("起点", "终点"):
        for other_key in ("起点", "终点"):
            if other_key != matched_key:
                ov = CANAL_FACTS[other_key]
                coords.append({
                    "name": ov.get("name", other_key),
                    "lon": ov["lon"],
                    "lat": ov["lat"],
                })

    return {
        "matched_key": matched_key,
        "value": value,
        "coords": coords,
        "snapshot": SNAPSHOT,
    }


def get_all_coords() -> list[dict[str, Any]]:
    """返回运河所有含坐标的事实点（起点 + 终点）。"""
    coords: list[dict[str, Any]] = []
    for key in ("起点", "终点"):
        v = CANAL_FACTS[key]
        coords.append({
            "name": v["name"],
            "lon": v["lon"],
            "lat": v["lat"],
        })
    return coords


# =========================================================================
# CLI：python -m src.agents.canal_facts "起点坐标"
# =========================================================================
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "起点坐标"
    print(f"\n🔍 查询：{q}\n")
    result = query_fact(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n📍 全部坐标点：")
    print(json.dumps(get_all_coords(), ensure_ascii=False, indent=2))
