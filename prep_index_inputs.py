# -*- coding: utf-8 -*-
"""
prep_index_inputs.py
====================
准备五维指数计算的两个缺失输入：
1. data/processed/roads_clip.geojson —— 广西 OSM 路网裁剪到 H3 网格范围，只留主干路
2. data/processed/counties.geojson   —— 高德API抓取南宁/钦州区县边界
                                        （freight 指标空间连接用的几何，统计数据另由用户提供）
运行：python prep_index_inputs.py
"""

# ========== 1. 导入依赖 ==========
import os                        # 路径检查
import time                      # 请求限速
import requests                  # 调高德行政区划分界 API
import geopandas as gpd          # 路网裁剪、写 GeoJSON
from shapely.geometry import Polygon, MultiPolygon   # 区县多边形构造
from shapely.ops import unary_union                  # 网格并集

# .env 读取 key（项目根没有则向上一级找）
def read_api_key():
    from pathlib import Path
    for p in (Path(".env").resolve(), Path("../.env").resolve()):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith(("AMAP_API_KEY=", "AMAP_KEY=")):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("未找到 AMAP_KEY")

KEY = read_api_key()

ROADS_SHP = os.path.join("data", "raw", "guangxi_roads.shp")        # 广西全境 OSM 路网
HEX_FILE = os.path.join("data", "processed", "hex_grid.geojson")    # H3 网格（裁剪范围）
ROADS_CLIP = os.path.join("data", "processed", "roads_clip.geojson")# 输出1：裁剪路网
COUNTIES_GEO = os.path.join("data", "processed", "counties.geojson")# 输出2：区县边界

# 主干路等级（Geofabrik shapefile 的 highway 字段取值），含 _link 匝道联络线
MAIN_CLASSES = {"motorway", "trunk", "primary", "secondary"}

# 抓区县边界的两个城市（adcode）
CITIES = {"南宁市": "450100", "钦州市": "450700"}


# ========== 2. 路网裁剪 ==========
def clip_roads():
    print(f"⏳ 读取路网 {ROADS_SHP}（约 100MB，稍等）...")
    try:
        roads = gpd.read_file(ROADS_SHP)             # 先按 cpg 声明的编码读
    except UnicodeDecodeError:                       # 中文 dbf 常见 GBK 伪装成 UTF-8
        print("   UTF-8 解码失败，改用 GBK 编码重读 ...")
        roads = gpd.read_file(ROADS_SHP, encoding="gbk")   # 指定 GBK 重读
    print(f"   原始路网 {len(roads)} 条，CRS={roads.crs.to_string()}")

    # 先按等级过滤（比先做空间裁剪快得多，能把 10 万条砍到几千条）
    hw = roads["highway"].fillna("")                 # 空值补空串
    mask = hw.isin(MAIN_CLASSES) | hw.str.endswith(
        tuple(c + "_link" for c in MAIN_CLASSES))    # motorway_link / trunk_link 等
    roads = roads[mask]                              # 只留主干路
    print(f"   主干路过滤后 {len(roads)} 条")

    # 裁剪范围 = 全部六边形的并集（近似运河缓冲区外扩边界）
    hexes = gpd.read_file(HEX_FILE)                  # 读 H3 网格
    region = unary_union(hexes.geometry.values)      # 网格并集面
    region_gdf = gpd.GeoDataFrame(geometry=[region], crs=hexes.crs)

    clipped = gpd.clip(roads, region_gdf)            # 空间裁剪到缓冲区范围
    clipped = clipped[["osm_id", "name", "highway", "geometry"]]  # 只留有用字段
    clipped.to_file(ROADS_CLIP, driver="GeoJSON")    # 写出
    print(f"💾 裁剪后路网 {len(clipped)} 条 -> {ROADS_CLIP}")


# ========== 3. 高德抓区县边界 ==========
def parse_polyline(pline):
    """解析高德 polyline 字符串为 shapely 面几何（多环用 | 分隔）。"""
    polys = []                                       # 收集各环
    for ring in pline.split("|"):                    # 多个环
        pts = []
        for pair in ring.split(";"):                 # "lng,lat;lng,lat;..."
            if "," not in pair:
                continue                             # 跳过空段
            lng, lat = pair.split(",")               # 拆经纬度
            pts.append((float(lng), float(lat)))
        if len(pts) >= 3:                            # 至少 3 点才成面
            polys.append(Polygon(pts))
    if not polys:
        return None                                  # 没有有效面
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)  # 单面/多面


def fetch_one(adcode):
    """按 adcode 查单个行政区的边界几何（extensions=all 时顶级才带 polyline）。"""
    r = requests.get(
        "https://restapi.amap.com/v3/config/district",
        params={"key": KEY, "keywords": adcode,
                "subdistrict": 0, "extensions": "all"},   # 只查本级
        timeout=20,
    ).json()
    if r.get("status") != "1":                            # key/配额错误
        raise SystemExit(f"高德返回错误: {r.get('info')} ({r.get('infocode')})")
    return parse_polyline(r["districts"][0].get("polyline", ""))


def fetch_counties():
    feats = []                                       # 收集 {county_name, geometry}
    for city, adcode in CITIES.items():              # 南宁、钦州逐市抓
        print(f"⏳ 抓取 {city} 区县边界 ...")
        r = requests.get(                            # 先拿区县清单（含 adcode）
            "https://restapi.amap.com/v3/config/district",
            params={"key": KEY, "keywords": adcode,
                    "subdistrict": 1, "extensions": "all"},
            timeout=20,
        ).json()
        if r.get("status") != "1":                   # 业务失败（key/配额问题）
            raise SystemExit(f"高德返回错误: {r.get('info')} ({r.get('infocode')})")
        for d in r["districts"][0]["districts"]:     # 遍历下属区县
            if d.get("level") != "district":         # 只要区县级
                continue
            geom = parse_polyline(d.get("polyline", ""))   # 先试清单自带的边界
            if geom is None:                         # 子级 polyline 常为空 → 按adcode单查
                geom = fetch_one(d["adcode"])
                time.sleep(0.3)                      # 逐个查询时限速
            if geom is not None:                     # 仍拿不到边界的跳过并提示
                feats.append({"county_name": d["name"], "geometry": geom})
            else:
                print(f"   ⚠️ {d['name']} 无边界数据，跳过")

    gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")   # 组装 GeoDataFrame
    gdf.to_file(COUNTIES_GEO, driver="GeoJSON")      # 写出
    print(f"💾 区县边界 {len(gdf)} 个 -> {COUNTIES_GEO}")
    print("   区县清单（stats_county.csv 的 county_name 请用这些名字）：")
    print("   " + "、".join(gdf["county_name"].tolist()))


# ========== 4. 主流程 ==========
if __name__ == "__main__":
    if not os.path.exists(ROADS_CLIP):               # 已生成过就跳过（可重复运行）
        clip_roads()
    else:
        print(f"✅ {ROADS_CLIP} 已存在，跳过路网裁剪")
    if not os.path.exists(COUNTIES_GEO):             # 区县边界同理
        fetch_counties()
    else:
        print(f"✅ {COUNTIES_GEO} 已存在，跳过区县边界抓取")
