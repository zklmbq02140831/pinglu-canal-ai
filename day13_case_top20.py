# -*- coding: utf-8 -*-
"""Day13案例验证准备：composite_index Top20高值网格 + 网格内企业类POI候选清单。
输出: data/processed/day13_case_top20.csv, data/processed/day13_case_candidates.csv
"""
import json
import pandas as pd
import h3

ROOT = r"d:\112\yunhe\pinglu-canal-ai"
RES = 7  # hex_id 前缀 87 => H3 res 7（格均面积≈5.16 km²）

# ---------- 1. Top20 网格 ----------
df = pd.read_csv(ROOT + r"\data\processed\land_sea_index_v1.csv")
top = df.sort_values("composite_index", ascending=False).head(20).copy()
top["center"] = top["hex_id"].apply(lambda h: h3.cell_to_latlng(h))
top["lat"] = top["center"].apply(lambda c: round(c[0], 6))
top["lng"] = top["center"].apply(lambda c: round(c[1], 6))
top.drop(columns="center", inplace=True)

STRIP_URBAN = {  # 凹槽条带·市区段（Day13协议第2节）
    "87415018bffffff": "向阳街道", "874150189ffffff": "水东街道",
    "87415018effffff": "沙埠镇", "87415018cffffff": "沙埠镇",
    "87415018dffffff": "沙埠镇", "87415018affffff": "水东街道",
    "874150188ffffff": "水东街道", "8741500a4ffffff": "子材街道",
    "874150016ffffff": "南珠街道",
}
STRIP_PORT = {  # 凹槽条带·港区段
    "874150123ffffff": "犀牛脚镇", "8741508d2ffffff": "钦州港经开区",
    "8741508d6ffffff": "钦州港经开区", "8741508d4ffffff": "保税港区",
}

def strip_label(h):
    if h in STRIP_URBAN:
        return "凹槽条带-市区段/" + STRIP_URBAN[h]
    if h in STRIP_PORT:
        return "凹槽条带-港区段/" + STRIP_PORT[h]
    return "非条带"

top["strip"] = top["hex_id"].apply(strip_label)

# ---------- 2. 距运河距离（中心点 -> 运河中心线, EPSG:32648）----------
with open(ROOT + r"\data\raw\canal_centerline.geojson", encoding="utf-8") as f:
    gj = json.load(f)

feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
lines = []
for ft in feats:
    g = ft.get("geometry") or ft
    if g.get("type") == "LineString":
        lines.append(g["coordinates"])
    elif g.get("type") == "MultiLineString":
        lines.extend(g["coordinates"])

try:
    from pyproj import Transformer
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    proj_lines = []
    for line in lines:
        pts = [tf.transform(x, y) for x, y in line]
        if len(pts) >= 2:
            proj_lines.append(LineString(pts))
    canal = unary_union(proj_lines)

    def dist_canal(lng, lat):
        p = tf.transform(lng, lat)
        return Point(p).distance(canal) / 1000.0
except ImportError:  # 兜底：WGS84最近顶点弦距
    import math
    verts = [tuple(xy) for line in lines for xy in line]

    def dist_canal(lng, lat):
        def hav(lon2, lat2):
            r = 6371000.0
            p1, p2 = math.radians(lat), math.radians(lat2)
            dp = p2 - p1
            dl = math.radians(lon2 - lng)
            a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
            return 2*r*math.asin(math.sqrt(a))
        return min(hav(vx, vy) for vx, vy in verts) / 1000.0

top["dist_canal_km"] = [round(dist_canal(x, y), 2) for x, y in zip(top["lng"], top["lat"])]
top["dist_port_km"] = top["port_dist"].round(2)

cols = ["hex_id", "composite_index", "strip", "lat", "lng",
        "dist_canal_km", "dist_port_km", "poi_density", "road_density", "gdp"]
print("=== Top20 高值网格 ===")
print(top[cols].to_string(index=False))
top[cols].to_csv(ROOT + r"\data\processed\day13_case_top20.csv",
                 index=False, encoding="utf-8-sig")

# ---------- 3. 网格内企业类 POI 候选 ----------
poi = pd.read_csv(ROOT + r"\data\processed\pois_amap_wgs84.csv")
poi["hex"] = [h3.latlng_to_cell(a, o, RES)
              for a, o in zip(poi["lat_wgs84"], poi["lng_wgs84"])]
top_set = set(top["hex_id"])
inc = poi[poi["hex"].isin(top_set)].copy()

# 企业类：公司企业/工厂/物流/仓储/园区
ENT = inc[inc["type"].astype(str).str.contains(
    "公司企业|工厂|物流|仓储|园区", na=False)].copy()

def ent_score(r):
    name = str(r["name"])
    s = 0
    if str(r["typecode"]).startswith("1701"):  # 工厂类
        s += 3
    for kw in ("公司", "厂", "集团", "物流", "仓储", "港务", "材料", "科技", "园"):
        if kw in name:
            s += 1
    for bad in ("便利店", "超市", "药店", "宾馆", "酒店", "饭店", "小吃", "奶茶"):
        if bad in name:
            s -= 5
    return -s

ENT["_s"] = ENT.apply(ent_score, axis=1)
ENT["dist_canal_km"] = [round(dist_canal(x, y), 2)
                        for x, y in zip(ENT["lng_wgs84"], ENT["lat_wgs84"])]
ENT["composite_index"] = ENT["hex"].map(top.set_index("hex_id")["composite_index"])
ENT["strip"] = ENT["hex"].map(top.set_index("hex_id")["strip"])

ENT = ENT.sort_values(["composite_index", "_s"])
out_cols = ["hex", "composite_index", "strip", "name", "type",
            "typecode", "address", "lng_wgs84", "lat_wgs84", "dist_canal_km"]
cand = ENT[out_cols].rename(columns={"hex": "hex_id", "name": "poi_name"})
cand.to_csv(ROOT + r"\data\processed\day13_case_candidates.csv",
            index=False, encoding="utf-8-sig")

print("\n=== 各Top20网格企业类候选（每格≤6，按格composite升序=条带优先）===")
for hex_id, grp in cand.groupby("hex_id"):
    meta = top[top["hex_id"] == hex_id].iloc[0]
    print(f"\n# {hex_id} composite={meta['composite_index']} {meta['strip']}")
    print(grp.drop(columns=["hex_id", "composite_index", "strip"])
             .head(6).to_string(index=False))
print(f"\n候选总数 {len(cand)}")
