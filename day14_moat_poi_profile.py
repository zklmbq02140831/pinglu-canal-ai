# -*- coding: utf-8 -*-
"""
day14_moat_poi_profile.py
=========================
凹槽假说替代验证——POI品类结构分析（Day14）。

背景：Day12 发现运河市区段「凹槽条带」（13 格 Gi* 显著冷点，工业口径产物）。
本脚本用**类目全量** POI 检验其成因假说「工业避河」（工业贴河、生活避河）：
  实验组 = 凹槽条带 13 格
  对照组 = 距港 20-30km 且距运河线 >3km 的城区格（与条带同处市区带、避开条带）
  生活类 = 餐饮(05)+购物(06)+生活服务(07)+医疗(09)+住宅(12)+科教文化(14)
  工业类 = 公司企业-工厂 + 仓储物流（按 type 文本判别）
指标 = 生活类密度 / 工业类密度 / 生活工业比，双组对比。

数据：高德 v3/place/polygon 多边形搜索按大类抓全（非关键词定向），
     GCJ-02→WGS-84 纠偏后落格（sjoin within）。
判定：实验组生活/工业比 >1.5×对照组 → 支持「工业避河」；
     两组接近 → 假说不成立，改用「城区段功能未分化」叙事。
     无论哪种，凹槽条带 Gi* 统计事实不受影响。

运行：python day14_moat_poi_profile.py
Key ：d:/112/yunhe/.env 的 AMAP_KEY（兼容 AMAP_API_KEY）
缓存：data/raw/moat_pois_cache.json（重跑不再请求 API）
输出：data/raw/moat_pois_full_wgs84.csv + output/moat_poi_profile.png
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from convert_gcj02_to_wgs84 import gcj02_to_wgs84

ROOT = Path(__file__).resolve().parent
HEX_FILE = ROOT / "data/processed/hex_grid.geojson"
CANAL_FILE = ROOT / "data/raw/canal_centerline.geojson"
CACHE_FILE = ROOT / "data/raw/moat_pois_cache.json"
POI_OUT = ROOT / "data/raw/moat_pois_full_wgs84.csv"
PNG_OUT = ROOT / "output/moat_poi_profile.png"

PORT_LON, PORT_LAT = 108.60, 21.70          # 钦州港锚点（与 compute_index 一致）
UTM = "EPSG:32648"

# 凹槽条带 13 格（Day13 协议复算清单，与 Day12 z∈[-3.84,-2.10] 校验吻合）
STRIP_IDS = [
    "874150123ffffff", "87415018bffffff", "874150189ffffff",
    "87415018effffff", "87415018cffffff", "87415018dffffff",
    "87415018affffff", "8741508d2ffffff", "8741508d6ffffff",
    "874150188ffffff", "8741500a4ffffff", "8741508d4ffffff",
    "874150016ffffff",
]

LIFE_PREFIX = ("05", "06", "07", "09", "12", "14")   # 餐饮/购物/生活服务/医疗/住宅/科教
CATS = [f"{i:02d}0000" for i in range(1, 21)]        # 高德 20 个大类，类目全量

API_URL = "https://restapi.amap.com/v3/place/polygon"
PAGE_SIZE = 25
SLEEP = 0.35                                          # 个人 key 3 QPS 留余量
SUSPECT_N = 190                                       # polygon搜索实测约200条翻页上限，≥190即细分


# ---------- key ----------
def read_api_key() -> str:
    for p in (Path("d:/112/yunhe/.env"), ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() in ("AMAP_KEY", "AMAP_API_KEY"):
                        return v.strip().strip("'\"")
    return os.environ.get("AMAP_KEY") or os.environ.get("AMAP_API_KEY") or ""


# ---------- 抓取：多边形搜索，超阈值自动 2×2 细分 ----------
def fetch_polygon_series(key, polygon_pts, typecode, cache, ck):
    """单(多边形,大类)序列抓取；返回 (POI列表, 是否完整)。仅完整序列写入缓存，
    防止配额/网络错误把截断结果当完整数据静默落盘。"""
    if ck in cache:
        return cache[ck], True
    ring = ";".join(f"{x:.6f},{y:.6f}" for x, y in polygon_pts)
    ring = ring + ";" + f"{polygon_pts[0][0]:.6f},{polygon_pts[0][1]:.6f}"
    pois, complete, quota_dead = [], False, False
    for page in range(1, 101):
        d = None
        for attempt in range(3):              # 网络抖动重试：20s超时+递增退避
            try:
                r = requests.get(API_URL, params={
                    "key": key, "polygon": ring, "types": typecode,
                    "offset": PAGE_SIZE, "page": page,
                    "extensions": "base",
                }, timeout=20)
                d = r.json()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"   ⚠️ {ck} p{page} 连续3次网络失败: {e}")
                time.sleep(1.5 * (attempt + 1))
        if d is None:
            break                             # 网络失败 → 不完整、不缓存
        time.sleep(SLEEP)
        if d.get("infocode") == "10003":
            print("⛔ 日配额超限，终止"); quota_dead = True; break
        if d.get("status") != "1":
            if d.get("infocode") == "10006":  # 该范围无此类目 → 视为完整空集
                complete = True
            else:
                print(f"   ⚠️ {ck} p{page}: {d.get('info')}({d.get('infocode')})")
            break
        batch = d.get("pois") or []
        if not batch:
            complete = True
            break
        pois.extend(batch)
        if len(pois) >= int(d.get("count") or 0):
            complete = True
            break
    if quota_dead:
        _save_cache(cache); sys.exit(1)       # 配额超限：不缓存该序列，重跑可无损续抓
    if complete:
        cache[ck] = pois
        _save_cache(cache)
    return pois, complete


def fetch_cat_with_split(key, bb, typecode, cache, depth=0):
    """抓 (bbox,大类)；疑似翻页截断则 2×2 细分递归。bb=(minx,miny,maxx,maxy)。
    返回 (POI列表, 是否完整)；任一子序列不完整则整体标记不完整。"""
    ck = f"{typecode}_{depth}_" + "_".join(f"{v:.3f}" for v in bb)
    pts = [(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])]
    pois, ok = fetch_polygon_series(key, pts, typecode, cache, ck)
    if ok and len(pois) >= SUSPECT_N and depth < 4:
        mx, my = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        subs = [(bb[0], bb[1], mx, my), (mx, bb[1], bb[2], my),
                (bb[0], my, mx, bb[3]), (mx, my, bb[2], bb[3])]
        merged, ids, all_ok = [], set(), True
        for sb in subs:
            sub_pois, sub_ok = fetch_cat_with_split(key, sb, typecode, cache, depth + 1)
            all_ok = all_ok and sub_ok
            for p in sub_pois:
                if p.get("id") and p["id"] not in ids:
                    ids.add(p["id"]); merged.append(p)
        print(f"   ↳ {typecode} depth{depth} 触发细分: {len(pois)} → 合并去重 {len(merged)}")
        return merged, all_ok
    return pois, ok


def _save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# ---------- 主流程 ----------
def main():
    key = read_api_key()
    if not key:
        sys.exit("未找到 AMAP_KEY / AMAP_API_KEY")

    # ===== 1. 分组与 bbox =====
    hexes = gpd.read_file(HEX_FILE)
    canal = gpd.read_file(CANAL_FILE)
    utm = hexes.to_crs(UTM)
    cent = utm.geometry.centroid
    canal_utm = canal.to_crs(UTM).geometry.union_all()
    port_utm = gpd.GeoSeries([Point(PORT_LON, PORT_LAT)],
                             crs="EPSG:4326").to_crs(UTM).iloc[0]
    hexes["d_canal_km"] = cent.distance(canal_utm).to_numpy() / 1000
    hexes["d_port_km"] = cent.distance(port_utm).to_numpy() / 1000

    exp = hexes[hexes.hex_id.isin(STRIP_IDS)]
    ctl = hexes[(hexes.d_port_km.between(20, 30)) & (hexes.d_canal_km > 3)]
    assert len(exp) == 13, f"实验组应为13格，实际{len(exp)}"
    print(f"[1] 实验组（凹槽条带）: {len(exp)} 格 | 对照组（20-30km距港 & >3km距河）: {len(ctl)} 格")

    bbox_exp = exp.total_bounds
    bbox_ctl = ctl.total_bounds
    for name, bb in (("实验组bbox", bbox_exp), ("对照组bbox", bbox_ctl)):
        w = gpd.GeoSeries([box(*bb)], crs="EPSG:4326").to_crs(UTM).area.iloc[0] / 1e6
        print(f"    {name}: {np.round(bb, 4).tolist()}  ≈{w:.1f} km²")

    # 组面积（UTM 米制）
    area_exp = utm[utm.hex_id.isin(STRIP_IDS)].area.sum() / 1e6
    area_ctl = utm[utm.hex_id.isin(ctl.hex_id)].area.sum() / 1e6
    print(f"    组面积: 实验 {area_exp:.2f} km² | 对照 {area_ctl:.2f} km²")

    # ===== 2. 全量抓取（两个 bbox 合并池，poi_id 去重） =====
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    raw, failures = {}, []
    for gname, bb in (("exp", tuple(bbox_exp)), ("ctl", tuple(bbox_ctl))):
        for tc in CATS:
            got, ok = fetch_cat_with_split(key, bb, tc, cache)
            print(f"    {gname} 大类{tc[:2]}: {len(got)}" + ("" if ok else "  ⚠️不完整"))
            if not ok:
                failures.append(f"{gname}/{tc[:2]}")
                continue                  # 不把残缺数据并入分析池
            for p in got:
                pid = p.get("id")
                if pid and pid not in raw:
                    raw[pid] = p
    _save_cache(cache)
    print(f"[2] 两 bbox 合并去重后 POI 总数: {len(raw)}")
    if failures:
        print(f"⛔ {len(failures)} 个(组,大类)数据不完整，已中止分析: {', '.join(failures)}")
        print("   待配额恢复后重跑本脚本即可无损续抓（缓存完整序列不会重复请求）")
        sys.exit(2)

    # ===== 3. 纠偏 + 落表 =====
    recs = []
    for p in raw.values():
        loc = p.get("location") or ""
        if "," not in loc:
            continue
        lng, lat = (float(v) for v in loc.split(",")[:2])
        wgs = gcj02_to_wgs84(lng, lat)
        recs.append({
            "poi_id": p.get("id"), "name": p.get("name"),
            "lng_wgs84": wgs[0], "lat_wgs84": wgs[1],
            "type": p.get("type"), "typecode": p.get("typecode"),
        })
    poi = pd.DataFrame(recs)
    poi.to_csv(POI_OUT, index=False, encoding="utf-8-sig")
    print(f"[3] 纠偏后落盘 {len(poi)} 条 → {POI_OUT.name}")

    # ===== 4. 品类分组 + 落格 =====
    tc = poi["typecode"].astype(str).str.zfill(6)
    poi["life"] = tc.str[:2].isin(LIFE_PREFIX)
    ty = poi["type"].astype(str)
    poi["indus"] = ty.str.contains("工厂") | ty.str.contains("仓储")
    print(f"    生活类 {poi.life.sum()} 条 | 工业类 {poi.indus.sum()} 条")
    print("    工业类 type 构成:", poi.loc[poi.indus, "type"].value_counts().head(6).to_dict())

    gpoi = gpd.GeoDataFrame(poi, geometry=gpd.points_from_xy(poi.lng_wgs84, poi.lat_wgs84),
                            crs="EPSG:4326")
    j = gpd.sjoin(gpoi, hexes[["hex_id", "geometry"]], how="inner", predicate="within")
    j_exp = j[j.hex_id.isin(STRIP_IDS)]
    j_ctl = j[j.hex_id.isin(ctl.hex_id)]

    def metrics(jj, area):
        life = int(jj.life.sum()); ind = int(jj.indus.sum())
        return life / area, ind / area, (life / ind if ind else np.nan), life, ind

    m_exp = metrics(j_exp, area_exp)
    m_ctl = metrics(j_ctl, area_ctl)

    # ===== 5. 对照表 + 判定 =====
    tbl = pd.DataFrame(
        [m_exp, m_ctl],
        index=["实验组(凹槽条带13格)", "对照组(城区带格)"],
        columns=["生活类密度(个/km²)", "工业类密度(个/km²)", "生活/工业比", "生活类POI数", "工业类POI数"])
    tbl.loc[:, ["生活类密度(个/km²)", "工业类密度(个/km²)"]] = \
        tbl[["生活类密度(个/km²)", "工业类密度(个/km²)"]].round(3)
    tbl["生活/工业比"] = tbl["生活/工业比"].round(2)
    print("\n==== 品类结构对照表 ====")
    print(tbl.to_string())

    ratio_ratio = m_exp[2] / m_ctl[2]
    print(f"\n比值倍数（实验/对照）: {ratio_ratio:.2f}x")
    if ratio_ratio > 1.5:
        verdict = "实验组生活/工业比显著高于对照组(>1.5x) → 支持「工业避河」假说"
    else:
        verdict = "两组接近 → 「工业避河」假说不成立，改用「城区段功能未分化」叙事"
    print(f"[判定] {verdict}")
    print("[备注] 凹槽条带 Gi* 显著性为统计事实，不受本次判读影响")

    # ===== 6. 双柱状图 =====
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(2); w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, [m_exp[0], m_ctl[0]], w, label="生活类密度", color="#4C89C8")
    b2 = ax.bar(x + w/2, [m_exp[1], m_ctl[1]], w, label="工业类密度", color="#D9822B")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["实验组\n凹槽条带13格", "对照组\n城区带格"])
    ax.set_ylabel(r"POI 密度（个/km$^2$）")
    ax.set_title("生活类 vs 工业类 POI 密度")
    ax.legend()

    ax = axes[1]
    bars = ax.bar(["实验组", "对照组"], [m_exp[2], m_ctl[2]],
                  color=["#B0413E", "#7A9E7E"], width=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=10)
    ax.axhline(m_exp[2] / 1.5, ls="--", lw=1, c="gray")
    ax.text(0.98, m_exp[2] / 1.5, "1.5x 阈值", ha="right", va="bottom",
            fontsize=8, color="gray", transform=ax.get_yaxis_transform())
    ax.set_ylabel("生活 / 工业比")
    ax.set_title(f"比值倍数 {ratio_ratio:.2f}x\n{verdict.split(' → ')[0]}")
    fig.suptitle("凹槽假说替代验证：POI 品类结构（Day14，高德类目全量口径）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG_OUT, dpi=150)
    print(f"\n💾 图已保存 → {PNG_OUT}")


if __name__ == "__main__":
    main()
