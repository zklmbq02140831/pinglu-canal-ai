# -*- coding: utf-8 -*-
"""
day14c_moat_keycat.py —— Day14晚 战时额度管理:C方案补齐 ctl 侧三关键类
=======================================================================
背景:高德月配额战时管理。本脚本补齐对照组(29格)侧三个关键类目
12住宅/14科教文化/17公司企业,消除"exp-bbox重叠区下限"标记,然后零网络重算。

补丁(抓取层升级,第1步验收项):
  ①错误码分级:
    QPS族  {10004,10014,10015,10019,10020,10021} → 指数退避重试(1,2,4,8,16s,至多5次)后仍败=未完成
    配额族 {10003,10044,10029} → 立即停整个任务+当前(类目,子格)不缓存+exit(3)
    网络异常 → 同退避重试;10006=完整空集;其余status!=1 → 未完成不入缓存
  ②checkpoint: 每序列即抓即存 cache;每(组,类目)完成即写 done manifest
  ③done标记: data/raw/moat_done_manifest.json;重算只纳入有done标记的(组,类目)

抓取策略(纪律:同一覆盖C1/C2只允许成功一个;A方案全bbox不执行):
  C1: 29格并集凹多边形逐类抓;并集若为MultiPolygon则按连通片分别抓(仍属C1,无冗余覆盖);
      len≥190 触发 hex集合二分细分(对格子集求并集递归,永不越出29格覆盖);
      单格并集即单六边形环;完成校验 29格内N ≥ 重叠区已知下限(12:45/14:47/17:299),
      不满足视为截断 → 降级C2
  C2: 逐格抓 29格×3类(单六边形环,无顶点上限风险)
  预算硬顶: C1≤100、C2≤200、合计≤260;超顶 exit(5),已完成部分均已落盘
新cache key与旧数据不冲突:
  C1: f"{tc}_U{depth}_{md5}_{part}" ; C2: f"{tc}_G_{hex_id}" ; 旧: f"{tc}_{depth}_{坐标}"
运行: python day14c_moat_keycat.py
"""
import hashlib
import json
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
from shapely.geometry import Point
from shapely.ops import unary_union

from convert_gcj02_to_wgs84 import gcj02_to_wgs84

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "data/raw/moat_pois_cache.json"
MANIFEST_FILE = ROOT / "data/raw/moat_done_manifest.json"
NUMJSON = ROOT / "data/processed/day14_poi_profile.json"
COUNTS_CSV = ROOT / "data/processed/day14_keycat_counts.csv"
PNG_OUT = ROOT / "output/moat_poi_profile.png"

API_URL = "https://restapi.amap.com/v3/place/polygon"
PAGE_SIZE = 25
SLEEP = 0.35
QPS_CODES = {"10004", "10014", "10015", "10019", "10020", "10021"}
QUOTA_CODES = {"10003", "10044", "10029"}
SUSPECT_N = 190
MAX_C1, MAX_C2, MAX_TOTAL = 100, 200, 260

STRIP_IDS = ["874150123ffffff", "87415018bffffff", "874150189ffffff",
             "87415018effffff", "87415018cffffff", "87415018dffffff",
             "87415018affffff", "8741508d2ffffff", "8741508d6ffffff",
             "874150188ffffff", "8741500a4ffffff", "8741508d4ffffff",
             "874150016ffffff"]
KEY_CATS = ["050000", "060000", "070000", "090000", "120000", "140000", "170000"]
TODO_CATS = ["120000", "140000", "170000"]
OVERLAP_FLOOR = {"120000": 45, "140000": 47, "170000": 299}
PORT_LON, PORT_LAT = 108.60, 21.70

req_counter = {"c1": 0, "c2": 0, "total": 0}


class QuotaHalt(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def read_api_key() -> str:
    for p in (Path("d:/112/yunhe/.env"), ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() in ("AMAP_KEY", "AMAP_API_KEY"):
                        return v.strip().strip("'\"")
    return ""


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _burn(kind):
    req_counter[kind] += 1
    req_counter["total"] += 1
    cap = MAX_C1 if kind == "c1" else MAX_C2
    if req_counter[kind] > cap or req_counter["total"] > MAX_TOTAL:
        print(f"⛔ 预算硬顶触顶 {req_counter} → 立即停(已完成部分均已落盘)")
        raise BudgetExceeded


def http_place(key, ring, typecode, page):
    """单次HTTP。网络异常与QPS族均按指数退避重试(至多5次);返回最终json(可能为None)。"""
    d = None
    delays = [1, 2, 4, 8, 16]
    for attempt in range(6):
        try:
            r = requests.get(API_URL, params={
                "key": key, "polygon": ring, "types": typecode,
                "offset": PAGE_SIZE, "page": page, "extensions": "base",
            }, timeout=20)
            d = r.json()
            if str(d.get("infocode") or "") in QPS_CODES and attempt < 5:
                print(f"   ⏳ QPS族退避重试({attempt+1}) infocode={d.get('infocode')}")
                time.sleep(delays[attempt])
                continue
            break
        except requests.RequestException as e:
            if attempt >= 5:
                print(f"   ⚠️ 网络连续失败: {e}")
                break
            time.sleep(delays[attempt])
    return d


def fetch_series(key, ring, typecode, cache, ck, kind):
    """抓一个(多边形,类目)序列。返回 (pois, complete)。仅complete入缓存。"""
    if ck in cache:
        return cache[ck], True
    pois, complete, page = [], False, 1
    while page <= 100:
        _burn(kind)
        d = http_place(key, ring, typecode, page)
        if d is None:
            print(f"   ⚠️ {ck} p{page} 网络失败→未完成,不缓存")
            break
        time.sleep(SLEEP)
        infocode = str(d.get("infocode") or "")
        if infocode in QUOTA_CODES:
            save_json(CACHE_FILE, cache)
            print(f"⛔ 配额族错误 {infocode} @ {ck} p{page} → 立即停,已落盘,明早报告")
            raise QuotaHalt
        if infocode in QPS_CODES:
            print(f"   ⚠️ QPS族 {infocode} 重试耗尽 @ {ck} p{page} → 未完成,不缓存")
            break
        if d.get("status") != "1":
            if infocode == "10006":
                complete = True                       # 该范围无此类目=完整空集
            else:
                print(f"   ⚠️ {ck} p{page}: {d.get('info')}({infocode})")
            break
        batch = d.get("pois") or []
        if not batch:
            complete = True
            break
        pois.extend(batch)
        if len(pois) >= int(d.get("count") or 0):
            complete = True
            break
        page += 1
    if complete:
        cache[ck] = pois
        save_json(CACHE_FILE, cache)
    return pois, complete


def ring_of_geom(geom):
    """单面外环 → 高德polygon串(6位小数,闭合)。顶点>400返回None。"""
    coords = list(geom.exterior.coords)
    if len(coords) > 400:
        return None
    pts = [f"{x:.6f},{y:.6f}" for x, y in coords]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return ";".join(pts)


def union_rings(hex_subset):
    """格集合 → [(ring_str, part_id)];某连通片顶点超限则抛错(禁止静默丢覆盖)。"""
    u = unary_union(list(hex_subset.geometry))
    parts = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    out = []
    for i, p in enumerate(sorted(parts, key=lambda g: g.area, reverse=True)):
        r = ring_of_geom(p)
        if r is None:
            raise ValueError(f"并集连通片{i}顶点超限({len(p.exterior.coords)})")
        out.append((r, i))
    return out


def fetch_union_cat(key, hex_subset, typecode, cache, depth=0, kind="c1"):
    """C1:对格集合并集抓一个类目;len≥190 触发 hex 集合二分细分。"""
    if len(hex_subset) == 1:
        row = hex_subset.iloc[0]
        ck = f"{typecode}_G_{row.hex_id}"
        return fetch_series(key, ring_of_geom(row.geometry), typecode, cache, ck, kind)
    tag = hashlib.md5(",".join(sorted(hex_subset.hex_id)).encode()).hexdigest()[:8]
    ck = f"{typecode}_U{depth}_{tag}"
    if ck in cache:
        return cache[ck], True
    rings = union_rings(hex_subset)                    # 顶点超限在此抛ValueError→降级C2
    merged, ids, all_ok = [], set(), True
    for ring, part_id in rings:
        ck_p = f"{typecode}_U{depth}_{tag}_{part_id}"
        pois, ok = fetch_series(key, ring, typecode, cache, ck_p, kind)
        all_ok = all_ok and ok
        for p in pois:
            if p.get("id") and p["id"] not in ids:
                ids.add(p["id"]); merged.append(p)
    if all_ok and len(merged) >= SUSPECT_N and depth < 5:
        sub = hex_subset.sort_values("cx")
        half = len(sub) // 2
        ma, oka = fetch_union_cat(key, sub.iloc[:half], typecode, cache, depth + 1, kind)
        mb, okb = fetch_union_cat(key, sub.iloc[half:], typecode, cache, depth + 1, kind)
        seen, fin = set(), []
        for p in list(ma) + list(mb):
            if p.get("id") and p["id"] not in seen:
                seen.add(p["id"]); fin.append(p)
        print(f"   ↳ U{depth} {typecode[:2]} 二分细分: {len(merged)} → {len(fin)}")
        cache[ck] = fin
        save_json(CACHE_FILE, cache)
        return fin, (oka and okb)
    if all_ok:
        cache[ck] = merged
        save_json(CACHE_FILE, cache)
    return merged, all_ok


def fetch_pergrid_cat(key, ctl_hexes, typecode, cache):
    """C2:逐格抓。返回 (合并POI, 全部完整?)。与C1共用_G_键,重跑零浪费。"""
    merged, ids, all_ok = [], set(), True
    for _, row in ctl_hexes.iterrows():
        ck = f"{typecode}_G_{row.hex_id}"
        pois, ok = fetch_series(key, ring_of_geom(row.geometry), typecode, cache, ck, "c2")
        all_ok = all_ok and ok
        for p in pois:
            if p.get("id") and p["id"] not in ids:
                ids.add(p["id"]); merged.append(p)
    return merged, all_ok


# ---------- 第3步:全量重算(零网络) ----------
LIFE6 = ("05", "06", "07", "09", "12", "14")


def recalc(hexes, ctl_ids, done):
    cache = load_json(CACHE_FILE, {})
    recs = {}
    for pois in cache.values():
        for p in pois:
            pid = p.get("id")
            if pid and pid not in recs:
                recs[pid] = p
    df = pd.DataFrame([{
        "poi_id": k, "type": v.get("type", ""), "typecode": str(v.get("typecode", "")),
        "lng": float(v["location"].split(",")[0]), "lat": float(v["location"].split(",")[1]),
    } for k, v in recs.items() if "," in (v.get("location") or "")])
    wgs = [gcj02_to_wgs84(a, b) for a, b in zip(df.lng, df.lat)]
    df["lng_w"] = [w[0] for w in wgs]
    df["lat_w"] = [w[1] for w in wgs]
    print(f"POI池(全缓存去重): {len(df)}")

    canal_utm = gpd.read_file(ROOT / "data/raw/canal_centerline.geojson").to_crs("EPSG:32648").geometry.union_all()
    utm = hexes.to_crs("EPSG:32648")
    cent = utm.geometry.centroid
    port_utm = gpd.GeoSeries([Point(PORT_LON, PORT_LAT)], crs="EPSG:4326").to_crs("EPSG:32648").iloc[0]
    hexes["d_canal_km"] = cent.distance(canal_utm).to_numpy() / 1000
    hexes["d_port_km"] = cent.distance(port_utm).to_numpy() / 1000
    area = {"exp": utm[utm.hex_id.isin(STRIP_IDS)].area.sum() / 1e6,
            "ctl": utm[utm.hex_id.isin(ctl_ids)].area.sum() / 1e6}

    gpoi = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng_w, df.lat_w), crs="EPSG:4326")
    j = gpd.sjoin(gpoi, hexes[["hex_id", "geometry"]], how="inner", predicate="within")
    tcc = j["typecode"].str.zfill(6)
    j["tc2"] = tcc.str[:2]
    j["type"] = j["type"].astype(str)

    missing = [f"{s}/{c}" for s in ("exp", "ctl") for c in KEY_CATS
               if not done.get(f"{s}/{c}")]
    if missing:
        print(f"⚠️ 无done标记(不进比值口径): {missing}")

    res, rows = {}, []
    for side, ids_ in (("exp", STRIP_IDS), ("ctl", list(ctl_ids))):
        jj = j[j.hex_id.isin(ids_)]
        gated = lambda c2: done.get(f"{side}/{c2}0000") is not None   # ③done门控
        life = sum(int((jj.tc2 == p).sum()) for p in LIFE6 if gated(p))
        ind_w = int((jj.tc2 == "17").sum()) if gated("17") else 0
        # 注意:GeoDataFrame.type属性返回几何类型("Point"),会遮蔽POI type列——必须bracket访问
        ind_n = int(jj["type"].str.contains("工厂|仓储").sum()) if gated("17") else 0
        a = area[side]
        res[side] = {
            "area_km2": round(a, 2), "life": life, "ind_wide": ind_w, "ind_narrow": ind_n,
            "life_d": life / a, "ind_wide_d": ind_w / a, "ind_narrow_d": ind_n / a,
            "ratio_wide": (life / ind_w) if ind_w else None,
            "ratio_narrow": (life / ind_n) if ind_n else None,
            "total_in_grid": int(len(jj)),
        }
        for c2 in LIFE6:
            rows.append({"side": side, "cat": c2 + "0000", "count": int((jj.tc2 == c2).sum()),
                         "density": round((jj.tc2 == c2).sum() / a, 3),
                         "done": bool(done.get(f"{side}/{c2}0000"))})
        rows.append({"side": side, "cat": "170000", "count": ind_w,
                     "density": round(ind_w / a, 3), "done": bool(gated("17"))})
        rows.append({"side": side, "cat": "narrow_工厂仓储", "count": ind_n,
                     "density": round(ind_n / a, 3), "done": bool(gated("17"))})

    print("\n==== Day14 最终对照表(全量口径,两档工业并列) ====")
    print(pd.DataFrame(rows).pivot_table(index="cat", columns="side", values="count").to_string())
    e, c = res["exp"], res["ctl"]
    for s, r in (("exp", e), ("ctl", c)):
        print(f"[{s}] 面积{r['area_km2']}km² 生活{r['life']}({r['life_d']:.2f}/km²) "
              f"工业宽{r['ind_wide']}({r['ind_wide_d']:.2f}) 窄{r['ind_narrow']}({r['ind_narrow_d']:.2f}) "
              f"比宽{r['ratio_wide'] and round(r['ratio_wide'], 2)} "
              f"比窄{r['ratio_narrow'] and round(r['ratio_narrow'], 2)}")
    if e["ratio_wide"] and c["ratio_wide"]:
        mult = e["ratio_wide"] / c["ratio_wide"]
        print(f"比值倍数: 宽口径 {mult:.2f}x", end="")
        res["mult_wide"] = round(mult, 2)
        if e["ratio_narrow"] and c["ratio_narrow"]:
            mult_n = e["ratio_narrow"] / c["ratio_narrow"]
            print(f" | 窄口径 {mult_n:.2f}x", end="")
            res["mult_narrow"] = round(mult_n, 2)
        else:
            print(" | 窄口径倍数不可算(某侧窄分母为0)", end="")
        print()
    res["missing_done_marks"] = missing
    res["request_cost_tonight"] = dict(req_counter)
    save_json(NUMJSON, res)
    pd.DataFrame(rows).to_csv(COUNTS_CSV, index=False, encoding="utf-8-sig")
    plot(e, c)
    return res


def plot(e, c):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(3); w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, [e["life_d"], e["ind_wide_d"], e["ind_narrow_d"]], w,
                label="实验组(凹槽条带13格)", color="#B0413E")
    b2 = ax.bar(x + w/2, [c["life_d"], c["ind_wide_d"], c["ind_narrow_d"]], w,
                label="对照组(29格)", color="#4C89C8")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["生活类(6类)", "工业宽口径\n(17公司企业)", "工业窄口径\n(工厂/仓储文本)"])
    ax.set_ylabel(r"POI 密度（个/km$^2$）")
    ax.set_title("关键类POI密度对比(全量口径)")
    ax.legend(fontsize=9)
    ax = axes[1]
    mults = [e["ratio_wide"] / c["ratio_wide"] if c["ratio_wide"] else 0,
             e["ratio_narrow"] / c["ratio_narrow"] if c["ratio_narrow"] else 0]
    bars = ax.bar(["宽口径(17类)", "窄口径(工厂/仓储)\n主张口径"], mults,
                  color=["#999999", "#B0413E"], width=0.45)
    ax.bar_label(bars, fmt="%.2fx", fontsize=10)
    ax.axhline(1.5, ls="--", lw=1, c="gray")
    ax.text(0.98, 1.5, "1.5x 阈值", ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_ylabel("生活/工业比 倍数(实验/对照)")
    ax.set_title("比值倍数(假说非结论,见局限)")
    fig.suptitle("凹槽条带 POI 品类画像(Day14 全量口径)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_OUT, dpi=150)
    print(f"💾 图已更新 → {PNG_OUT}")


def main():
    key = read_api_key()
    if not key:
        sys.exit("未找到 AMAP_KEY")
    done = load_json(MANIFEST_FILE, {})
    hexes = gpd.read_file(ROOT / "data/processed/hex_grid.geojson")
    canal_utm = gpd.read_file(ROOT / "data/raw/canal_centerline.geojson").to_crs("EPSG:32648").geometry.union_all()
    utm = hexes.to_crs("EPSG:32648")
    cent = utm.geometry.centroid
    port_utm = gpd.GeoSeries([Point(PORT_LON, PORT_LAT)], crs="EPSG:4326").to_crs("EPSG:32648").iloc[0]
    hexes["d_canal_km"] = cent.distance(canal_utm).to_numpy() / 1000
    hexes["d_port_km"] = cent.distance(port_utm).to_numpy() / 1000
    ctl = hexes[(hexes.d_port_km.between(20, 30)) & (hexes.d_canal_km > 3)].copy()
    ctl["cx"] = ctl.geometry.centroid.x
    print(f"[分组] ctl {len(ctl)} 格 | 历史done标记 {len(done)} 项")

    for cat in KEY_CATS:
        done.setdefault(f"exp/{cat}", "bbox_full")      # 历史全bbox抓取已完成(日志零⚠️)
        if cat in ("050000", "060000", "070000", "090000"):
            done.setdefault(f"ctl/{cat}", "bbox_full")
    save_json(MANIFEST_FILE, done)

    cache = load_json(CACHE_FILE, {})
    for cat in TODO_CATS:
        if done.get(f"ctl/{cat}"):
            print(f"[跳过] ctl/{cat} 已有done标记: {done[f'ctl/{cat}']}")
            continue
        poly29 = unary_union(list(ctl.geometry))
        mode = None
        try:
            pois, ok = fetch_union_cat(key, ctl, cat, cache, 0, "c1")
            if not ok:
                raise RuntimeError("C1子序列不完整")
            got_in = 0
            for p in pois:
                loc = p.get("location") or ""
                if "," not in loc:
                    continue
                x, y = gcj02_to_wgs84(*[float(v) for v in loc.split(",")[:2]])
                if poly29.covers(Point(x, y)):
                    got_in += 1
            if got_in < OVERLAP_FLOOR[cat]:
                raise RuntimeError(f"C1疑似截断: 29格内{got_in} < 重叠区下限{OVERLAP_FLOOR[cat]}")
            mode = f"C1(29格内{got_in})"
        except (RuntimeError, ValueError) as err:
            print(f"   ⚠️ ctl/{cat} C1失败({err}) → 降级C2逐格抓")
            pois, ok = fetch_pergrid_cat(key, ctl, cat, cache)
            if not ok:
                print(f"⛔ ctl/{cat} C2亦不完整 → 停,明早报告")
                save_json(MANIFEST_FILE, done)
                sys.exit(4)
            mode = "C2"
        done[f"ctl/{cat}"] = mode
        save_json(MANIFEST_FILE, done)
        print(f"[完成] ctl/{cat} via {mode} | 累计请求 {dict(req_counter)}")
    print(f"[C抓取结束] 请求消耗 {dict(req_counter)}")

    recalc(hexes, ctl.hex_id.tolist(), done)


if __name__ == "__main__":
    try:
        main()
    except QuotaHalt:
        sys.exit(3)
    except BudgetExceeded:
        sys.exit(5)
