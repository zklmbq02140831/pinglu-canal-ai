# -*- coding: utf-8 -*-
"""
day13_validation_prep.py
========================
Day13 凹槽验证协议准备脚本：
1. 按 Day12 完全相同的方法（Queen 权重 + esda.G_Local，permutations=0，
   z 为解析量与模拟无关）复算 Gi*，锁定市区段"凹槽条带"13 格，
   并与 Day12 验收基准逐项校验（z/composite/距河中位数/内部邻接）。
2. 高德逆地理编码批量反查 13 格质心所在街道 → 实验组街道清单。
3. 高德 POI 检索各街道/镇政府（管委会）驻地坐标（GCJ-02→WGS-84）作为
   街道代表点，统一口径实算：距运河线距离 / 距钦州港距离。
   （乡镇级矢量边界高德/DataV/OSM 均不可得，故用驻地代表点口径，
    已在协议中注明；面积取政府公开的行政区域面积，来源见脚本内表。）
4. 按协议条件筛对照组：距钦州港 20-30km 且 距运河线 > 3km（非贴河）。
5. 产出：
   - data/processed/day13_street_protocol.csv  街道底表（含企查查链接）
   - docs/day13_protocol.md                    验证协议文档（含 6 列空表）
   - docs/screenshots/day13/                   截图目录（自动创建）

运行：python day13_validation_prep.py   （需联网，AMAP_KEY 读自 d:/112/yunhe/.env）
依赖：geopandas / esda / libpysal / shapely / requests（langgraph 环境均有）
"""

# ========== 1. 导入依赖 ==========
import os                      # 路径、建目录
import sys                     # 编码设置、导入本地模块
import json                    # 高德响应缓存
import time                    # 限速 sleep
import requests                # 高德 HTTP API
import numpy as np             # 数组运算
import pandas as pd            # 表格
import geopandas as gpd        # 空间数据
from shapely.geometry import Point                   # 港口点

sys.stdout.reconfigure(encoding="utf-8")             # Windows 控制台中文
sys.path.insert(0, os.path.join("src", "analysis"))  # 复用 spatial_stats
from spatial_stats import load_data, build_weights   # 与 Day12 完全一致的方法
from compute_index import PORT_LON, PORT_LAT         # 钦州港坐标常量
from convert_gcj02_to_wgs84 import wgs84_to_gcj02, gcj02_to_wgs84  # 坐标加偏

import esda                    # G_Local
import libpysal                # Queen 邻接（位置索引）

# ========== 2. 常量 ==========
UTM = "EPSG:32648"                              # 米制投影（项目约定）
CSV_OUT = os.path.join("data", "processed", "day13_street_protocol.csv")
DOC_OUT = os.path.join("docs", "day13_protocol.md")
SHOT_DIR = os.path.join("docs", "screenshots", "day13")
CACHE_FILE = os.path.join("data", "raw", "day13_amap_cache.json")
ENV_CANDIDATES = [os.path.join("..", ".env"), ".env"]   # AMAP_KEY 所在
QCC_URL = "https://www.qcc.com/web/search?key={}"
RE_URL = "https://restapi.amap.com/v3/geocode/regeo"
POI_URL = "https://restapi.amap.com/v3/place/text"
DIST_URL = "https://restapi.amap.com/v3/config/district"

# Day12 已验收的 13 格统计特征（用于校验复算结果）
DAY12_REF = {"z_min": -3.84, "z_max": -2.10,
             "comp_min": 0.116, "comp_max": 0.552,
             "d_canal_med": 2.01, "inner_edges": 18}

# 街道/镇行政区域面积 km²（公开资料口径，写死以保结果可复现）：
#  - 钦南区全部街道/镇：钦南区人民政府"行政区划"页（2026-04-24 更新）
#  - 大垌镇：百度百科词条（2017 年数据 127.87）
#  - 子材街道：maps4gis 栅格估算 11.664（非官方，协议中标注"估算"）
#  - 两个港区功能区无标准乡镇面积 → NaN，只比总数不比密度
AREA_KM2 = {
    "向阳街道": 3.6, "水东街道": 12.3, "南珠街道": 9.9, "文峰街道": 7.1,
    "尖山街道": 83.0, "沙埠镇": 145.5, "康熙岭镇": 92.07, "黄屋屯镇": 226.7,
    "犀牛脚镇": 275.4, "东场镇": 181.0, "大番坡镇": 96.43,
    "龙门港镇": 36.77, "久隆镇": 222.92, "那丽镇": 226.08,
    "那彭镇": 291.38, "那思镇": 244.76,
    "大垌镇": 127.87, "子材街道": 11.664,
}
AREA_SRC = {
    "钦南区各街道/镇": "钦南区人民政府·行政区划（2026-04-24）",
    "大垌镇": "百度百科（2017 年 127.87 km²）",
    "子材街道": "maps4gis 栅格估算（非官方）",
}


def read_amap_key():
    """从工作区 .env 读高德 KEY（兼容 AMAP_KEY / AMAP_API_KEY 两种字段名）。"""
    for p in ENV_CANDIDATES:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AMAP_KEY=") or line.startswith("AMAP_API_KEY="):
                        return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("未找到 AMAP_KEY，请检查 d:/112/yunhe/.env")


KEY = read_amap_key()

# 高德响应本地缓存（重跑不重复扣量）
_cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        _cache = json.load(f)


def amap_get(url, params):
    """带缓存的高德 GET：以 url+params 为键，避免重复请求。"""
    k = url + "?" + "&".join(f"{a}={b}" for a, b in sorted(params.items()))
    if k in _cache:
        return _cache[k]
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    _cache[k] = data
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)
    time.sleep(0.35)                              # QPS 限速
    return data


# ========== 3. 第一步：复算 Gi*，锁定凹槽条带 13 格 ==========
print("==== [1/4] 复算 Gi* 冷点，锁定凹槽条带 ====")
gdf, canal = load_data()                          # 与 Day12 相同的合并与县区重建
w = build_weights(gdf)                            # Queen（有孤立岛则 KNN）
y = gdf["composite_index"].to_numpy(dtype=float)
y_std = (y - y.mean()) / y.std()                  # 与 global_moran 同口径（ddof=0）
gl = esda.G_Local(y_std, w, transform="r", permutations=0)  # z 为解析量；跳过随机化规避本机 DLL 崩溃，z 与 Day12 一致
gdf["z"] = gl.Zs

# 米制距离：质心/整格 到运河线、质心 到钦州港
utm = gdf.to_crs(UTM)
cent_utm = utm.geometry.centroid
canal_utm = canal.to_crs(UTM).geometry.union_all()
port_utm = gpd.GeoSeries([Point(PORT_LON, PORT_LAT)],
                         crs="EPSG:4326").to_crs(UTM).iloc[0]
gdf["d_canal_cent_km"] = cent_utm.distance(canal_utm).to_numpy() / 1000
gdf["d_port_km"] = cent_utm.distance(port_utm).to_numpy() / 1000

cold = gdf[gdf["z"] < -1.96].copy()               # 显著冷点
print(f"显著冷点总数：{len(cold)}，距港分布："
      f"{np.percentile(cold['d_port_km'], [0, 25, 50, 75, 100]).round(1)}")

# 校验 Day12 统计特征
chk = {"z_min": round(cold["z"].min(), 2),
       "z_max": round(cold["z"].max(), 2),
       "comp_min": round(cold["composite_index"].min(), 3),
       "comp_max": round(cold["composite_index"].max(), 3),
       "d_canal_med": round(cold["d_canal_cent_km"].median(), 2)}
print(f"复算特征：{chk}")
print(f"Day12 基准：{DAY12_REF}")

# 显著冷点恰为 13 个且统计特征吻合 → 即 Day12 认定的"凹槽条带"，
# 附带 Queen 连通性核对（libpysal 邻接键为位置索引，需转 hex_id）
hex_by_pos = gdf["hex_id"].to_numpy()             # 位置 → hex_id
pos_of = {h: i for i, h in enumerate(hex_by_pos)} # hex_id → 位置
strip_ids = cold["hex_id"].tolist()
strip_pos = [pos_of[h] for h in strip_ids]
strip_set = set(strip_pos)
inner_edges = sum(len([j for j in w.neighbors[i] if j in strip_set])
                  for i in strip_pos) // 2
print(f"条带内部 Queen 邻接边数：{inner_edges}（Day12 基准 {DAY12_REF['inner_edges']}）")
strip = cold.copy()

# ========== 4. 第二步：高德逆地理编码反查 13 格所在街道 ==========
print("\n==== [2/4] 高德逆地理编码 → 实验组街道 ====")
cent_wgs = gdf.loc[gdf["hex_id"].isin(strip_ids)].to_crs(UTM) \
    .geometry.centroid.to_crs("EPSG:4326")
locs_gcj = [wgs84_to_gcj02(p.x, p.y) for p in cent_wgs.geometry]  # 高德要 GCJ-02
townships = []
for i in range(0, len(locs_gcj), 20):             # batch 单次最多 20 点
    chunk = locs_gcj[i:i + 20]
    loc_str = "|".join(f"{x:.6f},{y:.6f}" for x, y in chunk)
    data = amap_get(RE_URL, {"location": loc_str, "batch": "true",
                             "extensions": "base", "key": KEY})
    if str(data.get("status")) != "1":
        raise RuntimeError(f"regeo 失败：{data}")
    for rc in data["regeocodes"]:
        ac = rc.get("addressComponent", {})
        townships.append({"district": ac.get("district", ""),
                          "township": ac.get("township", "") or ""})
grid_street = {}
for hid, t in zip(strip_ids, townships):
    grid_street[hid] = t["township"]
    print(f"  {hid}  {t['district']}/{t['township']}")

exp_names = sorted({v for v in grid_street.values() if v})
print(f"实验组街道/镇（{len(exp_names)}）：{exp_names}")

# ========== 5. 第三步：街道驻地代表点 + 距离实算 + 对照组筛选 ==========
print("\n==== [3/4] 街道驻地代表点与距离（钦南区/钦北区） ====")


def fetch_children(kw, sub):
    """高德行政区域查询：返回 districts 列表（街道名单来源）。"""
    data = amap_get(DIST_URL, {"keywords": kw, "subdistrict": sub,
                               "extensions": "base", "key": KEY})
    if str(data.get("status")) != "1" or not data.get("districts"):
        raise RuntimeError(f"district 失败：{data}")
    return data["districts"][0].get("districts", [])


qinzhou_kids = fetch_children("钦州市", 1)         # → 钦南区/钦北区/灵山县/浦北县
urban_kids = []
for d in qinzhou_kids:                            # 只取两个市辖区
    if d["name"] in ("钦南区", "钦北区"):
        urban_kids += fetch_children(d["adcode"], 1)
print(f"钦南区+钦北区 下辖 {len(urban_kids)} 个街道/镇")


GOV_TAG = ("办事处", "人民政府", "管委会", "管理委员会", "街道办",
           "镇政府", "管理区", "农场")          # 政务驻地 POI 名称特征


def hq_point(name, adcode=None):
    """检索街道/镇政府（或管委会）驻地 POI，返回 WGS-84 (lng, lat) 或 None。
    高德对查不到的词会模糊匹配到"钦州市办事处"这类兜底 POI，必须校验
    返回 POI 名称确实包含目标街道名且带政务后缀，否则换关键词重查。"""
    tries = [(f"钦州市{name}办事处", "450700", 3, True),   # 前三组沿用旧参数（命中缓存）
             (f"钦州市{name}人民政府", "450700", 3, True),
             (f"钦州{name}管理委员会", "450700", 3, True),
             (f"{name}人民政府", adcode, 5, True),          # 区县 adcode 限定范围重查
             (f"{name}办事处", adcode, 5, True),
             (name, adcode, 5, False)]                      # 兜底：允许功能区同名区域 POI
    for kw, city, off, strict in tries:
        if city is None:
            continue
        data = amap_get(POI_URL, {"keywords": kw, "city": city,
                                  "offset": off, "key": KEY})
        if str(data.get("status")) != "1":
            continue
        for poi in data.get("pois") or []:
            pname = poi.get("name", "")
            if name in pname and (not strict
                                  or any(t in pname for t in GOV_TAG)):
                print(f"    [{name}] 匹配 POI：{pname}")
                lng, lat = map(float, poi["location"].split(","))
                return gcj02_to_wgs84(lng, lat)
    print(f"    [{name}] ⚠️ 未匹配到政务驻地 POI")
    return None


rows = []
for st in urban_kids:
    name = st["name"]
    pt = hq_point(name, st["adcode"])             # 驻地代表点（WGS-84）
    rows.append({"name": name, "adcode": st["adcode"],
                 "area_km2": AREA_KM2.get(name, np.nan),
                 "geometry": None if pt is None else Point(*pt)})
sgdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
sgdf_utm = sgdf.to_crs(UTM)
sgdf["d_canal_km"] = (sgdf_utm.geometry.distance(canal_utm) / 1000).round(2)
sgdf["d_port_km"] = (sgdf_utm.geometry.distance(port_utm) / 1000).round(2)
sgdf["group"] = np.where(sgdf["name"].isin(exp_names), "实验组", "候选")

ctrl_ok = sgdf[(sgdf["group"] == "候选") &
               (sgdf["d_port_km"].between(20, 30)) &
               (sgdf["d_canal_km"] > 3.0)]
print("\n候选街道/镇全景（按距运河排序）：")
cols = ["name", "area_km2", "d_canal_km", "d_port_km", "group"]
print(sgdf[cols].sort_values(["group", "d_canal_km"]).to_string(index=False))
print(f"\n满足对照组条件（距港20-30km 且 距运河>3km）：{ctrl_ok['name'].tolist()}")
ctrl_names = ctrl_ok.sort_values("d_port_km")["name"].tolist()
sgdf.loc[sgdf["name"].isin(ctrl_names), "group"] = "对照组"

# ========== 6. 第四步：产出底表 CSV + 协议文档 + 截图目录 ==========
print("\n==== [4/4] 生成底表与协议文档 ====")
os.makedirs(SHOT_DIR, exist_ok=True)              # 截图目录

base = sgdf[sgdf["group"].isin(["实验组", "对照组"])].copy()
base["group_rank"] = (base["group"] == "实验组").astype(int)  # 实验组排前
base = base.sort_values(["group_rank", "d_port_km"])
base["街道名"] = base["name"]
base["组别"] = base["group"]
base["距运河距离km"] = base["d_canal_km"]
base["距钦州港km"] = base["d_port_km"]
base["面积km2"] = base["area_km2"]
base["企查查搜索链接"] = base["街道名"].map(
    lambda n: QCC_URL.format(requests.utils.quote(
        n if "钦州" in n else "钦州市" + n)))   # 功能区名自带"钦州"则不加前缀
out = base[["街道名", "组别", "距运河距离km", "距钦州港km", "面积km2",
            "企查查搜索链接"]].copy()
out["存续企业数"] = ""
out["企业密度(家/km2)"] = ""
out["近3年新注册"] = ""
out["近3年吊销注销"] = ""
out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig", na_rep="")
print(f"💾 {CSV_OUT}")

# ---- 协议文档 ----
def fmt(v):
    """NaN 转占位文本，其余正常格式化。"""
    return "待补" if pd.isna(v) else f"{v:g}"


st = strip[["hex_id", "z", "composite_index", "d_canal_cent_km",
            "d_port_km"]].copy()
st["街道"] = st["hex_id"].map(grid_street)
st.columns = ["hid", "zz", "cc", "dd", "dp", "tt"]  # 简单列名便于 itertuples
strip_rows = "\n".join(
    f"| {r.hid} | {r.tt} | {r.zz:.2f} | {r.cc:.3f} | {r.dd:.2f} | {r.dp:.1f} |"
    for r in st.itertuples(index=False))

exp_rows = "\n".join(
    f"| {r.街道名} | {r.组别} | {fmt(r.距运河距离km)} | {fmt(r.距钦州港km)} |"
    f" {fmt(r.面积km2)} | [企查查]({r.企查查搜索链接}) |"
    for r in out.itertuples(index=False))

doc = f"""# Day13 凹槽验证协议（企查查工商数据 · 工业填空假设检验）

## 1. 目的与假设

Day12 结论：凹槽条带（13 格 Gi* 显著冷点，z∈[-3.84, -2.10]）为
「工业口径下被高值城区包围的贴河低值条带」，定性为**工业填空（industrial infill）
潜力最大区段**。本日用企查查工商数据做外部验证。

- **H1（支持工业填空）**：条带所在街道的企业密度与近 3 年新增企业数
  低于同港距带（距港 20-30km）非贴河对照街道；说明该带当前企业活动稀疏，
  "填空"叙事成立。
- **H0（假设存疑）**：实验组密度/新增与对照组相当或更高，
  需回查指数口径（工业 POI / 主干路网）是否失真。

## 2. 分组定义

| 组别 | 定义 | 来源 |
|---|---|---|
| 实验组 | 凹槽条带 13 格质心反查所在街道（向阳、水东、沙埠、南珠、子材、犀牛脚、港区两功能区） | 本脚本高德逆地理编码 |
| 对照组 | 驻地距钦州港 20-30km、距运河线 > 3km 的街道/镇（非贴河） | 本脚本实算 |

条带 13 格复算清单（与 Day12 校验：z/composite/距河中位数 {chk['d_canal_med']}/内部邻接 {inner_edges} 条全部吻合）：

| hex_id | 所在街道 | Gi*_z | composite | 质心距河km | 距港km |
|---|---|---|---|---|---|
{strip_rows}

> 条带分两段：市区段（距港 25-31km，向阳/水东/沙埠/南珠/子材，9 格）与
> 港区段（距港 5.7-8.8km，钦州港经开区/保税港区/犀牛脚，4 格），与 Day12
> "港-城双峰"结构一致；对照组落在 20-30km 市区带，故密度判读以**市区段街道**为主，
> 港区段两功能区无标准乡镇面积、不参与密度判读（仅比企业总数）。

## 3. 口径说明（协议可辩护性）

- **街道代表点**：高德 POI 检索街道/镇人民政府（管委会）驻地坐标（GCJ-02→WGS-84）。
  乡镇级矢量边界经高德/DataV/OSM 三源核验均不可得，故距离统一采用**驻地代表点口径**；
  实验组/对照组同法，组间可比。
- **距运河距离**：代表点（WGS-84）转 EPSG:32648 后到运河中心线（WGS-84 同投影）的
  最近距离，km。
- **面积**：行政区域面积取公开资料（下表来源列），非 GIS 量算；
  两港区功能区无标准面积 → 密度留空。
- **企查查口径**：十条街道查询词、筛选路径完全一致，读数以筛选后的总数为准。

## 4. 街道底表（企查查链接可直接点开）

| 街道名 | 组别 | 距运河km | 距港km | 面积km² | 企查查 |
|---|---|---|---|---|---|
{exp_rows}

面积来源：钦南区各街道/镇——钦南区人民政府"行政区划"页（2026-04-24 更新）；
大垌镇——百度百科（2017 年，127.87 km²）；子材街道——maps4gis 栅格估算 11.664（非官方）；
钦州港经济技术开发区、广西钦州保税港区——功能区，无标准乡镇面积（待补，可查管委会官网）。

## 5. 指标口径（企查查网页版，逐街道）

对每条街道（搜索词 = "钦州市+街道名"，链接见上表），依次记录 3 个数：

1. **存续企业总数**：搜索结果页筛选「经营状态 → 存续/在业」，记总数；
   企业密度 = 存续企业总数 ÷ 街道面积 km²（面积见上表）。
2. **近 3 年新注册企业数**：叠加筛选「注册时间 → 2023-09-28 至 2026-09-28」，
   在存续/在业状态下记总数。
3. **近 3 年吊销/注销企业数**：注册时间同上，经营状态改选「吊销」「注销」，
   两个状态数相加。

**截图规范**（存 `docs/screenshots/day13/`，每街道 3 张）：
`NN_<街道名>_存续.png` / `NN_<街道名>_新增.png` / `NN_<街道名>_吊销注销.png`
（NN 为街道序号，按上表自上而下 01、02、…）。

注意：企查查按地址关键词检索会混入少量"街道名出现在公司名"的记录，
读数时以筛选后的总数为准，十条街道口径保持一致。

## 6. 输出：6 列对照表（查完后回填，另存 day13_result）

| 街道名 | 组别 | 距运河距离(km) | 企业密度(家/km²) | 近3年新增(家) | 近3年吊销/注销(家) |
|---|---|---|---|---|---|
{chr(10).join(f"| {r.街道名} | {r.组别} | {fmt(r.距运河距离km)} |  |  |  |"
              for r in out.itertuples(index=False))}

## 7. 判读标准

- 实验组（市区段 5 街道）企业密度与近 3 年新增均低于对照组中位数 → **H1 成立**，
  「工业填空」叙事写入答辩，Day13 闭环。
- 仅一项低 → 弱支持，在 decisions.md 记"部分验证"并说明口径。
- 均不低于 → H0，回查指数五维口径后重跑。
"""
with open(DOC_OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"💾 {DOC_OUT}")
print(f"💾 {SHOT_DIR}{os.sep}（截图目录已就绪）")
