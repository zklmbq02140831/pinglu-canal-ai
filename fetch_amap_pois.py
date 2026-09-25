# -*- coding: utf-8 -*-
"""
fetch_amap_pois.py
==================
用高德地图 Web 服务 API 抓取南宁市、钦州市的工业类 POI
（工业用地 / 仓库 / 港口），转成 GeoDataFrame，去重后保存为 GeoPackage。

运行：python fetch_amap_pois.py
依赖：requests pandas geopandas shapely
Key ：从 .env 读取 AMAP_KEY（本项目根没有 .env 时会自动找上级目录的 .env）

⚠️ 坐标系说明：高德返回的是 GCJ-02（火星坐标），不是 WGS-84。
   后续与 OSM 路网 / SRTM DEM 等 WGS-84 数据叠加前需要先纠偏。
"""

# ========== 1. 导入依赖 ==========
import os.path      # 路径拼接 / 建目录
import os           # 读环境变量 AMAP_KEY 兜底
import time         # 控制请求间隔，避免触发高德 QPS 限流
from pathlib import Path  # 面向对象路径操作，用来找 .env

import requests     # HTTP 客户端，调用高德 REST API
import pandas as pd     # 组装记录表 / 去重
import geopandas as gpd  # GeoDataFrame 构建与 GeoPackage 写出
from shapely.geometry import Point  # 备用：手动构造点几何（points_from_xy 已覆盖）


# ========== 2. 读取 .env 拿 AMAP_KEY ==========
def find_key(name: str = "AMAP_KEY"):
    """按优先级查找 .env 文件并取出指定变量，找不到再看系统环境变量。"""
    # 候选 .env 路径：脚本所在目录 -> 脚本上级目录（d:/112/yunhe）
    for env_path in (Path(__file__).resolve().parent / ".env",
                     Path(__file__).resolve().parent.parent / ".env"):
        if env_path.exists():                          # 文件存在才解析
            # utf-8-sig 兼容 Windows 记事本保存时加的 BOM 头
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()                    # 去首尾空白
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)          # 只按第一个 = 切，防止值里有 =
                    if k.strip() == name:
                        return v.strip().strip("'\"")  # 去掉可能的引号和空格
    # .env 里没有，最后退回系统环境变量
    return os.environ.get(name)


AMAP_KEY = find_key("AMAP_KEY")
if not AMAP_KEY:
    # 没有 key 直接退出，提示用户配置位置
    raise SystemExit("未找到 AMAP_KEY：请在项目根 .env 或上级目录 .env 中配置")

# ========== 3. 常量配置 ==========
# 高德「关键字搜索」POI 接口（v3）
API_URL = "https://restapi.amap.com/v3/place/text"

# 城市 -> 高德行政区划代码 adcode（用 adcode 比中文名更精确，无重名歧义）
CITY_CODES = {
    "南宁市": "450100",
    "钦州市": "450700",
}

# 业务类别 -> 高德 POI 分类编码 typecode
# ⚠️ 编码按真实分类表修正（用关键词反查实测验证，2026-09）：
#    140100 实为「博物馆」、150500 实为「地铁站」、120202 实为「工业大厦建筑物」
TYPE_CODES = {
    "工业用地": "120100",  # 商务住宅 → 产业园区（工业园区都挂在这）
    "仓库":    "070501",  # 生活服务 → 物流速递 → 物流仓储场地
    "港口":    "150300",  # 交通设施服务 → 港口码头（含车渡口/货运码头子类）
}

PAGE_SIZE = 25      # 每页条数（高德 v3 接口 offset 上限就是 25）
SLEEP = 0.35        # 每次请求间隔秒数：个人 key 限 3 QPS，留余量防限流

# 输出文件（相对项目根），保持用户指定的文件名 pois_osm.gpkg
OUT_PATH = os.path.join("data", "raw", "pois_osm.gpkg")


# ========== 4. 单城市单类别抓取函数（自动翻页） ==========
def fetch_city_type(city_adcode: str, typecode: str) -> list:
    """抓取 指定 adcode 城市 + 指定 typecode 的全部 POI，返回 dict 列表。"""
    pois = []                       # 累积所有页的 POI 记录
    page = 1                        # 从第 1 页开始翻
    while True:
        # 组装请求参数：types 按 typecode 精确过滤
        params = {
            "key": AMAP_KEY,        # 开发者 key
            "types": typecode,      # 分类编码，如 140100
            "city": city_adcode,    # 城市 adcode，如 450100
            "citylimit": "true",    # 严格限制在城市边界内，不混入邻近城市
            "offset": PAGE_SIZE,    # 每页条数
            "page": page,           # 当前页号
            "extensions": "base",   # base=基础字段（含 location 坐标），够用
        }
        # 发 GET 请求，超时 15 秒
        resp = requests.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()     # HTTP 层报错（4xx/5xx）直接抛异常
        data = resp.json()          # 解析 JSON 响应

        # 高德用 status=1 表示业务成功，否则 info/infocode 给出错误码
        if data.get("status") != "1":
            raise RuntimeError(
                f"高德返回错误: {data.get('info')} (infocode={data.get('infocode')})"
            )

        batch = data.get("pois") or []   # 本页 POI 列表（可能为空）
        if not batch:                    # 空页说明已经翻到底
            break
        pois.extend(batch)               # 收进结果

        # count 是该查询命中的总数；够数或翻到第 100 页（高德上限）就停
        total = int(data.get("count") or 0)
        if len(pois) >= total or page >= 100:
            break
        page += 1                        # 继续翻下一页
        time.sleep(SLEEP)                # 限速，避免 QPS 超限
    return pois


# ========== 5. 主流程：逐城市逐类别抓取并整理成记录 ==========
records = []                            # 汇总所有扁平化后的记录
for city_name, adcode in CITY_CODES.items():
    for cat_name, tc in TYPE_CODES.items():
        print(f"⏳ 抓取 {city_name} · {cat_name} (typecode={tc}) ...")
        try:
            batch = fetch_city_type(adcode, tc)   # 调上面的翻页抓取
        except Exception as e:                    # 单个类别失败不影响其他类别
            print(f"   ✗ 失败: {e}")
            continue

        print(f"   ✓ {len(batch)} 条")
        for poi in batch:              # 把高德 JSON 拍平成一行 dict
            loc = poi.get("location") or ""       # 形如 "108.36,22.81" 的字符串
            if "," not in loc:                    # 没坐标的记录（极少）丢弃
                continue
            lng, lat = loc.split(",")[:2]         # 拆出经度、纬度
            records.append({
                "id":          poi.get("id"),        # 高德 POI 唯一 ID，去重键
                "name":        poi.get("name"),      # POI 名称
                "category":    cat_name,             # 我们的业务分类标签
                "typecode":    poi.get("typecode"),  # 高德原始分类编码
                "address":     poi.get("address"),   # 地址
                "province":    poi.get("pname"),     # 省名
                "city":        poi.get("cityname"),  # 城市名
                "district":    poi.get("adname"),    # 区县名
                "source_city": city_name,            # 来源城市（我们查询用的）
                "lng":         float(lng),           # 经度（GCJ-02）
                "lat":         float(lat),           # 纬度（GCJ-02）
            })
        time.sleep(SLEEP)              # 类别之间也限速

# ========== 6. 转 GeoDataFrame + 去重 ==========
if not records:
    # 一条都没抓到就别继续了，大概率是 key/typecode/网络问题
    raise SystemExit("没有抓到任何 POI，请检查 AMAP_KEY 与 typecode 配置")

df = pd.DataFrame(records)             # dict 列表 -> 表

# pandas -> GeoPandas：把经纬度两列转成 Point 几何列，坐标系声明为 EPSG:4326
# 注意：高德实际是 GCJ-02 偏移坐标，这里声明 4326 仅为写出格式统一
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lng"], df["lat"]),
    crs="EPSG:4326",
).drop(columns=["lng", "lat"])         # 原始两列坐标已变成 geometry，删掉冗余列

before = len(gdf)                      # 记录去重前行数
gdf = gdf.drop_duplicates(subset="id") # 按高德 POI id 去重（跨类别重复的情况）
gdf = gdf.reset_index(drop=True)       # 重置索引，写文件更干净
print(f"\n去重：{before} → {len(gdf)}（按高德 POI id）")

# ========== 7. 保存 GeoPackage ==========
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)   # 确保 data/raw 存在
# driver="GPKG" 写 GeoPackage；layer 指定表名；mode 默认 "w" 覆盖旧 layer
gdf.to_file(OUT_PATH, driver="GPKG", layer="industrial_pois")
print(f"💾 已保存到 {OUT_PATH}")

# ========== 8. 打印汇总：总数 + 前 5 行 ==========
print(f"\n==== POI 汇总 ====")
print(f"POI 总数   : {len(gdf)}")
print(f"分类分布   : {gdf['category'].value_counts().to_dict()}")
print(f"城市分布   : {gdf['source_city'].value_counts().to_dict()}")
print(f"\n前 5 行预览：")
# 只挑可读性好的几列打印，避免终端刷屏
print(gdf[["name", "category", "typecode", "district", "source_city", "geometry"]].head())
