# -*- coding: utf-8 -*-
"""
clean_pois.py
=============
清洗高德 POI 数据（WGS-84 版）：
1. 读 data/raw/pois_amap_wgs84.csv
2. 按 type 字段包含关键词（物流/仓储/工业园/产业园/港口/码头/货运/工业）过滤
3. 剔除 name 里明显无关的记录（餐饮/火锅/饭店等）
4. 转 GeoDataFrame（EPSG:4326）保存为 data/processed/pois_clean.geojson
5. 打印清洗前后数据量对比

运行：python clean_pois.py
依赖：pandas / geopandas
"""

# ========== 1. 导入依赖 ==========
import os                        # 路径拼接 / 建目录

import pandas as pd              # 读 CSV、字符串过滤
import geopandas as gpd          # 构造 GeoDataFrame、写 GeoJSON
from shapely.geometry import Point  # 把经纬度变成点几何

# 输入：已纠偏为 WGS-84 的 POI 表；输出：清洗后的 GeoJSON
IN_PATH = os.path.join("data", "raw", "pois_amap_wgs84.csv")
OUT_PATH = os.path.join("data", "processed", "pois_clean.geojson")

# type 字段（高德分类链，如「交通设施服务;港口码头;港口」）包含任一关键词即保留
TYPE_KEYWORDS = ["物流", "仓储", "工业园", "产业园", "港口", "码头", "货运", "工业"]

# name 里出现即剔除的无关词（POI 名称层面的噪音，如园区里挂着招牌的餐馆）
NAME_BLACKLIST = ["餐饮", "火锅", "饭店", "美食", "烧烤", "小吃", "奶茶", "便利店"]


# ========== 2. 读数据 ==========
df = pd.read_csv(IN_PATH, encoding="utf-8-sig")     # 读入 WGS-84 版 POI 表
n_raw = len(df)                                     # 记下清洗前的总量
print(f"读入 {n_raw} 条 POI：{IN_PATH}")

# type 列可能有缺失值（NaN），fillna("") 先补成空串避免报错
df["type"] = df["type"].fillna("")

# ========== 3. 按 type 关键词过滤（白名单） ==========
# 对每个关键词判断 type 是否包含它；任意一个命中就保留该行
mask_type = df["type"].apply(
    lambda t: any(kw in t for kw in TYPE_KEYWORDS)
)
df_type = df[mask_type]                             # 应用布尔掩码得到过滤结果
n_type = len(df_type)                               # 记录这一步后的数量
print(f"① type 含工业/物流/港口关键词: {n_raw} -> {n_type} 条 "
      f"(剔除 {n_raw - n_type} 条)")

# ========== 4. 剔除 name 明显无关的记录 ==========
# name 同样补空串；保留 name 不含任何黑名单词的行
mask_name = ~df_type["name"].fillna("").apply(
    lambda n: any(bad in n for bad in NAME_BLACKLIST)
)
df_clean = df_type[mask_name]                       # 得到最终清洗结果
n_clean = len(df_clean)                             # 最终数量
print(f"② 剔除 name 含餐饮/便利店等噪音: {n_type} -> {n_clean} 条 "
      f"(剔除 {n_type - n_clean} 条)")

# ========== 5. 转 GeoDataFrame 并保存 ==========
# 用 WGS-84 坐标列构造点几何；from_xy(x经度, y纬度)
geometry = gpd.points_from_xy(df_clean["lng_wgs84"], df_clean["lat_wgs84"])
gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")  # 声明坐标系

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)  # 确保 data/processed 存在
gdf.to_file(OUT_PATH, driver="GeoJSON")             # 写 GeoJSON（GDAL 原生支持）
print(f"💾 已保存清洗后 POI：{OUT_PATH}")

# ========== 6. 打印清洗前后对比 ==========
print("\n==== 清洗前后数据量对比 ====")
print(f"原始数据      : {n_raw} 条")
print(f"type 白名单后 : {n_type} 条（保留率 {n_type / n_raw:.1%}）")
print(f"name 黑名单后 : {n_clean} 条（保留率 {n_clean / n_raw:.1%}）")
print(f"总计剔除      : {n_raw - n_clean} 条")

# 按关键词来源粗看构成，方便人工核对清洗效果
print("\n清洗后 POI 的关键词来源分布：")
print(gdf["keyword_source"].value_counts())
