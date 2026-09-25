# -*- coding: utf-8 -*-
"""
fetch_osm_pois.py
=================
用 OSMnx 从 OpenStreetMap 抓取钦州市和南宁市的工业相关 POI（兴趣点），
转成 GeoDataFrame，去重后保存为 GeoPackage。

⚠️  注意：本脚本绕开 Nominatim（地理编码），改用经纬度边界框直接查 Overpass API。
    如果你本机网络到 overpass-api.de 主站不通，请在 OVERPASS_ENDPOINTS 里换镜像。

运行前提：已安装 osmnx 和 geopandas
    pip install osmnx geopandas pyogrio
"""

# ========== 1. 导入库 ==========
import os.path                                    # 拼接路径 / 建目录
import warnings                                   # 屏蔽 GeoPandas 的无关告警
import pandas as pd                               # 表拼接 / 去重
import geopandas as gpd                           # GeoDataFrame / 写文件
import osmnx as ox                                # OSM 数据抓取主库（封装 Overpass API）

# GeoPandas 在处理混合几何类型时会有 FutureWarning，屏蔽掉，保持日志干净
warnings.filterwarnings("ignore", category=FutureWarning)


# ========== 2. 全局配置 ==========
# 启用 OSMnx 本地缓存（~/.osmnx/cache/），相同 Overpass 请求不会重复发起
ox.settings.use_cache = True

# Overpass API 镜像优先级链——国内网络经常需要跳过 overpass-api.de 主站
# OSMnx 允许运行时修改 ox.settings.overpass_url，抓取前尝试第一个，失败再换
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",     # 主站（德国）
    "https://overpass.kumi.systems/api/interpreter",  # 备用镜像 1
    "https://overpass.private.coffee/api/interpreter",# 备用镜像 2
]

# 目标 POI 的 OSM tags：字典里每个 key=value 会被 OR 组合在一条 Overpass 查询里
# 即：(industrial=port) OR (landuse=industrial) OR (man_made=warehouse)
TAGS = {
    "industrial": "port",        # 码头 / 港口
    "landuse":    "industrial",  # 工业用地（工厂聚集区）
    "man_made":   "warehouse",   # 仓库 / 物流
}

# 两个城市的 bbox：(west, south, east, north)——Overpass 标准顺序
# 坐标来源：openstreetmap.org 手动查得的大致边界，略向外扩 0.1° 防遗漏
CITY_BBOXES = {
    "钦州市": (107.80, 21.50, 109.70, 23.00),
    "南宁市": (107.30, 22.10, 108.70, 23.50),
}

# 输出文件路径（相对项目根）
OUT_PATH = os.path.join("data", "raw", "pois_osm.gpkg")


# ========== 3. 逐城市抓取 ==========
def try_fetch_bbox(west, south, east, north, tags):
    """
    用 Overpass 镜像链重试 features_from_bbox，返回 GeoDataFrame 或 None。
    features_from_bbox 参数顺序：(west, south, east, north) + tags 关键字。
    """
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = url             # 切换镜像
        try:
            # features_from_bbox: 直接向 Overpass 发 bbox + tags 查询，
            # 返回该矩形内所有匹配的 OSM 点 / 线 / 面要素（GeoDataFrame）。
            # 比 features_from_place 省掉了 Nominatim 地理编码这一步。
            return ox.features_from_bbox(west, south, east, north, tags=tags)
        except Exception as e:
            last_err = e
            print(f"   镜像 {url.split('/')[2]} 失败: {e.__class__.__name__}")
            continue
    # 所有镜像都不通
    print(f"   ✗ 全部 Overpass 镜像耗尽；最后错误: {last_err}")
    return None


# 遍历两个城市，逐个调用上面的重试函数
all_gdfs = []                                     # 存放每个城市返回的 GeoDataFrame
for city, (west, south, east, north) in CITY_BBOXES.items():
    print(f"\n⏳ 正在抓取 {city} 的工业 POI ...  bbox=({west},{south},{east},{north})")

    gdf = try_fetch_bbox(west, south, east, north, TAGS)

    if gdf is None or len(gdf) == 0:
        print(f"   ✗ {city} 未抓到数据")
        continue

    # 给结果加一列来源城市，方便后续分析和地图可视化
    gdf["source_city"] = city
    print(f"   ✓ {city} 抓到 {len(gdf)} 条")
    all_gdfs.append(gdf)


# ========== 4. 合并 + 去重 ==========
if not all_gdfs:
    raise RuntimeError(
        "两个城市都没抓到数据——Overpass 全部不可达。\n"
        "检查网络代理 / 换个 Overpass 镜像 / 或用高德地理编码拿到城市边界后调用 features_from_polygon。"
    )

# 沿行方向拼接成一张大表
gdf = pd.concat(all_gdfs)
print(f"\n合并后共 {len(gdf)} 条")

# OSMnx 返回的 GeoDataFrame 自带 `osmid` 索引（OSM 内部 ID），
# 跨城市几乎不会重复，但保险起见还是去重一次。
# drop_duplicates 默认对比所有列；只用 osmid 一列即可。
if "osmid" in gdf.columns:
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["osmid"])
    if len(gdf) != before:
        print(f"去重：{before} → {len(gdf)}（按 osmid）")

# 重置索引，避免拼接后多级索引导致写出 GeoPackage 时 Fiona 告警
gdf = gdf.reset_index()


# ========== 5. 保存为 GeoPackage ==========
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)   # 确保目录存在

# to_file 默认用 Fiona/GDAL，driver="GPKG" 指定 GeoPackage 格式
# layer= 指定表名；GeoPackage 支持多 layer，这里只存一张叫 industrial_pois
gdf.to_file(OUT_PATH, driver="GPKG", layer="industrial_pois")
print(f"\n💾 已保存到 {OUT_PATH}")


# ========== 6. 打印概要 ==========
print(f"\n==== POI 汇总 ====")
print(f"总条数        : {len(gdf)}")
print(f"几何类型      : {gdf.geometry.geom_type.value_counts().to_dict()}")
print(f"覆盖城市      : {gdf['source_city'].value_counts().to_dict()}")
print(f"\n前 5 行预览：")
print(gdf.head())
