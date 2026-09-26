# -*- coding: utf-8 -*-
"""
verify_canal_line.py
====================
对手工描出的平陆运河中心线（data/raw/canal_centerline.geojson）做质量体检：
- CRS 自动纠正到 EPSG:4326
- 几何有效性 / 自相交检查
- 描点数、总长度、相邻点最大间距、起终点坐标
- folium + 高德卫星底图可视化 canal_check.html
- 描点超过 300 个时输出简化版 canal_simplified.geojson

运行：python verify_canal_line.py
依赖：geopandas / folium / shapely
"""

# ========== 1. 导入依赖 ==========
import os                    # 路径拼接、检查文件是否存在
import math                  # sin/cos/sqrt：显示用 GCJ-02 加偏

import geopandas as gpd      # 读 GeoJSON、坐标系转换、长度计算
import folium                # 生成 HTML 交互地图
from shapely.geometry import Point   # 相邻描点间距计算

IN_PATH = os.path.join("data", "raw", "canal_centerline.geojson")  # 手工描线成果
HTML_PATH = "canal_check.html"                                       # 可视化报告
SIMPLIFIED_PATH = os.path.join("data", "processed", "canal_simplified.geojson")

# 高德/autonavi 卫星瓦片：style=6 是卫星影像（注意它是 GCJ-02 坐标系）
GAODE_SAT = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"


# ========== 2. WGS-84 → GCJ-02（仅用于在高德卫星底图上正确显示） ==========
def wgs84_to_gcj02(lng, lat):
    """标准国测局加偏算法；autonavi 瓦片是 GCJ-02，WGS-84 的线直接叠上去会偏 ~500 米。"""
    if not (73.66 < lng < 135.05 and 3.86 < lat < 53.55):  # 境外不加偏
        return lng, lat
    # 下面是国测局偏移多项式（和 convert_gcj02_to_wgs84.py 互为逆变换）
    x, y = lng - 105.0, lat - 35.0
    dlat = (-100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
            + (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0/3.0
            + (20.0*math.sin(y*math.pi) + 40.0*math.sin(y/3.0*math.pi)) * 2.0/3.0
            + (160.0*math.sin(y/12.0*math.pi) + 320.0*math.sin(y*math.pi/30.0)) * 2.0/3.0)
    dlng = (300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
            + (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0/3.0
            + (20.0*math.sin(x*math.pi) + 40.0*math.sin(x/3.0*math.pi)) * 2.0/3.0
            + (150.0*math.sin(x/12.0*math.pi) + 300.0*math.sin(x/30.0*math.pi)) * 2.0/3.0)
    radlat = lat / 180.0 * math.pi                      # 纬度转弧度
    magic = 1 - 0.00669342162296594323 * math.sin(radlat) ** 2  # 1-e²sin²φ
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323))
                             / (magic * sqrtmagic) * math.pi)   # 纬度偏移(度)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat                       # GCJ-02 坐标


# ========== 3. 读文件与 CRS 处理 ==========
if not os.path.exists(IN_PATH):                          # 手工成果不存在就直接停
    raise SystemExit(f"未找到描线文件：{IN_PATH}")

gdf = gpd.read_file(IN_PATH)                             # 读 GeoJSON（DataFrame+geometry）
print(f"读入 {len(gdf)} 个要素：{IN_PATH}")

# 3a. CRS 检查：不是 4326 就转过去（folio瓦片和后续长度计算都按经纬度）
if gdf.crs is None:                                      # 没声明坐标系
    print("⚠️ 原文件未声明 CRS，按 EPSG:4326 处理")
elif gdf.crs.to_epsg() != 4326:                          # 声明了但不是 4326
    print(f"⚠️ 原文件CRS是 {gdf.crs.to_string()}，已自动转换")
    gdf = gdf.to_crs("EPSG:4326")                        # 转成经纬度

# 只关心第一条线（手工描线通常单要素）
geom = gdf.geometry.iloc[0]
if geom.geom_type == "MultiLineString":                  # 万一描成了多段
    from shapely.ops import linemerge                    # 才需要这个工具
    geom = linemerge(geom)                               # 尝试合并成一条线
    print("⚠️ 检测到 MultiLineString，已尝试 linemerge 合并")

# 3b. 几何有效性：无效就用 buffer(0) 修复（重合自清）
if not geom.is_valid:
    print("⚠️ 几何无效，尝试 buffer(0) 修复 ...")
    geom = geom.buffer(0)                                # 经典自愈修复法
    if geom.geom_type != "LineString":                   # buffer 可能把线变面/空
        print("⚠️ buffer(0) 后不再是 LineString，保留原始几何继续检查")
        geom = gdf.geometry.iloc[0]                      # 回退到原几何
else:
    print("✅ 几何有效性：valid")

# 3c. 自相交检查：LineString 用 is_simple（True=无自交）
print(f"{'✅' if geom.is_simple else '❌'} 自相交检查：is_simple = {geom.is_simple}"
      + ("" if geom.is_simple else "（存在自相交，建议检查描点顺序）"))

coords = list(geom.coords)                               # 所有描点 (lon, lat) 列表
n_pts = len(coords)                                      # 描点总数

# ========== 4. 统计信息 ==========
print("\n==== 中心线质量统计 ====")
print(f"描点总数：{n_pts} 个")

# 4a. 投影到 UTM 48N（EPSG:32648，单位米）再算长度，避免经纬度直接算长度的失真
# 注意：要用合并/修复后的 geom 重投影，而不是 gdf 里的原始几何（可能是多段线）
geom_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:32648").iloc[0]
length_km = geom_utm.length / 1000                       # 线长度(米) -> 公里
print(f"线总长度：{length_km:.2f} 公里（UTM 48N 投影长度）")

# 4b. 相邻描点间距：在投影坐标里逐段算两点距离
utm_coords = list(geom_utm.coords)                       # 投影后的描点
seg_km = [Point(a).distance(Point(b)) / 1000             # 每段长度(公里)
          for a, b in zip(utm_coords[:-1], utm_coords[1:])]
max_gap = max(seg_km)                                    # 最大段间距
max_idx = seg_km.index(max_gap)                          # 最大段的位置（第几个点后）
print(f"相邻描点最大间距：{max_gap:.2f} 公里（第 {max_idx}~{max_idx + 1} 点之间）")
if max_gap > 5:                                          # 超过 5 公里 = 疑似跳段
    print(f"⚠️ 存在大跳段，可能有误点：最大间距 {max_gap:.2f} 公里 "
          f"> 5 公里，请检查第 {max_idx} 与第 {max_idx + 1} 个描点")

# 4c. 起终点坐标（WGS-84 经纬度，保留 6 位小数）
print(f"起点坐标：{coords[0][0]:.6f}, {coords[0][1]:.6f}")
print(f"终点坐标：{coords[-1][0]:.6f}, {coords[-1][1]:.6f}")

# ========== 5. folium 可视化报告 ==========
# 底图中心取起终点中点（folium 要 (lat, lng) 顺序）
center = [(coords[0][1] + coords[-1][1]) / 2, (coords[0][0] + coords[-1][0]) / 2]
m = folium.Map(location=center, zoom_start=10)           # 初始化地图
folium.TileLayer(tiles=GAODE_SAT, attr="高德卫星影像", name="高德卫星").add_to(m)

# 线体：注意瓦片是 GCJ-02，先把每个 WGS-84 描点加偏后再画，才能和卫星影像对齐
gcj_pts = [wgs84_to_gcj02(lon, lat) for lon, lat in coords]        # 全部描点加偏
folium.PolyLine(
    locations=[(lat, lon) for lon, lat in gcj_pts],      # (lat, lng) 顺序
    color="red", weight=5, opacity=0.9, tooltip="平陆运河中心线",
).add_to(m)                                              # 红色粗线

# 起点绿标记 / 终点蓝标记（弹窗里同时给 WGS-84 坐标方便核对）
folium.Marker(
    location=(gcj_pts[0][1], gcj_pts[0][0]),             # 起点位置(GCJ显示坐标)
    popup=f"起点：平塘江口（WGS84: {coords[0][0]:.6f}, {coords[0][1]:.6f}）",
    icon=folium.Icon(color="green", icon="play"),
).add_to(m)
folium.Marker(
    location=(gcj_pts[-1][1], gcj_pts[-1][0]),           # 终点位置
    popup=f"终点：钦州港（WGS84: {coords[-1][0]:.6f}, {coords[-1][1]:.6f}）",
    icon=folium.Icon(color="blue", icon="stop"),
).add_to(m)

# 每隔 10 个描点放一个带序号的小圆点，方便定位「第几个点」出了问题
for i, (lon, lat) in enumerate(gcj_pts):
    if i % 10 == 0:                                      # 序号是 10 的倍数才画
        folium.CircleMarker(
            location=(lat, lon), radius=3,               # 小圆点
            color="orange", fill=True, fill_opacity=0.9,
            tooltip=f"描点 #{i}", popup=f"描点 #{i}\nWGS84: {coords[i][0]:.6f}, {coords[i][1]:.6f}",
        ).add_to(m)

folium.LayerControl().add_to(m)                          # 右上角图层开关
m.save(HTML_PATH)                                        # 保存成独立 HTML
print(f"\n💾 可视化报告已生成：{HTML_PATH}（浏览器打开即可核对描线）")

# ========== 6. 描点过多时输出简化版 ==========
if n_pts > 300:                                          # 超过 300 个描点才简化
    sim_geom = geom.simplify(0.0005)                     # 0.0005度 ≈ 50米容差
    n_sim = len(sim_geom.coords)                         # 简化后的描点数
    os.makedirs(os.path.dirname(SIMPLIFIED_PATH), exist_ok=True)
    # 用单要素 GeoDataFrame 写出（保留 4326 坐标系声明）
    gpd.GeoDataFrame({"name": ["canal_simplified"]},
                     geometry=[sim_geom], crs="EPSG:4326"
                     ).to_file(SIMPLIFIED_PATH, driver="GeoJSON")
    print(f"\n💾 简化版已保存：{SIMPLIFIED_PATH}")
    print(f"简化前后点数对比：{n_pts} -> {n_sim} 个"
          f"（减少 {n_pts - n_sim} 个，压缩率 {1 - n_sim / n_pts:.1%}）")
else:
    print(f"\n描点数 {n_pts} ≤ 300，无需生成简化版")
