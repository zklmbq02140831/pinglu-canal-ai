# -*- coding: utf-8 -*-
"""
build_hex_grid.py
=================
把平陆运河中心线做成 10 公里缓冲区，再用 H3 res=7 六边形切分：
1. 读中心线（优先简化版，没有就用原始版）
2. 投影坐标系下做 10 km 缓冲区（UTM 48N，避免经纬度直接缓冲失真）
3. h3.polygon_to_cells（v4 新 API）切六边形网格
4. 输出 hex_grid.geojson + folium 叠加验证页 hex_check.html + 统计

运行：python build_hex_grid.py
依赖：geopandas / h3>=4.0 / folium / shapely
"""

# ========== 1. 导入依赖 ==========
import os                    # 路径拼接、判断文件是否存在
import math                  # sin/cos/sqrt：folium 显示用 GCJ-02 加偏

import h3                    # Uber H3 六边形地理索引（要求 >=4.0 新 API）
import geopandas as gpd      # GeoJSON 读写、坐标系转换
import folium                # 生成 HTML 验证地图
from shapely.geometry import Polygon  # 每个 H3 格子转成面几何

SIMPLIFIED = os.path.join("data", "processed", "canal_simplified.geojson")  # 简化线
ORIGINAL = os.path.join("data", "raw", "canal_centerline.geojson")          # 原始线
OUT_GEOJSON = os.path.join("data", "processed", "hex_grid.geojson")         # 网格输出
OUT_HTML = "hex_check.html"                                                 # 验证页

BUFFER_M = 10000            # 缓冲半径（米）：中心线两侧各 10 公里
H3_RES = 7                  # H3 分辨率 7：平均六边形边长约 1.4km，面积约 5.2km²

GAODE_SAT = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"


# ========== 2. WGS-84 → GCJ-02（autonavi 卫星瓦片是 GCJ-02，叠加前必须加偏） ==========
def wgs84_to_gcj02(lng, lat):
    """国测局标准加偏算法，与 convert_gcj02_to_wgs84.py 的反变换互逆。"""
    if not (73.66 < lng < 135.05 and 3.86 < lat < 53.55):   # 境外不加偏
        return lng, lat
    x, y = lng - 105.0, lat - 35.0                          # 平移到算法原点
    dlat = (-100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
            + (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0/3.0
            + (20.0*math.sin(y*math.pi) + 40.0*math.sin(y/3.0*math.pi)) * 2.0/3.0
            + (160.0*math.sin(y/12.0*math.pi) + 320.0*math.sin(y*math.pi/30.0)) * 2.0/3.0)
    dlng = (300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
            + (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0/3.0
            + (20.0*math.sin(x*math.pi) + 40.0*math.sin(x/3.0*math.pi)) * 2.0/3.0
            + (150.0*math.sin(x/12.0*math.pi) + 300.0*math.sin(x/30.0*math.pi)) * 2.0/3.0)
    radlat = lat / 180.0 * math.pi                          # 纬度转弧度
    magic = 1 - 0.00669342162296594323 * math.sin(radlat) ** 2   # 1-e²sin²φ
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323))
                             / (magic * sqrtmagic) * math.pi)      # 纬度偏移量(度)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat                           # 返回 GCJ-02 坐标


# ========== 3. 读中心线（优先简化版，回退原始版） ==========
if os.path.exists(SIMPLIFIED):                # 简化版存在就用它（点少、算得快）
    line_path = SIMPLIFIED
    print(f"使用简化版中心线：{SIMPLIFIED}")
else:                                         # 否则回退原始手工线
    line_path = ORIGINAL
    print(f"未找到简化版，使用原始中心线：{ORIGINAL}")

gdf = gpd.read_file(line_path)                # 读 GeoJSON
if gdf.crs is None:                           # 没声明坐标系就按 4326 处理
    gdf = gdf.set_crs("EPSG:4326")
elif gdf.crs.to_epsg() != 4326:               # 声明了别的坐标系就转成 4326
    print(f"⚠️ 原文件CRS是 {gdf.crs.to_string()}，已自动转换")
    gdf = gdf.to_crs("EPSG:4326")

line = gdf.geometry.iloc[0]                   # 取第一条线
if line.geom_type == "MultiLineString":       # 手工线常见多段，先合并
    from shapely.ops import linemerge
    line = linemerge(line)                    # 合并成一条 LineString
print(f"中心线长度：{line.length * 111.32 * math.cos(math.radians(line.centroid.y)):.1f} "
      f"公里（经纬度粗估，精确值见 verify_canal_line.py）")

# ========== 4. 投影坐标系下做缓冲区（严格按顺序：投影→缓冲→转回） ==========
line_utm = gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:32648").iloc[0]
#                                              a. 转 UTM 48N（广西，单位：米）
buffer_utm = line_utm.buffer(BUFFER_M)        # b. 10km 缓冲区（米制下计算）
buffer_wgs = (
    gpd.GeoSeries([buffer_utm], crs="EPSG:32648").to_crs("EPSG:4326").iloc[0]
)                                             # c. 转回 EPSG:4326，供 H3 使用
area_km2 = buffer_utm.area / 1e6              # 缓冲区面积（投影坐标系下算，单位 km²）

# ========== 5. H3 切六边形网格（v4 新 API：polygon_to_cells） ==========
def cells_from_geom(geom):
    """把 Polygon/MultiPolygon 切成 H3 格子 id 列表（MultiPolygon 需逐个面处理）。"""
    cells = []                                 # 收集所有格子的 H3 id
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]  # 拆成单面列表
    for poly in polys:                         # 逐个面切格子
        # h3 4.x 只认自家 LatLngPoly 类型（不收 shapely），
        # 需把外环顶点从 shapely 的 (lng,lat) 转成 h3 的 (lat,lng)
        ring = list(poly.exterior.coords)      # 外环顶点 [(lng,lat),...]
        if ring[0] == ring[-1]:                # 去掉闭合重复点（LatLngPoly 自动闭合）
            ring = ring[:-1]
        llpoly = h3.LatLngPoly([(lat, lng) for lng, lat in ring])  # 转 h3 类型
        cells.extend(h3.polygon_to_cells(llpoly, H3_RES))  # v4 新 API（替代旧 polyfill）
    return cells

cells = cells_from_geom(buffer_wgs)           # 得到缓冲区内全部六边形 id
print(f"\nH3 切分完成：res={H3_RES}，共 {len(cells)} 个六边形")

# 每个格子转成 shapely 面 + hex_id 属性，组装 GeoDataFrame
rows = []                                     # 暂存 (hex_id, Polygon)
for cid in cells:                             # 逐格子取边界
    # cell_to_boundary 返回 ((lat,lng),...)，shapely 要 (lng,lat)，翻转一下
    ring = [(lng, lat) for lat, lng in h3.cell_to_boundary(cid)]
    rows.append((cid, Polygon(ring)))         # 闭合成六边形面

hex_gdf = gpd.GeoDataFrame(                   # 组装 GeoDataFrame
    {"hex_id": [r[0] for r in rows]},         # 属性列：hex_id
    geometry=[r[1] for r in rows],            # 几何列：六边形面
    crs="EPSG:4326",                          # 声明经纬度坐标系
)

os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)   # 确保 processed 目录存在
hex_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")             # 保存网格 GeoJSON
print(f"💾 网格已保存：{OUT_GEOJSON}")

# ========== 6. folium 叠加验证页 ==========
# 地图中心取缓冲区质心（folium 要 (lat,lng) 顺序）
c = buffer_wgs.centroid                       # 缓冲区质心（WGS-84）
cc_lng, cc_lat = wgs84_to_gcj02(c.x, c.y)     # 质心加偏成 GCJ-02，对齐卫星底图
m = folium.Map(location=(cc_lat, cc_lng), zoom_start=10)   # 初始化地图
folium.TileLayer(tiles=GAODE_SAT, attr="高德卫星影像", name="高德卫星").add_to(m)

# 6a. 六边形网格：半透明填充，先画（在线和边界的下层）
fg_hex = folium.FeatureGroup(name="H3 网格 (res=7)")       # 一个图层组收所有格子
for row in hex_gdf.itertuples():             # 逐格子画多边形
    # 同样要把格子顶点加偏成 GCJ-02，否则整片网格偏 ~500 米
    pts = [wgs84_to_gcj02(x, y) for x, y in zip(row.geometry.exterior.xy[0],
                                               row.geometry.exterior.xy[1])]
    folium.Polygon(
        locations=[(lat, lng) for lng, lat in pts],   # folium 要 (lat,lng)
        color="#4a90d9", weight=1,                    # 细边框
        fill=True, fill_color="#4a90d9", fill_opacity=0.3,   # 半透明填充
        tooltip=f"hex_id: {row.hex_id}",              # 鼠标悬停显示格子 id
    ).add_to(fg_hex)
fg_hex.add_to(m)

# 6b. 中心线：红色线
fg_line = folium.FeatureGroup(name="运河中心线")
coords = list(line.coords)                    # 中心线描点 (lng,lat)
for i in range(len(coords) - 1):              # 分段画 PolyLine（folium 线不能直接加偏）
    a = wgs84_to_gcj02(*coords[i])            # 段起点加偏
    b = wgs84_to_gcj02(*coords[i + 1])        # 段终点加偏
    folium.PolyLine([(a[1], a[0]), (b[1], b[0])],
                    color="red", weight=4, opacity=0.9).add_to(fg_line)
fg_line.add_to(m)

# 6c. 缓冲区边界：橙色虚线（只画外轮廓，不填充）
fg_buf = folium.FeatureGroup(name="10km 缓冲区边界")
buf_polys = (buffer_wgs.geoms if buffer_wgs.geom_type == "MultiPolygon" else [buffer_wgs])
for bp in buf_polys:                          # 逐面画虚线边界
    bx, by = bp.exterior.xy                   # 外环顶点坐标序列
    pts = [wgs84_to_gcj02(x, y) for x, y in zip(bx, by)]      # 逐点加偏
    folium.PolyLine(
        [(lat, lng) for lng, lat in pts],
        color="orange", weight=2, dash_array="6,6",   # 橙色虚线
        tooltip="10 km 缓冲区",
    ).add_to(fg_buf)
fg_buf.add_to(m)

folium.LayerControl().add_to(m)               # 右上角图层开关（可单独开关网格/线/边界）
m.save(OUT_HTML)                              # 保存 HTML
print(f"💾 叠加验证页已生成：{OUT_HTML}")

# ========== 7. 打印统计与合理性检查 ==========
print("\n==== H3 网格统计 ====")
print(f"网格总数     : {len(hex_gdf)} 个（H3 res={H3_RES}）")
print(f"缓冲区面积   : {area_km2:.1f} km²")
# 合理性检查：res=7 单格约 5.16 km²，10km 缓冲带约 2900 km²，正常应切出 500~600 个
if not (300 <= len(hex_gdf) <= 800):
    print("⚠️ 网格数量异常，建议调整buffer半径或H3分辨率")
else:
    print("✅ 网格数量在合理区间 300~800")
