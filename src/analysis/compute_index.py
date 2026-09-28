# -*- coding: utf-8 -*-
"""
src/analysis/compute_index.py
=============================
平陆运河经济带「五维陆海联动指数」（熵权法）

五维指标（每个 H3 网格一行）：
  poi_density  网格内 POI 密度（个/km²）              正向
  road_density 网格内路网长度密度（km/km²）           正向
  slope_mean   网格内平均坡度（度，来自 DEM）         负向
  gdp          所在区县 GDP（亿元，区县级继承）       正向
  port_dist    网格质心到钦州港的大地线距离（km）     负向

流程：main() = load_data() -> compute_indicators() -> entropy_weights()
              -> composite_score() -> export()

运行：python src/analysis/compute_index.py
依赖：geopandas / rasterio / numpy / pandas / pyproj / folium
"""

# ========== 1. 导入依赖 ==========
import os                      # 路径、建目录
import sys                     # 缺文件时退出
import math                    # cos/degrees/arctan：坡度计算、GCJ加偏

import numpy as np             # 数组运算（坡度、熵权法）
import pandas as pd            # 表格处理、CSV 读写
import geopandas as gpd        # 空间连接、投影、叠加
import rasterio                # 读 DEM
from rasterio.features import rasterize   # 把六边形栅格化到 DEM 网格
from pyproj import Geod        # 大地线距离（椭球面，非直线）
import folium                  # 可视化 HTML
import branca.colormap as cm   # YlOrRd 色带

from shapely.geometry import Point        # 质心点、钦州港点
from shapely.ops import linemerge         # 多段线合并

# ========== 2. 常量与字段统一映射（所有文件名/列名只在这里改） ==========
HEX_FILE    = "data/processed/hex_grid.geojson"            # H3 网格
POI_FILE    = "data/processed/pois_amap_wgs84.csv"         # POI（WGS-84）
ROADS_FILE  = "data/processed/roads_clip.geojson"          # 裁剪后主干路网
DEM_FILE    = "data/processed/dem_wgs84.tif"               # DEM（EPSG:4326）
COUNTY_CSV  = "data/raw/stats_county.csv"                  # 区县统计（county_name,gdp_yiyuan）
COUNTY_GEO  = "data/processed/counties.geojson"            # 区县边界（高德抓取）
CANAL_CANDIDATES = [                                       # 运河中心线（仅作图）
    "data/processed/canal_centerline.geojson",
    "data/raw/canal_centerline.geojson",
]
OUT_CSV  = "data/processed/land_sea_index_v1.csv"          # 指数表输出
OUT_HTML = "output/index_map.html"                         # 可视化输出

# 字段名统一映射：规格约定 lon/lat/type_code，
# 实际 POI CSV 列是 lng_wgs84/lat_wgs84/typecode —— 在读入时统一改名
POI_COL_MAP = {"lng_wgs84": "lon", "lat_wgs84": "lat", "typecode": "type_code"}
HEX_ID_COL   = "hex_id"        # 网格 id 列名
COUNTY_COL   = "county_name"   # 区县名列
GDP_COL      = "gdp_yiyuan"    # 区县 GDP（亿元）列

IND_COLS = ["poi_density", "road_density", "slope_mean", "gdp", "port_dist"]
NEG_COLS = ["slope_mean", "port_dist"]   # 负向指标（越大越差）

PORT_LON, PORT_LAT = 108.60, 21.70        # 钦州港坐标（规格固定值）
EQ_AREA = "EPSG:6933"                     # 等积投影（算面积）
UTM     = "EPSG:32648"                    # UTM 48N（算路网长度）

GAODE_SAT = ("https://webst01.is.autonavi.com"
             "/appmaptile?style=6&x={x}&y={y}&z={z}")   # 高德卫星瓦片


# ========== 3. GCJ-02 加偏（高德卫星瓦片是 GCJ-02，WGS-84 图层叠加前必须加偏） ==========
def wgs84_to_gcj02(lng, lat):
    """国测局标准加偏，仅用于 folium 显示对齐，不写入任何数据文件。"""
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
    magic = 1 - 0.00669342162296594323 * math.sin(radlat) ** 2  # 1-e²sin²φ
    sq = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323))
                             / (magic * sq) * math.pi)      # 纬度偏移(度)
    dlng = (dlng * 180.0) / (6378245.0 / sq * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


# ========== 4. 第一步：读数据 ==========
def load_data():
    """读全部输入；缺文件一次性列出并退出；打印各文件字段和前 3 行供核对。"""
    needed = [HEX_FILE, POI_FILE, ROADS_FILE, DEM_FILE, COUNTY_GEO, COUNTY_CSV]
    missing = [p for p in needed if not os.path.exists(p)]  # 收集缺失文件
    if missing:                                             # 有缺失就全列出再退出
        sys.exit("❌ 缺少输入文件：\n  " + "\n  ".join(missing)
                 + "\n（区县统计请提供 " + COUNTY_CSV
                 + "，列：county_name,gdp_yiyuan）")

    canal_file = next(p for p in CANAL_CANDIDATES           # 中心线取第一个存在的
                      if os.path.exists(p))                 # （仅作图用，缺失可容忍）

    d = {}                                                  # 装所有数据的字典
    d["hexes"] = gpd.read_file(HEX_FILE)                    # 网格（hex_id + 面）
    poi_raw = pd.read_csv(POI_FILE, encoding="utf-8-sig")   # POI 原始表
    # 先删掉 GCJ-02 的 lng/lat 原始列，避免 lat_wgs84 重命名为 lat 后出现重名列
    poi_raw = poi_raw.drop(columns=[c for c in ("lng", "lat")
                                    if c in poi_raw.columns])
    d["pois"] = poi_raw.rename(columns=POI_COL_MAP)         # 统一列名 lon/lat/type_code
    d["roads"] = gpd.read_file(ROADS_FILE)                  # 主干路网
    d["counties"] = gpd.read_file(COUNTY_GEO)               # 区县边界
    d["stats"] = pd.read_csv(COUNTY_CSV)                    # 区县统计（GDP）
    d["canal"] = gpd.read_file(canal_file)                  # 中心线

    # ---- 打印各文件字段与前 3 行，供人工核对字段映射 ----
    print("【输入核对】")
    print(f"hex_grid   列: {list(d['hexes'].columns)}  行: {len(d['hexes'])}")
    print(f"pois       列: {list(d['pois'].columns)}  行: {len(d['pois'])}")
    print(f"roads_clip 列: {list(d['roads'].columns)}  行: {len(d['roads'])}")
    print(f"counties   列: {list(d['counties'].columns)}  行: {len(d['counties'])}")
    print(f"stats      列: {list(d['stats'].columns)}  行: {len(d['stats'])}")
    print(f"stats 前3行:\n{d['stats'].head(3).to_string()}")

    # stats 里若出现边界里没有的区县名，提前警告（常见：带"市/区"后缀不一致）
    extra = set(d["stats"][COUNTY_COL]) - set(d["counties"][COUNTY_COL])
    if extra:
        print(f"⚠️ stats 中的区县在边界里找不到（检查名称是否一致）: {extra}")
    return d


# ========== 5. 第二步：计算五维指标 ==========
def compute_indicators(d):
    """逐网格计算 poi_density / road_density / slope_mean / gdp / port_dist。"""
    hexes = d["hexes"]
    n = len(hexes)                                          # 网格数
    idx = hexes[HEX_ID_COL].to_numpy()                      # hex_id 数组

    # ---- 0. 网格面积 km²（等积投影 EPSG:6933 下计算） ----
    area_km2 = hexes.to_crs(EQ_AREA).area.to_numpy() / 1e6  # m² -> km²
    print(f"网格面积：min={area_km2.min():.2f} max={area_km2.max():.2f} km²")

    # ---- 1. poi_density：POI 空间连接落入哪个网格 ----
    pts = gpd.GeoDataFrame(                                 # POI 表 -> 点要素
        d["pois"], geometry=gpd.points_from_xy(d["pois"]["lon"], d["pois"]["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, hexes[[HEX_ID_COL, "geometry"]],  # 点落在哪个面里
                       predicate="within", how="inner")
    poi_cnt = joined.groupby(HEX_ID_COL).size()             # 每网格 POI 计数
    poi_density = pd.Series(idx, index=idx).map(poi_cnt     # 没点的网格记 0
                     ).fillna(0).to_numpy() / area_km2      # 计数 / 面积 = 密度

    # ---- 2. road_density：路网与网格求交的长度 ----
    hex_utm = hexes[[HEX_ID_COL, "geometry"]].to_crs(UTM)   # 网格转米制
    road_utm = d["roads"].to_crs(UTM)                       # 路网转米制
    inter = gpd.overlay(hex_utm, road_utm[["geometry"]],    # 相交出"格内路段"
                        how="intersection", keep_geom_type=False)
    # 交集中可能混入点/面碎片，只保留线（面×线的交集本质是线）
    inter = inter[inter.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    inter["km"] = inter.length / 1000                       # 每段长度 km
    road_km = inter.groupby(HEX_ID_COL)["km"].sum()         # 每网格总长
    road_density = pd.Series(idx, index=idx).map(road_km
                     ).fillna(0).to_numpy() / area_km2      # 长度 / 面积

    # ---- 3. slope_mean：DEM -> 坡度 -> 按网格分区统计均值 ----
    with rasterio.open(DEM_FILE) as src:                    # 打开 DEM
        dem = src.read(1).astype("float64")                 # 高程数组(行,列)
        tr = src.transform                                  # 仿射变换（像素->经纬度）
        nod = src.nodata if src.nodata is not None else -32768
        valid = (dem != nod) & (dem > -100)                 # 有效像素掩码
        dem_m = np.where(valid, dem, 0.0)                   # 无效处填 0 参与梯度
        # 像元尺寸（度）转米：南北 111.32km/度，东西乘 cos(中心纬度)
        lat0 = (src.bounds.top + src.bounds.bottom) / 2     # DEM 中心纬度
        dy = abs(tr.e) * 111320.0                           # 行方向间距（米）
        dx = tr.a * 111320.0 * math.cos(math.radians(lat0)) # 列方向间距（米）
        gy, gx = np.gradient(dem_m, dy, dx)                 # 高程梯度（无量纲）
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))     # 坡度角（度）
        # 六边形栅格化到 DEM 网格：每个像素打上所属网格编号（1..n，0=网格外）
        # 注意 rasterio 约定 shapes 元素是 (geometry, value) 顺序
        hex_raster = rasterize(
            ((g, i + 1) for i, g in enumerate(hexes.geometry)),
            out_shape=dem.shape, transform=tr, fill=0, dtype="int32",
        )
        # bincount 按网格编号聚合：坡度和 / 像素数 = 平均坡度
        s_sum = np.bincount(hex_raster.ravel(),
                            weights=np.where(valid, slope, 0).ravel(),
                            minlength=n + 1)
        s_cnt = np.bincount(hex_raster.ravel(),
                            weights=valid.ravel().astype(float),
                            minlength=n + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope_mean = np.where(s_cnt[1:] > 0,            # 有像素的算均值
                                  s_sum[1:] / s_cnt[1:], np.nan)
        n_nodata = int(np.isnan(slope_mean).sum())          # DEM 没覆盖到的网格数
        if n_nodata:
            print(f"⚠️ {n_nodata} 个网格 DEM 无覆盖，坡度用全网均值兜底")
            slope_mean = np.where(np.isnan(slope_mean),
                                  np.nanmean(slope_mean), slope_mean)

    # ---- 4. gdp：网格质心 -> 区县 -> 继承区县 GDP ----
    stats = d["stats"].drop_duplicates(COUNTY_COL           # 区县名防重复
              ).set_index(COUNTY_COL)[GDP_COL]              # 区县名 -> GDP(亿元)
    cent = hexes[[HEX_ID_COL, "geometry"]].copy()           # 复制一份做质心
    # 经纬度下求质心有偏差警告：先在 UTM 米制下求质心，再转回经纬度
    cent["geometry"] = hexes.to_crs(UTM).geometry.centroid.to_crs("EPSG:4326")
    cj = gpd.sjoin(cent, d["counties"][["county_name", "geometry"]],
                   how="left", predicate="within")          # 质心落在哪个区县
    # sjoin 可能一对多（质心压线），去重保第一条
    cj = cj.drop_duplicates(HEX_ID_COL).set_index(HEX_ID_COL)
    # 区县边界缝隙里的网格（within 查不到）：就近继承最近区县，比取均值更准
    miss = cj[cj["county_name"].isna()].index               # 没定位到区县的网格
    if len(miss):
        near = gpd.sjoin_nearest(                           # 找最近区县（米制下算）
            cent.set_index(HEX_ID_COL).loc[miss].to_crs(UTM),
            d["counties"][["county_name", "geometry"]].to_crs(UTM),
            how="left", distance_col="dist_m",
        )
        near = near[~near.index.duplicated()]               # 同距并列时保一条
        cj.loc[near.index, "county_name"] = near["county_name"].to_numpy()
        print(f"ℹ️ {len(miss)} 个网格落在区县边界缝隙，已就近继承区县")
    gdp = cj["county_name"].map(stats).to_numpy()           # 区县名映射成 GDP
    fallback = float(np.nanmean(gdp)) if np.isfinite(gdp).any() else 0.0
    n_fb = int(pd.isna(gdp).sum())                          # 查不到区县/统计的网格数
    if n_fb:
        print(f"⚠️ {n_fb} 个网格无法定位区县或区县无统计，用均值 {fallback:.1f} 兜底")
        gdp = np.where(pd.isna(gdp), fallback, gdp)
    hex_county = cj["county_name"].fillna("未知").to_dict() # hex_id -> 区县名（报告用）

    # ---- 5. port_dist：质心到钦州港的大地线距离 ----
    geod = Geod(ellps="WGS84")                              # WGS-84 椭球
    port_dist = np.empty(n)                                 # 结果数组
    for i, (lon, lat) in enumerate(                         # 逐网格质心算距离
            zip(cent.geometry.x, cent.geometry.y)):
        _, _, dist = geod.inv(lon, lat, PORT_LON, PORT_LAT) # 大地线正反解
        port_dist[i] = dist / 1000.0                        # m -> km

    df = pd.DataFrame({                                     # 汇总成一张指标表
        HEX_ID_COL: idx,
        "poi_density": np.round(poi_density, 4),
        "road_density": np.round(road_density, 4),
        "slope_mean": np.round(slope_mean, 4),
        "gdp": np.round(gdp, 2),
        "port_dist": np.round(port_dist, 3),
    })
    print("【指标计算完成】")
    print(df[IND_COLS].describe().loc[["min", "max", "mean"]].to_string())
    return df, hex_county


# ========== 6. 第三步：熵权法求权重 ==========
def entropy_weights(df):
    """负向指标先转正向 -> min-max 到 (0.0001,1) -> 比重 -> 信息熵 -> 权重。"""
    x = df[IND_COLS].astype(float).copy()                   # 原始指标矩阵
    for c in NEG_COLS:                                      # 负向转正向：
        x[c] = x[c].max() - x[c]                            # max-x 形式（大=好）
    # min-max 标准化到开区间 (0.0001, 1)，防止 ln(0)
    x_std = (x - x.min()) / (x.max() - x.min()) * 0.9999 + 0.0001
    p = x_std / x_std.sum()                                 # 比重 p_ij
    n = len(df)                                             # 样本数（网格数）
    e = -(p * np.log(p)).sum() / math.log(n)                # 信息熵 e_j ∈ (0,1)
    w = (1 - e) / (1 - e).sum()                             # 权重：1-e 归一化
    print("【熵权法权重】（权重和 = 1）")
    for c, wj in zip(IND_COLS, w):                          # 两列格式打印
        print(f"  {c:<14}: {wj:.4f}")
    print(f"  合计        : {w.sum():.4f}")
    return x_std, pd.Series(w, index=IND_COLS)


# ========== 7. 第四步：综合指数 ==========
def composite_score(x_std, w):
    """加权求和得到综合指数，保留 4 位小数。"""
    comp = (x_std * w).sum(axis=1).round(4)                 # Σ w_j * x'_ij
    print("【综合指数】min={:.4f} max={:.4f} mean={:.4f}".format(
        comp.min(), comp.max(), comp.mean()))
    return comp


# ========== 8. 第五步：导出 CSV + 可视化 HTML ==========
def export(df, x_std, comp, hex_county, canal):
    out = df.copy()                                         # 不动原表
    for c in IND_COLS:                                      # 追加五个标准化列
        out[c + "_std"] = x_std[c].round(4)
    out["composite_index"] = comp                           # 综合指数
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # 写 CSV
    print(f"💾 指数表已保存：{OUT_CSV}")

    # ---- folium 地图 ----
    hexes = gpd.read_file(HEX_FILE)                         # 重新读网格（带几何）
    hexes = hexes.merge(out[[HEX_ID_COL, "composite_index"]],  # 挂上指数
                        on=HEX_ID_COL, how="left")
    # 五级分位切分（qcut 保证每级格数接近）
    hexes["level"] = pd.qcut(hexes["composite_index"], 5, labels=[1, 2, 3, 4, 5])
    cmap = cm.linear.YlOrRd_05.scale(0, 1)                  # YlOrRd 五级色带
    m = folium.Map(location=[22.3, 108.9], zoom_start=9,    # 初始视野（规格指定）
                   tiles=None)
    folium.TileLayer(tiles=GAODE_SAT, attr="高德卫星",      # 高德卫星底图
                     name="高德卫星影像").add_to(m)

    fg_hex = folium.FeatureGroup(name="五维指数网格")        # 网格图层
    for row in hexes.itertuples():                          # 逐网格画多边形
        ext = row.geometry.exterior                         # 外环
        pts = [wgs84_to_gcj02(x, y) for x, y in             # 顶点加偏对齐底图
               zip(ext.xy[0], ext.xy[1])]
        folium.Polygon(
            locations=[(lat, lng) for lng, lat in pts],
            color="#888888", weight=0.5,                    # 细灰描边
            fill=True, fill_opacity=0.75,                   # 实色填充
            fill_color=cmap.rgb_hex_str((row.level - 1) / 4.0),  # 五级取色
            tooltip=(f"hex_id: {row.hex_id}<br>"            # 悬停信息
                     f"composite_index: {row._asdict().get('composite_index', '')}"),
        ).add_to(fg_hex)
    fg_hex.add_to(m)

    fg_canal = folium.FeatureGroup(name="运河中心线")        # 中心线图层
    g0 = canal.geometry.iloc[0]                             # 中心线几何
    if g0.geom_type == "MultiLineString":                   # 多段线先尝试合并
        g0 = linemerge(g0)
    lines = [g0] if g0.geom_type == "LineString" else list(g0.geoms)  # 未连通则逐段
    for ln in lines:                                        # 每条线分段画（加偏）
        coords = list(ln.coords)                            # 描点 (lng,lat)
        for a, b in zip(coords[:-1], coords[1:]):
            ga, gb = wgs84_to_gcj02(*a), wgs84_to_gcj02(*b)     # 段两端加偏
            folium.PolyLine([(ga[1], ga[0]), (gb[1], gb[0])],
                            color="red", weight=3).add_to(fg_canal)  # 红色宽3
    fg_canal.add_to(m)

    plng, plat = wgs84_to_gcj02(PORT_LON, PORT_LAT)         # 钦州港加偏显示
    folium.Marker(                                          # 港口标记
        location=(plat, plng), popup="钦州港",
        icon=folium.Icon(color="blue", icon="anchor"),
    ).add_to(m)

    folium.LayerControl().add_to(m)                         # 图层开关
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)   # 确保 output 目录存在
    m.save(OUT_HTML)                                        # 写 HTML
    print(f"💾 可视化地图已保存：{OUT_HTML}")

    # ---- Top10 网格报告 ----
    top = out.sort_values("composite_index", ascending=False).head(10)
    print("\n【composite_index 最高的 10 个网格】")
    for row in top.itertuples():                            # 逐行打印
        print(f"  {row.hex_id}  index={row.composite_index:.4f}  "
              f"区县={hex_county.get(row.hex_id, '未知')}")


# ========== 9. 主函数 ==========
def main():
    print("==== 平陆运河经济带五维陆海联动指数（熵权法） ====\n")
    print("[1/5] 读取输入数据 ...")
    d = load_data()
    print("\n[2/5] 计算五维指标 ...")
    df, hex_county = compute_indicators(d)
    print("\n[3/5] 熵权法求权重 ...")
    x_std, w = entropy_weights(df)
    print("\n[4/5] 计算综合指数 ...")
    comp = composite_score(x_std, w)
    print("\n[5/5] 导出结果 ...")
    export(df, x_std, comp, hex_county, d["canal"])
    print(f"\n计算完成：{len(df)}个网格，输出文件已保存")


if __name__ == "__main__":
    main()
