# -*- coding: utf-8 -*-
"""
src/analysis/spatial_stats.py
=============================
平陆运河经济带五维指数的空间统计分析：
1. 全局 Moran's I（Queen 权重，有孤立岛则改 KNN k=6）
2. moran_scatter.png   —— Moran 散点图
3. lisa_map.png        —— LISA 局部聚类图（p<0.05，HH红/LL蓝/HL橙/LH绿/不显著灰）
4. hotspot_map.png     —— G_Local 热点/冷点图
5. gradient_curve.png  —— 距运河/距钦州港距离梯度衰减曲线（2km 分箱）
6. county_radar.png    —— 四县区五指标雷达图对比

运行：python src/analysis/spatial_stats.py
依赖：libpysal / esda / geopandas / matplotlib / numpy / pandas
"""

# ========== 1. 导入依赖 ==========
import os                      # 路径、建目录
import math                    # ceil：分箱上限
import sys                     # 把本目录加入 sys.path 以复用 compute_index 常量
import numpy as np             # 数组运算、分箱
import pandas as pd            # 表格、groupby
import geopandas as gpd        # 空间数据
import matplotlib              # 绘图
matplotlib.use("Agg")          # 无界面环境强制 Agg 后端（只存 PNG 不弹窗）
import matplotlib.pyplot as plt        # 画图主接口
from matplotlib.patches import Patch   # 手工图例色块
import libpysal                # 空间权重矩阵（Queen/KNN）
from libpysal.weights import lag_spatial   # 空间滞后算子（各版本通用）
import esda                    # Moran's I / LISA / G_Local

from shapely.geometry import Point     # 钦州港点

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同目录导入
from compute_index import PORT_LON, PORT_LAT   # 复用钦州港坐标常量 (108.60, 21.70)

# ========== 2. 常量与字段映射 ==========
HEX_FILE = os.path.join("data", "processed", "hex_grid.geojson")   # 网格面
CSV_FILE = os.path.join("data", "processed", "land_sea_index_v1.csv")  # 指数表
COUNTY_GEO = os.path.join("data", "processed", "counties.geojson") # 区县边界
CANAL_CANDIDATES = [                    # 运河中心线（画图叠加用，找不到就省略）
    os.path.join("data", "processed", "canal_centerline.geojson"),
    os.path.join("data", "raw", "canal_centerline.geojson"),
]
OUT_DIR = "output"                      # PNG 输出目录
GRID_COL = "hex_id"                     # 规格叫 grid_id，实际文件是 hex_id
COUNTY_COL = "county_name"              # 县区列（CSV 没有，由边界重建）
IND_STD = ["poi_density_std", "road_density_std", "slope_mean_std",
           "gdp_std", "port_dist_std"]  # 五个标准化指标（雷达图用）
UTM = "EPSG:32649"                      # UTM 49N（规格指定，米制算距离）
ALPHA = 0.05                            # 显著性水平

# 中文字体（Windows 自带黑体），负号正常显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ========== 3. 读数据并合并（网格 + 指数 + 县区） ==========
def load_data():
    """读网格面和指数表，按 grid_id 合并，并重建县区归属列。"""
    hexes = gpd.read_file(HEX_FILE)                         # 网格面（EPSG:4326）
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")        # 指数表
    gdf = hexes.merge(df, on=GRID_COL, how="inner",         # 按 hex_id 一对一合并
                      validate="one_to_one")
    print(f"合并后网格数：{len(gdf)}")

    # ---- 重建县区列（CSV 中无县区列）：质心落在哪个县区 + 边界缝隙就近继承 ----
    counties = gpd.read_file(COUNTY_GEO)                    # 县区边界
    cent = gdf[[GRID_COL, "geometry"]].copy()               # 质心副本
    cent["geometry"] = gdf.to_crs("EPSG:32648").geometry.centroid.to_crs("EPSG:4326")
    cj = gpd.sjoin(cent, counties[[COUNTY_COL, "geometry"]],
                   how="left", predicate="within")          # 质心在哪个县区
    cj = cj.drop_duplicates(GRID_COL).set_index(GRID_COL)   # 压线去重保一条
    miss = cj[cj[COUNTY_COL].isna()].index                  # 边界缝隙里的网格
    if len(miss):                                           # 就近继承最近县区
        near = gpd.sjoin_nearest(
            cent.set_index(GRID_COL).loc[miss].to_crs(UTM),
            counties[[COUNTY_COL, "geometry"]].to_crs(UTM), how="left")
        cj.loc[near.index.drop_duplicates(), COUNTY_COL] = \
            near[COUNTY_COL].to_numpy()
    gdf[COUNTY_COL] = gdf[GRID_COL].map(cj[COUNTY_COL])     # 挂回主表

    # ---- 运河中心线（仅画图叠加；找不到就置 None，梯度曲线照算——只用距离） ----
    canal = None
    for p in CANAL_CANDIDATES:                              # 逐个候选路径
        if os.path.exists(p):                               # 存在就读第一个
            canal = gpd.read_file(p)
            break
    print(f"县区分布：{gdf[COUNTY_COL].value_counts().to_dict()}")
    print(f"运河线图层：{'已加载' if canal is not None else '未找到，省略'}")
    return gdf, canal


# ========== 4. 空间权重矩阵（Queen，孤立岛则 KNN k=6） ==========
def build_weights(gdf):
    """先试 Queen 邻接；若存在孤立岛（无邻居的网格）改用 KNN k=6。"""
    w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)  # 共边/共点邻接
    if len(w.islands) > 0:                                  # 有孤立岛
        print(f"⚠️ Queen 权重存在 {len(w.islands)} 个孤立岛，改用 KNN k=6")
        w = libpysal.weights.KNN.from_dataframe(gdf, k=6,   # 最近 6 邻
                                                use_index=False)
    else:
        print(f"Queen 权重就绪：平均邻居数 {w.mean_neighbors:.1f}，无孤立岛")
    w.transform = "r"                                       # 行标准化（滞后量可比）
    return w


# ========== 5. 第一步：全局 Moran's I ==========
def global_moran(gdf, w):
    """标准化综合指数的全局空间自相关。返回 (y_std, moran)。"""
    y = gdf["composite_index"].to_numpy(dtype=float)        # 原始综合指数
    y_std = (y - y.mean()) / y.std()                        # z-score 标准化
    moran = esda.Moran(y_std, w, transformation="r")        # 全局 Moran
    print(f"全局 Moran's I = {moran.I:.4f}  "
          f"(p={moran.p_sim:.4f}, z={moran.z_sim:.4f})")    # 打印三要素
    return y_std, moran


# ========== 6. 第二步：Moran 散点图 ==========
def moran_scatter(y_std, moran, w, out_png):
    """横轴标准化指数，纵轴空间滞后；标注回归线和 I 值。"""
    fig, ax = plt.subplots(figsize=(8, 6))                  # 画布
    lag = lag_spatial(w, y_std)                             # 空间滞后（邻域均值）
    ax.scatter(y_std, lag, s=14, alpha=0.6,                 # 542 个网格散点
               color="#4a7fb5", edgecolors="none")
    k, b = np.polyfit(y_std, lag, 1)                        # 最小二乘拟合线（斜率≈I）
    xs = np.linspace(y_std.min(), y_std.max(), 50)          # 拟合线取点
    ax.plot(xs, k * xs + b, color="red", lw=1.5,            # 红色拟合线
            label=f"拟合线 y={k:.3f}x{b:+.3f}")
    ax.axhline(0, color="grey", lw=0.6)                     # 四象限参考线
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("标准化综合指数")
    ax.set_ylabel("空间滞后变量（邻域均值）")
    ax.set_title(f"Moran 散点图  I={moran.I:.4f}, p={moran.p_sim:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)                           # 300 dpi 输出
    plt.close(fig)                                          # 释放画布
    print(f"💾 {out_png}")


# ========== 7. 通用：叠运河线的主题地图 ==========
def draw_map(gdf, colors, canal, title, legend_items, out_png):
    """colors: 每网格的颜色列表；legend_items: [(标签, 颜色)]。"""
    fig, ax = plt.subplots(figsize=(10, 8))                 # 画布
    gdf.plot(ax=ax, color=colors, edgecolor="white",       # 网格填色
             linewidth=0.2)
    if canal is not None:                                   # 运河线叠加（深红）
        canal.plot(ax=ax, color="darkred", linewidth=2)
    ax.set_title(title, fontsize=13)                        # 标题
    ax.set_axis_off()                                       # 经纬度刻度关掉更干净
    handles = [Patch(facecolor=c, edgecolor="white", label=t)  # 手工图例
               for t, c in legend_items]
    if canal is not None:
        handles.append(Patch(facecolor="darkred", label="运河中心线"))
    ax.legend(handles=handles, loc="upper center",          # 图例放图下方
              bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")      # 300 dpi
    plt.close(fig)
    print(f"💾 {out_png}")


# ========== 8. 第三步：LISA 局部聚类图 ==========
def lisa_map(gdf, w, y_std, canal, out_png):
    """esda.Moran_Local：仅 p<0.05 网格着色，四类聚集/离散。"""
    ml = esda.Moran_Local(y_std, w, transformation="r")     # 局部 Moran
    sig = ml.p_sim < ALPHA                                  # 显著性掩码
    q = ml.q                                                # 1=HH 2=LH 3=LL 4=HL
    cls = np.where(sig, q, 0)                               # 不显著=0（灰）
    cmap = {0: "#bbbbbb", 1: "#d7191c", 2: "#1a9641",       # 灰/红/绿
            3: "#2c7bb6", 4: "#ff7f00"}                     # 蓝/橙
    colors = [cmap[c] for c in cls]                         # 逐网格颜色
    legend = [("高-高聚集 HH（红）", "#d7191c"),            # 图例五项
              ("低-低聚集 LL（蓝）", "#2c7bb6"),
              ("高-低离散 HL（橙）", "#ff7f00"),
              ("低-高离散 LH（绿）", "#1a9641"),
              ("不显著（灰）", "#bbbbbb")]
    draw_map(gdf, colors, canal,
             f"LISA聚类图（p<{ALPHA}）", legend, out_png)
    n_hh = int((cls == 1).sum())                            # 统计四类数量
    n_ll = int((cls == 3).sum())
    print(f"LISA：HH={n_hh} 个，LL={n_ll} 个"
          f"（HL={int((cls == 4).sum())}, LH={int((cls == 2).sum())}）")
    return n_hh, n_ll


# ========== 9. 第四步：G_Local 热点/冷点图 ==========
def hotspot_map(gdf, w, y_std, canal, out_png):
    """esda.G_Local：z>1.96 热点红，z<-1.96 冷点蓝，其余灰。"""
    gl = esda.G_Local(y_std, w, transform="r")              # Getis-Ord Gi*
    z = gl.Zs                                               # z 得分
    colors = ["#d7191c" if v > 1.96 else                    # 热点红
              "#2c7bb6" if v < -1.96 else "#bbbbbb"         # 冷点蓝 / 灰
              for v in z]
    legend = [("显著热点 z>1.96（红）", "#d7191c"),         # 图例三项
              ("显著冷点 z<-1.96（蓝）", "#2c7bb6"),
              ("不显著（灰）", "#bbbbbb")]
    draw_map(gdf, colors, canal,
             "G_Local 热点分析图（z检验，α=0.05）", legend, out_png)
    n_hot = int((z > 1.96).sum())                           # 热点数
    n_cold = int((z < -1.96).sum())                         # 冷点数
    print(f"G_Local：热点={n_hot} 个，冷点={n_cold} 个")
    return n_hot, n_cold


# ========== 10. 第五步：距离梯度衰减曲线 ==========
def gradient_curve(gdf, canal, out_png):
    """质心重投影 32649，算到运河/到港距离，2km 分箱求指数均值。"""
    utm = gdf.to_crs(UTM)                                   # 网格转米制
    cent = utm.geometry.centroid                            # 米制质心
    d_canal = None
    if canal is not None:                                   # 到运河距离（km）
        canal_utm = canal.to_crs(UTM).geometry.union_all()  # 合并线段
        d_canal = cent.distance(canal_utm).to_numpy() / 1000
    port = gpd.GeoSeries([Point(PORT_LON, PORT_LAT)],       # 钦州港点（WGS84）
                         crs="EPSG:4326").to_crs(UTM).iloc[0]
    d_port = cent.distance(port).to_numpy() / 1000          # 到港距离（km）
    y = gdf["composite_index"].to_numpy(dtype=float)        # 因变量

    fig, ax = plt.subplots(figsize=(9, 5.5))                # 画布
    if d_canal is not None:                                 # 运河梯度线
        dfc = pd.DataFrame({"d": d_canal, "y": y})          # 分箱求均值
        dfc["bin"] = pd.cut(dfc["d"], bins=np.arange(0, dfc["d"].max() + 2, 2))
        m = dfc.groupby("bin", observed=True)["y"].mean()   # 每箱均值
        ax.plot([iv.mid for iv in m.index], m.values,       # 横轴取箱中点
                marker="o", color="#d7191c", label="距运河中心线")
    dfp = pd.DataFrame({"d": d_port, "y": y})               # 港口梯度线
    dfp["bin"] = pd.cut(dfp["d"], bins=np.arange(0, dfp["d"].max() + 2, 2))
    m = dfp.groupby("bin", observed=True)["y"].mean()
    ax.plot([iv.mid for iv in m.index], m.values,           # 第二条线
            marker="s", color="#2c7bb6", label="距钦州港")
    ax.set_xlabel("距离(km)")                               # 规格指定轴名
    ax.set_ylabel("composite_index 均值")
    ax.set_title("综合指数随距离的梯度衰减（2km 分箱）")
    ax.grid(alpha=0.3)                                      # 浅网格辅助读数
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)                           # 300 dpi
    plt.close(fig)
    print(f"💾 {out_png}")


# ========== 11. 第六步：县区雷达图 ==========
def county_radar(gdf, out_png):
    """按县区分组求五个标准化指标均值，4 条线雷达图对比。"""
    grp = gdf.groupby(COUNTY_COL)[IND_STD].mean()           # 县区 × 指标均值
    labels = ["POI密度", "路网密度", "坡度", "GDP", "港口距离"]  # 五个轴
    n = len(labels)                                         # 轴数
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]                                    # 闭合雷达图
    fig, ax = plt.subplots(figsize=(7, 7),                  # 极坐标画布
                           subplot_kw=dict(polar=True))
    for county, row in grp.iterrows():                      # 每县一条线
        vals = row.tolist() + row.tolist()[:1]              # 数值闭合
        ax.plot(angles, vals, marker="o", lw=1.5, label=county)
        ax.fill(angles, vals, alpha=0.08)                   # 淡填充
    ax.set_xticks(angles[:-1])                              # 轴刻度位置
    ax.set_xticklabels(labels)                              # 轴标签
    ax.set_title("四县区五维指标对比（标准化均值）", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))  # 图例放外侧
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")      # 300 dpi
    plt.close(fig)
    print(f"💾 {out_png}")


# ========== 12. 主函数 ==========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)                     # 确保 output 存在
    print("==== 平陆运河五维指数空间统计 ====\n")
    print("[1/6] 读数据并合并 ...")
    gdf, canal = load_data()
    print("\n[2/6] 构建空间权重 + 全局 Moran's I ...")
    w = build_weights(gdf)
    y_std, moran = global_moran(gdf, w)
    print("\n[3/6] Moran 散点图 ...")
    moran_scatter(y_std, moran, w,
                  os.path.join(OUT_DIR, "moran_scatter.png"))
    print("\n[4/6] LISA 局部聚类图 ...")
    n_hh, n_ll = lisa_map(gdf, w, y_std, canal,
                          os.path.join(OUT_DIR, "lisa_map.png"))
    print("\n[5/6] G_Local 热点图 + 梯度曲线 ...")
    n_hot, n_cold = hotspot_map(gdf, w, y_std, canal,
                                os.path.join(OUT_DIR, "hotspot_map.png"))
    gradient_curve(gdf, canal, os.path.join(OUT_DIR, "gradient_curve.png"))
    print("\n[6/6] 县区雷达图 ...")
    county_radar(gdf, os.path.join(OUT_DIR, "county_radar.png"))

    # ---- 末尾汇总打印（规格要求） ----
    print("\n==== 汇总 ====")
    print(f"全局 Moran's I = {moran.I:.4f}（p = {moran.p_sim:.4f}）")
    print(f"HH（高-高聚集）网格数：{n_hh}")
    print(f"LL（低-低聚集）网格数：{n_ll}")
    print(f"显著热点：{n_hot} 个；显著冷点：{n_cold} 个")


if __name__ == "__main__":
    main()
