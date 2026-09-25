# -*- coding: utf-8 -*-
"""
convert_gcj02_to_wgs84.py
=========================
把 fetch_pois_amap.py 抓回来的高德 POI 坐标（GCJ-02 火星坐标系）
转换为 WGS-84（GPS 标准坐标系），供 OSM 路网 / SRTM DEM 等数据叠加使用。

算法：标准 GCJ-02→WGS-84 偏移量估算法（先按 WGS→GCJ 正变换算出偏移量，
      再把 GCJ 坐标减去该偏移量），在广西地区误差约 1~2 米。

运行：python convert_gcj02_to_wgs84.py
依赖：pandas
"""

# ========== 1. 导入依赖 ==========
import os                     # 路径拼接 / 建目录
import math                   # sin/cos/sqrt 等三角函数与开方

import pandas as pd           # 读 CSV / 加列 / 写 CSV

# GCJ-02 椭球参数（Krasovsky 1940 椭球，高德/国测局坐标系的标准假设）
A = 6378245.0                 # 长半轴（米）
EE = 0.00669342162296594323   # 第一偏心率的平方

# 文件路径：输入用高德抓的原始 CSV，输出为转换后的新 CSV
IN_PATH = os.path.join("data", "raw", "pois_amap.csv")
OUT_PATH = os.path.join("data", "raw", "pois_amap_wgs84.csv")


# ========== 2. 判断坐标是否在中国境外 ==========
def out_of_china(lng: float, lat: float) -> bool:
    """GCJ-02 偏移只在中国境内生效，境外坐标原样返回（不做转换）。"""
    # 中国大致经纬度范围：经度 73.66~135.05，纬度 3.86~53.55
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


# ========== 3. 内部变换函数（标准算法的两个多项式） ==========
def _transform_lat(x: float, y: float) -> float:
    """纬度偏移多项式：输入 (lng-105, lat-35)，返回偏移量的中间量。"""
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y \
          + 0.1 * x * y + 0.2 * math.sqrt(abs(x))          # 基础多项式项
    ret += (20.0 * math.sin(6.0 * x * math.pi)
            + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0   # 经向周期扰动项
    ret += (20.0 * math.sin(y * math.pi)
            + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0   # 纬向周期扰动项
    ret += (160.0 * math.sin(y / 12.0 * math.pi)
            + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0  # 低频扰动项
    return ret


def _transform_lng(x: float, y: float) -> float:
    """经度偏移多项式：输入 (lng-105, lat-35)，返回偏移量的中间量。"""
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x \
          + 0.1 * x * y + 0.1 * math.sqrt(abs(x))          # 基础多项式项
    ret += (20.0 * math.sin(6.0 * x * math.pi)
            + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0   # 经向周期扰动项
    ret += (20.0 * math.sin(x * math.pi)
            + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0   # 纬向周期扰动项
    ret += (150.0 * math.sin(x / 12.0 * math.pi)
            + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0  # 低频扰动项
    return ret


# ========== 4. 正向变换：WGS-84 → GCJ-02 ==========
def wgs84_to_gcj02(lng: float, lat: float):
    """标准国测局正变换：算出某 WGS-84 点被加偏后的 GCJ-02 坐标。"""
    if out_of_china(lng, lat):                 # 境外点没有偏移
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)   # 纬度偏移中间量
    dlng = _transform_lng(lng - 105.0, lat - 35.0)   # 经度偏移中间量
    radlat = lat / 180.0 * math.pi                  # 纬度转弧度
    magic = math.sin(radlat)                        # sin(纬度)
    magic = 1 - EE * magic * magic                  # 1 - e²·sin²φ
    sqrtmagic = math.sqrt(magic)                    # √(1 - e²·sin²φ)
    # 把偏移中间量换算成度（纬度用子午线曲率半径，经度用卯酉圈曲率半径）
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (A / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat                   # 加偏后的 GCJ-02 坐标


# ========== 5. 反向变换：GCJ-02 → WGS-84（偏移量估算法） ==========
def gcj02_to_wgs84(lng: float, lat: float):
    """
    反变换没有解析解，用一次偏移量估算：
    1) 把 GCJ 点临时当 WGS 点，算它正变换后会偏到哪里（得到偏移量）
    2) 原 GCJ 坐标减去这个偏移量，即 WGS-84 坐标
    在广西地区该估算误差约 1~2 米，足够地图叠加使用。
    """
    if out_of_china(lng, lat):                 # 境外点原样返回
        return lng, lat
    tmp_lng, tmp_lat = wgs84_to_gcj02(lng, lat)  # 正变换：得到偏移后的位置
    dlng = tmp_lng - lng                         # 经度偏移量
    dlat = tmp_lat - lat                         # 纬度偏移量
    return lng - dlng, lat - dlat                # 反向减掉偏移 = WGS-84


# ========== 6. 主流程：读 CSV → 批量转换 → 写 CSV ==========
df = pd.read_csv(IN_PATH, encoding="utf-8-sig")   # 读入高德原始 POI 表
print(f"读入 {len(df)} 条 POI：{IN_PATH}")

wgs_list = [                                      # 逐行做坐标转换
    gcj02_to_wgs84(row.lng, row.lat)
    for row in df.itertuples()                    # itertuples 逐行取 lng/lat
]

# 拆成两列新值，插到 lat 列后面，方便和原列左右对比
df.insert(df.columns.get_loc("lat") + 1, "lng_wgs84",
          [p[0] for p in wgs_list])               # 新列1：WGS-84 经度
df.insert(df.columns.get_loc("lat") + 2, "lat_wgs84",
          [p[1] for p in wgs_list])               # 新列2：WGS-84 纬度

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")   # 写出（原列全保留）
print(f"💾 已保存转换结果：{OUT_PATH}")

# ========== 7. 打印转换前后对比样例（前 5 条） ==========
print("\n==== 转换前后坐标对比（前 5 条） ====")
for row in df.head().itertuples():                # 只取前 5 行展示
    # 用近似公式估算偏移距离：纬度差×110.5km/度，经度差×cos(纬度)×111.3km/度
    dy = (row.lat_wgs84 - row.lat) * 110540       # 南北方向偏移（米）
    dx = (row.lng_wgs84 - row.lng) * 111320 * math.cos(math.radians(row.lat))
    dist = math.hypot(dx, dy)                     # 合成水平偏移距离（米）
    print(f"{row.name}")
    print(f"  GCJ-02 : ({row.lng:.6f}, {row.lat:.6f})")
    print(f"  WGS-84 : ({row.lng_wgs84:.6f}, {row.lat_wgs84:.6f})   "
          f"偏移约 {dist:.0f} 米")
