# -*- coding: utf-8 -*-
"""
fetch_pois_amap.py
==================
用高德「关键字搜索」接口（v3/place/text）抓取钦州市、南宁市的
临港产业相关 POI（港口/码头/工业园/产业园/仓储/物流园/保税区/临港企业），
按 id 去重后导出 CSV，并打印统计。

运行：python fetch_pois_amap.py
Key ：.env 里的 AMAP_API_KEY（兼容旧名 AMAP_KEY；项目根没有 .env 时会向上找一级）
⚠️ 高德坐标为 GCJ-02（火星坐标），与 WGS-84 数据（OSM/DEM）叠加前需纠偏。
"""

# ========== 1. 导入依赖 ==========
import os             # 路径拼接 / 建目录 / 读系统环境变量兜底
import sys            # 日配额超限时退出整个程序
import time           # 每次请求后 sleep，防触发 QPS 限流
from pathlib import Path          # 定位 .env 文件
from collections import Counter   # 统计各城市 / 各关键词命中数

import requests       # HTTP 客户端，调高德 REST API
import pandas as pd   # 组装记录表并导出 CSV

# ========== 2. 常量配置 ==========
API_URL = "https://restapi.amap.com/v3/place/text"   # 高德关键字搜索接口（v3）

# 城市名 -> 行政区划代码 adcode（用 adcode 比中文名更精确，citylimit 配合严格限城）
CITIES = {
    "钦州市": "450700",
    "南宁市": "450100",
}

# 要抓的关键词列表（每个关键词 × 每个城市各跑一套翻页查询）
KEYWORDS = ["港口", "码头", "工业园", "产业园", "仓储", "物流园", "保税区", "临港企业"]

PAGE_SIZE = 25        # 每页条数（接口规定 offset 最大 25）
MAX_PAGE = 100        # 接口翻页上限（page 最大 100）
SLEEP = 0.3           # 每次请求后的间隔秒数（个人 key 限 3 QPS，留余量）

OUT_PATH = os.path.join("data", "raw", "pois_amap.csv")   # 输出 CSV 路径


# ========== 3. 读取 .env 中的 key ==========
def read_api_key() -> str:
    """按优先级找 key：.env 的 AMAP_API_KEY → .env 的 AMAP_KEY → 系统环境变量。"""
    fallback = ""     # 兼容旧字段名 AMAP_KEY，先记着备用
    # 依次尝试两个位置：脚本目录/.env -> 上级目录/.env（本项目的 .env 在 d:/112/yunhe）
    for p in (Path(__file__).resolve().parent / ".env",
              Path(__file__).resolve().parent.parent / ".env"):
        if p.exists():                                   # 文件存在才解析
            # utf-8-sig 兼容 Windows 记事本保存时加的 BOM 头
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()                      # 去首尾空白
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)            # 只按第一个 = 切，防止值里含 =
                    if k.strip() == "AMAP_API_KEY":      # 标准字段名，直接返回
                        return v.strip().strip("'\"")
                    if k.strip() == "AMAP_KEY" and not fallback:
                        fallback = v.strip().strip("'\"")  # 旧字段名，暂存备用
    # .env 都没有，最后退回系统环境变量
    return os.environ.get("AMAP_API_KEY") or os.environ.get("AMAP_KEY") or fallback


# ========== 4. 单城市单关键词抓取（自动翻页） ==========
def fetch_keyword(adcode: str, keyword: str):
    """
    抓取 指定 adcode 城市 + 关键词 的全部 POI，返回 (poi列表, 配额是否超限)。
    翻页规则：page 从 1 到 100，空页或凑够 count 就停。
    """
    pois = []                                  # 累积所有页的 POI
    for page in range(1, MAX_PAGE + 1):        # 按题目要求 page=1..100
        # 组装请求参数（严格按题目要求的参数集）
        params = {
            "key": AMAP_KEY,                   # 开发者 key
            "keywords": keyword,               # 关键词
            "city": adcode,                    # 传 adcode 而非城市名
            "citylimit": "true",               # 严格限制在城市边界内
            "offset": PAGE_SIZE,               # 每页 25 条
            "page": page,                      # 当前页号
        }
        # 发 GET 请求，超时 15 秒
        resp = requests.get(API_URL, params=params, timeout=15)
        time.sleep(SLEEP)                      # 每次请求后限速，防 QPS 超限
        data = resp.json()                     # 解析 JSON

        # 题目要求：infocode=10003 是日配额超限，打印提示并终止程序
        if data.get("infocode") == "10003":
            print("⛔ 高德日配额已超限（infocode=10003），终止抓取；"
                  "已抓到的数据会先保存到本地。")
            return pois, True                  # True = 配额死亡信号

        # 其他业务错误：报出错误码，放弃当前关键词（不让单点失败拖垮全局）
        if data.get("status") != "1":
            print(f"   ⚠️ {keyword} 第{page}页错误: "
                  f"{data.get('info')} ({data.get('infocode')})")
            break

        batch = data.get("pois") or []         # 本页 POI 列表
        if not batch:                          # 空页 = 已经翻到底
            break
        pois.extend(batch)                     # 收进结果

        # count 是该查询命中的总数；凑够数就不再翻页
        if len(pois) >= int(data.get("count") or 0):
            break
    return pois, False                         # False = 配额正常


# ========== 5. 主流程：城市 × 关键词 双层循环抓取 ==========
AMAP_KEY = read_api_key()                      # 先拿到 key 才能干活
if not AMAP_KEY:
    sys.exit("未找到 key：请在 .env 配置 AMAP_API_KEY（或 AMAP_KEY）")

records = []      # 最终写 CSV 的行（每行一个去重后的 POI）
city_of = {}      # poi_id -> 查询城市名（跨关键词去重后统计各城市数量用）
kw_hits = Counter()  # 各关键词命中条数（按接口原始返回计，含跨关键词重复）

for city_name, adcode in CITIES.items():       # 外层：城市
    for kw in KEYWORDS:                        # 内层：关键词
        print(f"⏳ {city_name} · {kw} ...")
        batch, quota_dead = fetch_keyword(adcode, kw)
        kw_hits[kw] += len(batch)              # 记录该关键词命中数

        for poi in batch:                      # 把高德 JSON 拍平成一行
            pid = poi.get("id")                # 高德 POI 唯一 ID = 去重键
            if not pid or pid in city_of:      # 空id或已在其他关键词抓到过 → 去重
                continue
            loc = poi.get("location") or ""    # 形如 "108.36,22.81" 的字符串
            if "," not in loc:                 # 没坐标的脏记录直接丢弃
                continue
            lng, lat = loc.split(",")[:2]      # 拆出经度、纬度
            records.append({                   # 按题目要求的 11 个字段组织
                "poi_id":         poi.get("id"),
                "name":           poi.get("name"),
                "lng":            float(lng),
                "lat":            float(lat),
                "type":           poi.get("type"),      # 高德完整分类链
                "typecode":       poi.get("typecode"),  # 分类编码
                "address":        poi.get("address"),
                "pname":          poi.get("pname"),     # 省
                "cityname":       poi.get("cityname"),  # 市
                "adname":         poi.get("adname"),    # 区县
                "keyword_source": kw,                   # 由哪个关键词命中
            })
            city_of[pid] = city_name           # 记录该 POI 属于哪个查询城市

        if quota_dead:                         # 日配额超限：先保存已抓数据再终止
            break
    if quota_dead:
        break

# ========== 6. 导出 CSV ==========
if not records:
    sys.exit("一条 POI 都没抓到，请检查 key / 网络 / 关键词")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)   # 确保 data/raw 目录存在
df = pd.DataFrame(records)                              # dict 列表 -> 表
# encoding="utf-8-sig"：带 BOM，Excel 双击打开中文不乱码；index=False 不写行号列
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n💾 已保存 {len(df)} 条去重后 POI 到 {OUT_PATH}")

# ========== 7. 打印统计 ==========
print("\n==== POI 统计 ====")
print(f"总 POI 数（去重后）: {len(df)}")
# 用 city_of 回查每个 poi_id 的查询城市，统计各城市数量
city_counts = Counter(city_of[pid] for pid in df["poi_id"])
print(f"各城市 POI 数      : {dict(city_counts)}")
print(f"各关键词命中数     : {dict(kw_hits)}")
print("\n前 5 行预览：")
print(df.head())
