# 数据字典：land_sea_index_v1.csv

> 生成方式：由 `data/processed/land_sea_index_v1.csv` 字段与 `src/analysis/compute_index.py` 五指标口径自动生成，数据来源引用人工补录（原方案引用清单照用）。
> 生成脚本：`compute_index.py`（main = load_data → compute_indicators → entropy_weights → composite_score → export）。

## 1. 文件概况

| 项 | 值 |
|---|---|
| 路径 | `data/processed/land_sea_index_v1.csv` |
| 粒度 | 每行 = 1 个 H3 六边形网格 |
| 网格规格 | **H3 res 7**（hex_id 前缀 `87`，格均 ≈5.16 km²，勿当 res 9） |
| 坐标系 | WGS-84（EPSG:4326） |
| 行数 | 542（运河走廊缓冲区内的网格） |
| 关联几何 | `data/processed/hex_grid.geojson`（hex_id + geometry，同源生成） |

## 2. 字段定义

| 字段 | 类型 | 含义 | 方向 | 口径与计算 |
|---|---|---|---|---|
| `hex_id` | str | H3 网格唯一标识 | — | H3 res 7，16 位十六进制 |
| `poi_density` | float | POI 密度 | 正向 | 网格内 POI 数 ÷ 网格面积（个/km²）。POI 经 GCJ-02→WGS-84 纠偏后空间连接（sjoin within）落格；**口径仅工业类 POI**，商贸/生活类不在内 |
| `road_density` | float | 路网长度密度 | 正向 | OSM 主干路网（primary/secondary）与网格求交的长度 ÷ 面积（km/km²）；求交在 EPSG:32648 米制下进行 |
| `slope_mean` | float | 平均坡度 | **负向** | SRTM DEM 逐像元梯度→坡度角（度），六边形栅格化到 DEM 网格后分区均值 |
| `gdp` | float | 所在区县 GDP | 正向 | 区县级继承值（亿元）：同区县所有网格取同一值 |
| `port_dist` | float | 到钦州港距离 | **负向** | 网格质心 → 钦州港（108.60°E, 21.70°N）pyproj.Geod 椭球大地线距离（km） |
| `poi_density_std` | float | POI 密度标准化得分 | — | 熵权法 min-max 归一化（下限 0.0001），负向指标反向 |
| `road_density_std` | float | 路网密度标准化得分 | — | 同上 |
| `slope_mean_std` | float | 坡度标准化得分 | — | 负向：坡度越小得分越高 |
| `gdp_std` | float | GDP 标准化得分 | — | 注意区县继承导致的地板/触顶效应（横州市触顶 1.0、钦北区地板 0.0001） |
| `port_dist_std` | float | 距港标准化得分 | — | 负向：越近得分越高 |
| `composite_index` | float | 五维陆海联动综合指数 | — | 熵权法权重加权 `*_std` 五维得分，值域 0~1；权重由数据离散度自动确定（Day11 验收截图 01） |

注：运行时派生列（不落盘）——`county`（格心点在 `counties.geojson` 县区面内反查）、各指标排名（`*_rank`，正向降序/负向升序，见 `app/test_linkage.py`）。

## 3. 数据来源引用（原方案引用清单）

| # | 字段 | 来源 | 引用方式 |
|---|---|---|---|
| 1 | `poi_density` | **高德地图 POI**（工业类，AMAP_KEY 检索，GCJ-02→WGS-84 纠偏） | 高德开放平台 Web 服务 API；检索窗口期见 `data/raw/` 抓取脚本与 `docs/day13_protocol.md` |
| 2 | `slope_mean` | **地理空间数据云 SRTM DEM**（http://www.gscloud.cn ，SRTM 90m/30m，dem_wgs84.tif） | 地理空间数据云（中国科学院计算机网络信息中心），SRTM GL1/GLO30 |
| 3 | `road_density` | **OSM 路网**（OpenStreetMap，primary/secondary 主干网，roads_clip.geojson） | © OpenStreetMap contributors，osmnx 抓取后裁剪 |
| 4 | `gdp` | **统计年鉴 / 政府公报**：4 县区政府 2024 年国民经济和社会发展统计公报及政府工作报告（数据集卡）；钦州统计年鉴 2023（`data/raw/qinzhou_yearbook_2023.pdf`，仅县区口径）佐证 | 原方案引用清单 |
| 5 | 网格底面 | H3 六边形（Uber H3 库 ≥4.0，`polygon_to_cells` 新 API）+ 运河缓冲区（EPSG:32648 下 10km 缓冲区后转回 4326） | 项目自产 `hex_grid.geojson` |
| 6 | 县区边界（派生用） | 高德行政区划分区抓取（`counties.geojson`） | 高德开放平台 |

## 4. 使用注意（引用时须同时声明）

1. **H3 res 7**：hex_id 前缀 `87`，格均 ≈5.16 km²，不是 res 9。
2. **工业口径**：POI 仅工业类、路网仅主干级——市区商贸繁荣不计入，市区段冷点即由此产生（见 dataset_card Day12 三轮抽检）。
3. **GDP 区县继承**：网格级 gdp 为区县同一值，格间区分度来自其余四维。
4. **GCJ-02 偏移**：所有高德抓取要素入库前已纠偏至 WGS-84；高德卫星瓦片底图仍需显示层加偏。
5. 标准化下限 0.0001 为防零项，`*_std` 与 `composite_index` 同步生成于 `compute_index.py`。
