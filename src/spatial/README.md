# src/spatial

空间分析模块，依赖 **OSMnx / GeoPandas / H3**。

职责：

- `osm.py`：OSM 路网、航道、港口要素抓取
- `geopandas_utils.py`：坐标系转换、裁剪、空间关联
- `h3_utils.py`：多边形 → H3 网格、聚合、邻域查询

输出落地到 `data/processed/` 或直接供给 Agent 使用。
