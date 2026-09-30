# -*- coding: utf-8 -*-
"""
app/test_linkage.py
===================
Streamlit 三项联动 spike 测试（目标是 PASS/FAIL 判定，不是成品）：

  联动①  地图点格 → 详情面板（五项指标 + 排名）
  联动②  侧栏筛选（县区 + composite 区间）→ 地图重渲染
  联动③  plotly 柱状图选县区 → 地图高亮对应格

自检结果逐条追加写入 output/test_linkage_log.jsonl，
页面底部有实时状态区。运行：
  python -m streamlit run app/test_linkage.py --server.headless=true --server.port=8511
"""

# ========== 1. 导入与路径 ==========
import os
import json
from datetime import datetime

import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)  # 之后全部用相对项目根的路径

IDX_FILE = "data/processed/land_sea_index_v1.csv"   # 指数表（无县区列）
HEX_FILE = "data/processed/hex_grid.geojson"        # H3 res7 网格（hex_id+geometry）
CTY_FILE = "data/processed/counties.geojson"        # 县区边界（county_name）
LOG_FILE = "output/test_linkage_log.jsonl"          # 自检日志

IND_COLS = ["poi_density", "road_density", "slope_mean", "gdp", "port_dist"]
NEG_COLS = {"slope_mean", "port_dist"}              # 负向指标（越小越好，排名升序）
IND_LABELS = {
    "poi_density": "POI密度", "road_density": "路网密度",
    "slope_mean": "平均坡度", "gdp": "区县GDP", "port_dist": "港口距离",
    "composite_index": "综合指数",
}

# ========== 2. 数据加载（缓存） ==========
@st.cache_data(show_spinner="加载数据…")
def load_data():
    idx = pd.read_csv(IDX_FILE)
    hexes = gpd.read_file(HEX_FILE)
    counties = gpd.read_file(CTY_FILE)

    # 指数表无县区列 → 格心点落在哪个县区面内反查（spike 允许 geographic centroid 警告）
    cents = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"hex_id": hexes["hex_id"]}, geometry=cents, crs=hexes.crs)
    j = gpd.sjoin(pts, counties[["county_name", "geometry"]],
                  how="left", predicate="within")
    hex2cty = j.drop_duplicates("hex_id").set_index("hex_id")["county_name"]
    idx["county"] = idx["hex_id"].map(hex2cty)

    # 排名：正向指标降序（第1名=最大），负向指标升序（第1名=最小）
    n = len(idx)
    for c in IND_COLS:
        idx[f"{c}_rank"] = idx[c].rank(ascending=(c in NEG_COLS), method="min").astype(int)
    idx["composite_index_rank"] = idx["composite_index"].rank(
        ascending=False, method="min").astype(int)

    grid_all = hexes.merge(idx[["hex_id", "county", "composite_index"]], on="hex_id", how="inner")
    return idx, hexes, grid_all


def log_event(event, ok, once_key=None, **kw):
    """追加一条自检结果到 JSONL；once_key 用于同一结果只记一次，防刷屏。"""
    if once_key:
        logged = st.session_state.setdefault("_logged", set())
        if once_key in logged:
            return
        logged.add(once_key)
    rec = {"ts": datetime.now().strftime("%H:%M:%S"),
           "event": event, "ok": bool(ok), **kw}
    os.makedirs("output", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def chip(container, ok, label, detail=""):
    """顶部三枚 PASS/FAIL/待测 状态章。"""
    txt = "✅ PASS" if ok else ("❌ FAIL" if ok is False else "⏳ 待测")
    container.markdown(f"**联动{label}**：{txt}  \n<small>{detail}</small>",
                       unsafe_allow_html=True)


# ========== 3. 地图构建 ==========
@st.cache_data
def color_scale():
    idx, _, _ = load_data()
    return cm.linear.YlOrRd_09.scale(idx["composite_index"].min(),
                                     idx["composite_index"].max())

LAT0, LON0 = 21.95, 108.65  # 运河走廊大致中心（固定，保证地图 hash 稳定）

def build_map(hex_gdf, highlight_gdf=None):
    """composite 着色 GeoJson 层 + 可选高亮层（联动③）。"""
    m = folium.Map(location=[LAT0, LON0], zoom_start=9, tiles="OpenStreetMap")
    scale = color_scale()
    if len(hex_gdf):
        hexes = hex_gdf.sort_values("hex_id")  # 排序保证 GeoJSON 确定性
        style = lambda f: {"fillColor": scale(f["properties"]["composite_index"]),
                           "color": "#666", "weight": 1, "fillOpacity": 0.75}
        folium.GeoJson(
            hexes[["hex_id", "composite_index", "geometry"]].to_json(),
            style_function=style,
            tooltip=folium.GeoJsonTooltip(fields=["hex_id"], aliases=["格ID:"]),
        ).add_to(m)
    if highlight_gdf is not None and len(highlight_gdf):
        hl = highlight_gdf.sort_values("hex_id")
        folium.GeoJson(
            hl[["hex_id", "geometry"]].to_json(),
            style_function=lambda f: {"fillColor": "#00bcd4", "color": "#00bcd4",
                                      "weight": 3, "fillOpacity": 0.25},
            tooltip=folium.GeoJsonTooltip(fields=["hex_id"], aliases=["高亮格:"]),
        ).add_to(m)
    return m


# ========== 4. 页面与数据 ==========
st.set_page_config(layout="wide", page_title="三项联动spike")
st.title("平陆运河 · 三项联动 spike 测试")

idx, hexes, grid_all = load_data()
vmin, vmax = float(idx["composite_index"].min()), float(idx["composite_index"].max())

# ---------- 联动②：侧栏筛选 ----------
st.sidebar.header("联动② 筛选")
county_opts = ["全部"] + sorted(idx["county"].dropna().unique().tolist())
county_sel = st.sidebar.selectbox("县区", county_opts, key="county_sel")
comp_range = st.sidebar.slider("composite 区间", vmin, vmax, (vmin, vmax),
                               step=0.01, key="comp_slider")

mask = idx["composite_index"].between(*comp_range)
if county_sel != "全部":
    mask &= idx["county"] == county_sel
idx_f = idx[mask]
hex_f = grid_all[grid_all["hex_id"].isin(idx_f["hex_id"])]
n_before, n_after = len(idx), len(idx_f)
st.sidebar.caption(f"筛选后格数：{n_after} / {n_before}")

# 联动② 自检：过滤条件生效且格数变化 → PASS（同一筛选签名只记一次）
if (county_sel != "全部" or comp_range != (round(vmin, 2), round(vmax, 2))):
    if n_after < n_before:
        log_event("link2_filter", True, once_key=f"link2:{county_sel}:{comp_range}",
                  county=county_sel, comp_range=list(comp_range),
                  n_before=n_before, n_after=n_after)
        st.session_state["link2_ok"] = True
        st.session_state["link2_detail"] = f"{county_sel} {comp_range} → {n_after}/{n_before} 格"
    elif n_after == 0:
        log_event("link2_filter", False, once_key=f"link2empty:{county_sel}:{comp_range}",
                  reason="过滤后 0 格（条件过严或县区无网格）")
        st.session_state["link2_ok"] = False

# ---------- 联动③：plotly 图表（放在地图前定义，选中县区用于高亮） ----------
cty_mean = (idx.dropna(subset=["county"])
              .groupby("county")["composite_index"].mean()
              .sort_values(ascending=False).reset_index())
fig = px.bar(cty_mean, x="county", y="composite_index",
             title="县区 × composite 均值（点击柱子选中）",
             custom_data=["county"])
fig.update_layout(height=330)

chart_event = None
try:
    chart_event = st.plotly_chart(fig, on_select="rerun", key="chart",
                                  selection_mode=["points"], use_container_width=True)
except TypeError as e:
    log_event("link3_param_unsupported", False, once_key="link3:typeerror", reason=str(e))
    st.error(f"st.plotly_chart 不支持 on_select 参数：{e}")

# 解析选中的县区：优先返回值 selection，回退 session_state
def county_from_point(p):
    for k in ("county", "x"):
        if p.get(k):
            return p[k]
    cd = p.get("customdata")
    if cd:
        return cd[0] if isinstance(cd, (list, tuple)) else cd
    return p.get("label")

sel_points = []
if chart_event is not None:
    try:
        sel_points = chart_event.selection.points
    except Exception:
        sel_points = []
if not sel_points:
    cs = st.session_state.get("chart") or {}
    sel_points = (cs.get("selection") or {}).get("points", []) if isinstance(cs, dict) else []

sel_county = county_from_point(sel_points[0]) if sel_points else None
highlight_gdf = grid_all[grid_all["county"] == sel_county] if sel_county else None

if sel_county:
    n_hl = len(highlight_gdf)
    if n_hl > 0:
        log_event("link3_chart_select", True, once_key=f"link3:{sel_county}",
                  county=sel_county, n_highlight=n_hl)
        st.session_state["link3_ok"] = True
        st.session_state["link3_detail"] = f"选中 {sel_county} → 高亮 {n_hl} 格"
    else:
        log_event("link3_chart_select", False, once_key=f"link3none:{sel_county}",
                  reason=f"选中 {sel_county} 但 0 格可高亮")
        st.session_state["link3_ok"] = False

# ---------- 联动① + 地图 ----------
st.subheader("联动① 地图点格 → 详情")
col_map, col_detail = st.columns([1.4, 1])
with col_map:
    m = build_map(hex_f, highlight_gdf)
    st_folium(m, key="map_click", returned_objects=["last_object_clicked", "last_clicked"],
              use_container_width=True, height=520)

# st_folium 把点击数据放 st.session_state["map_click"]（键=returned_objects 子集）
click_state = st.session_state.get("map_click") or {}
pt = click_state.get("last_object_clicked") or click_state.get("last_clicked")

gid, hit_row = None, None
if isinstance(pt, dict) and pt.get("lat") is not None:
    p = Point(pt["lng"], pt["lat"])
    sub = hexes[hexes.geometry.contains(p)]   # shapely 点在多边形内反查 grid_id
    if len(sub):
        gid = sub.iloc[0]["hex_id"]
        hit_row = idx[idx["hex_id"] == gid]

with col_detail:
    st.markdown("**该格五项指标 + 排名**")
    if hit_row is not None and len(hit_row):
        r = hit_row.iloc[0]
        rows = []
        for c in IND_COLS:
            rows.append({"指标": IND_LABELS[c], "数值": round(float(r[c]), 4),
                         "排名": f"{int(r[c + '_rank'])}/{len(idx)}"})
        rows.append({"指标": IND_LABELS["composite_index"],
                     "数值": round(float(r["composite_index"]), 4),
                     "排名": f"{int(r['composite_index_rank'])}/{len(idx)}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"格ID `{gid}` · 县区：{r['county']}")
        log_event("link1_map_click", True, once_key=f"link1:{gid}",
                  grid_id=gid, lat=pt["lat"], lng=pt["lng"])
        st.session_state["link1_ok"] = True
        st.session_state["link1_detail"] = f"{gid}（{r['county']}）"
    elif gid is not None:
        st.warning(f"点中格 `{gid}` 但该格不在指数表中")
        log_event("link1_map_click", False, once_key=f"link1noidx:{gid}",
                  reason="grid_id 不在指数表", grid_id=gid)
        st.session_state["link1_ok"] = False
    elif pt is not None:
        st.info("点击坐标未落入任何网格（点在格间隙/走廊外），换一格点击")
        log_event("link1_map_click", False, once_key=f"link1miss:{pt['lat']:.5f},{pt['lng']:.5f}",
                  reason="last_clicked 坐标不在任何多边形内",
                  lat=pt["lat"], lng=pt["lng"])
    else:
        st.info("👆 点击地图上任一六边形格查看详情")

# ---------- 顶部状态章（需在数据自检后渲染，放底部保证拿到结果） ----------
st.divider()
st.subheader("自检结果")
c1, c2, c3 = st.columns(3)
chip(c1, st.session_state.get("link1_ok"), "① 点格→详情",
     st.session_state.get("link1_detail", "点击地图格"))
chip(c2, st.session_state.get("link2_ok"), "② 筛选→地图",
     st.session_state.get("link2_detail", "左侧切县区/拖滑块"))
chip(c3, st.session_state.get("link3_ok"), "③ 图表→地图",
     st.session_state.get("link3_detail", "点击柱状图柱子"))

with st.expander("日志明细（output/test_linkage_log.jsonl）"):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        st.code("".join(lines[-30:]) or "(空)", language="json")
    else:
        st.code("(尚无日志)", language="json")
