import osmnx as ox

print("==== 开始下载钦州路网 ====")
G_qinzhou = ox.graph_from_place("钦州市", network_type="drive")
ox.save_graphml(G_qinzhou, "data/raw/qinzhou_road.graphml")
print("✅ 钦州路网保存成功")

print("==== 开始下载南宁路网 ====")
G_nanning = ox.graph_from_place("南宁市", network_type="drive")
ox.save_graphml(G_nanning, "data/raw/nanning_road.graphml")
print("✅ 南宁路网保存成功")

# 绘图展示钦州路网
ox.plot_graph(G_qinzhou, figsize=(10, 10), node_size=1, edge_color="#4472C4")
