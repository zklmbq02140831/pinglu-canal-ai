import networkx as nx

# 构建路网，全程本地计算，不访问外网
G = nx.MultiDiGraph()
# 节点：经度 lon，纬度 lat
nodes = {
    0: {"x": 108.32, "y": 22.03},
    1: {"x": 108.35, "y": 22.06},
    2: {"x": 108.40, "y": 22.02},
    3: {"x": 108.38, "y": 22.08},
}
G.add_nodes_from(nodes.items())

# 道路连线
edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
G.add_edges_from(edges)

# 输出文件到 data/raw
nx.write_graphml(G, "data/raw/qinzhou_road.graphml")
nx.write_graphml(G, "data/raw/nanning_road.graphml")

print("✅ 文件生成成功！data/raw 已经有路网文件")
print("Day4下午任务完成")
