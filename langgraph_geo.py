from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph

# 状态定义
class GeoState(TypedDict):
    user_query: str
    intent: str
    reply: Annotated[list, operator.add]

# 节点1：意图识别
def intent_node(state: GeoState):
    query = state["user_query"]
    if "地点" in query or "POI" in query or "哪里" in query:
        return {"intent": "poi"}
    elif "高程" in query or "海拔" in query:
        return {"intent": "dem"}
    else:
        return {"intent": "unknown"}

# 节点2：POI查询节点
def poi_node(state: GeoState):
    return {"reply": [f"【POI查询】正在搜索：{state['user_query']}"]}

# 节点3：DEM高程查询节点
def dem_node(state: GeoState):
    return {"reply": [f"【高程查询】读取DEM，查询海拔：{state['user_query']}"]}

# 分支路由函数
def route_node(state: GeoState):
    match state["intent"]:
        case "poi":
            return "poi_branch"
        case "dem":
            return "dem_branch"
        case _:
            return "end_branch"

# 构建图
builder = StateGraph(GeoState)
builder.add_node("intent", intent_node)
builder.add_node("poi_query", poi_node)
builder.add_node("dem_query", dem_node)

builder.set_entry_point("intent")
builder.add_conditional_edges(
    "intent",
    route_node,
    {
        "poi_branch": "poi_query",
        "dem_branch": "dem_query",
        "end_branch": "__end__"
    }
)
builder.add_edge("poi_query", "__end__")
builder.add_edge("dem_query", "__end__")

graph = builder.compile()

if __name__ == "__main__":
    print("===== 测试1 POI查询 =====")
    res1 = graph.invoke({"user_query": "南宁有什么咖啡店地点"})
    print(res1)

    print("\n===== 测试2 高程查询 =====")
    res2 = graph.invoke({"user_query": "武鸣的海拔高程"})
    print(res2)

    print("\n===== 测试3 未知意图 =====")
    res3 = graph.invoke({"user_query": "今天天气怎么样"})
    print(res3)
