from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph

# ========== 1. State：定义图的状态（全局共享数据） ==========
# TypedDict：规定状态里面有哪些字段
# Annotated[list, operator.add]：列表追加，每次节点返回会自动合并列表
class WorkState(TypedDict):
    messages: Annotated[list, operator.add]
    input_text: str
    result: str

# ========== 2. Node：节点，就是一个个处理函数 ==========
# 节点接收 state，返回字典用来更新 state
def judge_node(state: WorkState):
    text = state["input_text"]
    if len(text) > 10:
        return {"result": "长文本"}
    else:
        return {"result": "短文本"}

def long_handler(state: WorkState):
    return {"messages": [f"【长文本分支】内容：{state['input_text']}"]}

def short_handler(state: WorkState):
    return {"messages": [f"【短文本分支】内容：{state['input_text']}"]}

# ========== 3. Edge：边，控制流转；条件边用来做分支 ==========
def branch_route(state: WorkState):
    # 根据state里result的值，决定走哪个分支节点
    if state["result"] == "长文本":
        return "long"
    else:
        return "short"

# ========== 构建图 ==========
builder = StateGraph(WorkState)
# 添加节点，名字对应函数
builder.add_node("judge", judge_node)
builder.add_node("long", long_handler)
builder.add_node("short", short_handler)

# 设置入口节点：从 judge 开始跑
builder.set_entry_point("judge")

# 添加条件边：judge执行完，走branch_route判断，映射两个分支
builder.add_conditional_edges(
    "judge",
    branch_route,
    {
        "long": "long",
        "short": "short"
    }
)

# 普通边：两个分支节点执行完直接结束
builder.add_edge("long", "__end__")
builder.add_edge("short", "__end__")

# 编译图
graph = builder.compile()

# ========== 测试运行 ==========
if __name__ == "__main__":
    print("===== 测试1：短文本 =====")
    res1 = graph.invoke({"input_text": "你好LangGraph"})
    print(res1)

    print("\n===== 测试2：长文本 =====")
    res2 = graph.invoke({"input_text": "这是一段长度超过十个字符的测试文本"})
    print(res2)
