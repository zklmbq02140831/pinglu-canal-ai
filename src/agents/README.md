# src/agents

基于 **LangGraph + LangChain** 的 Agent 编排层。

典型职责：

- 定义图（StateGraph）及节点（检索 / 空间计算 / LLM 调用）
- 封装查询工具（OSM 查询、Qdrant 向量检索、H3 聚合）
- 多 Agent 协作与路由

所有 Agent 入口应保持纯函数 / 类，由 `src/api/` 驱动。
