# src/api

基于 **FastAPI + Uvicorn + Pydantic** 的对外 API 层。

职责：

- `main.py`：`FastAPI()` 实例、路由挂载、生命周期
- `routes/`：业务路由（问答、地图、空间统计）
- `schemas.py`：Pydantic 请求 / 响应模型
- `deps.py`：Agent 实例、向量库连接等依赖注入

运行：`uvicorn src.api.main:app --reload`
