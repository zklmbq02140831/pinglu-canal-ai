# pinglu-canal-ai

平陆运河 AI 项目，融合 LLM Agent、空间分析（OSM / GeoPandas / H3）与向量检索，提供面向运河场景的智能问答与决策支持。

## 目录结构

```
pinglu-canal-ai/
├── data/           # 数据目录（raw/processed/output 见子目录）
├── docs/           # 文档
├── frontend/       # 前端界面
├── notebooks/      # Jupyter 探索性笔记本
├── src/            # Python 源码
│   ├── agents/     # LangGraph Agent 编排
│   ├── spatial/    # 空间分析（OSM / GeoPandas / H3）
│   └── api/        # FastAPI 服务
├── requirements.txt
├── .env.example
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 ZHIPUAI_API_KEY / SILICONFLOW_API_KEY / AMAP_KEY

# 3. 启动 API
uvicorn src.api.main:app --reload
```

## 技术栈

- **Agent**：LangGraph + LangChain
- **LLM**：OpenAI 兼容接口（智谱 / SiliconFlow）
- **空间数据**：OSMnx、GeoPandas、H3
- **后端**：FastAPI + Uvicorn + Pydantic
- **向量检索**：Qdrant
