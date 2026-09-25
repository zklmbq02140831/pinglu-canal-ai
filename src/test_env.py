# 环境测试脚本 Day3 pinglu-canal-ai
import os
import sys

print("===== Python环境信息 =====")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

try:
    import langgraph
    print("✅ langgraph 导入成功")
except Exception as e:
    print(f"❌ langgraph 导入失败: {e}")

try:
    import langchain
    print("✅ langchain 导入成功")
except Exception as e:
    print(f"❌ langchain 导入失败: {e}")

try:
    import openai
    print("✅ openai 导入成功")
except Exception as e:
    print(f"❌ openai 导入失败: {e}")

try:
    import osmnx
    print("✅ osmnx 导入成功")
except Exception as e:
    print(f"❌ osmnx 导入失败: {e}")

try:
    import geopandas as gpd
    print("✅ geopandas 导入成功")
except Exception as e:
    print(f"❌ geopandas 导入失败: {e}")

print("\n===== 目录结构检查 =====")
dirs = [
    "data/raw",
    "data/processed",
    "data/output",
    "notebooks",
    "src/agents",
    "src/spatial",
    "src/api"
]
for d in dirs:
    if os.path.exists(d):
        print(f"✅ {d} 目录存在")
    else:
        print(f"❌ {d} 目录缺失")

print("\n🎉 Day3 项目骨架环境测试完成！")
