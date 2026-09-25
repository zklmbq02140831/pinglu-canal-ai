import pandas as pd

# skiprows=2 跳过前两行大标题，从真正数据行开始读
df = pd.read_excel("data/raw/customs_data.xlsx", skiprows=2)
print("=== 南宁海关进出口数据（清理后）===")
print(df.head())
print(f"\n数据总行数：{len(df)}")
# 查看列名
print("\n列名：")
print(df.columns.tolist())
