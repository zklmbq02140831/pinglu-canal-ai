import rasterio
import matplotlib.pyplot as plt

dem_path = "data/raw/srtm_58_08.img"
with rasterio.open(dem_path) as src:
    dem_data = src.read(1)
    print(f"DEM 宽：{src.width}，高：{src.height}")
    print(f"高程最小值：{dem_data.min()} m")
    print(f"高程最大值：{dem_data.max()} m")

plt.imshow(dem_data, cmap="terrain")
plt.title("广西SRTM 90m DEM")
plt.colorbar(label="高程(m)")
plt.show()
