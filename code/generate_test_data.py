# 生成模拟降水数据（适配gdal_py38环境，避免路径问题）
import xarray as xr
import numpy as np
import pandas as pd
import os

# 获取项目根目录（确保路径正确，不依赖工作目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "test_data.nc")

# 1. 设置时空范围（10×10网格，3天逐小时）
lon = np.arange(110, 120, 1)  # 经度110°E~119°E
lat = np.arange(20, 30, 1)   # 纬度20°N~29°N
time = pd.date_range("2023-07-01", periods=72, freq="H")  # 72小时

# 2. 生成降水数据（含极端点>20mm/h）
np.random.seed(123)  # 固定随机数，结果可复现
precip = np.random.rand(len(lat), len(lon), len(time)) * 30  # 0-30mm/h
precip[precip > 20] = precip[precip > 20]  # 标记极端点

# 3. 保存为NetCDF（路径用绝对路径，避免PyCharm路径问题）
ds = xr.Dataset(
    data_vars={"precip": (["lat", "lon", "time"], precip)},
    coords={"lat": lat, "lon": lon, "time": time}
)
ds.to_netcdf(data_path)

print(f"测试数据生成完成！路径：{data_path}")