import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import io  # 新增：导入io模块处理文件流

# --------------------------
# 初始化：获取项目根目录
# --------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
results_dir = os.path.join(project_root, "results")

# --------------------------
# 1. 系统首页界面
# --------------------------
st.title("三维极端降水事件识别系统（修复版）")
st.write("### 操作指南")
st.write("1. 左侧上传数据（支持NetCDF格式）→ 2. 调整DBSCAN参数 → 3. 运行识别 → 4. 查看/下载结果")
st.write("### 识别规则")
st.write("固定阈值20mm/h（降水>20mm/h为极端点），用3D DBSCAN聚类识别事件时空演变")

# --------------------------
# 2. 侧边栏：用户操作区
# --------------------------
st.sidebar.header("1. 上传数据")
uploaded_file = st.sidebar.file_uploader("选择NetCDF文件", type=["nc"])

# 若上传了文件，开始处理
if uploaded_file is not None:
    # 读取数据（修复核心：将UploadedFile转换为xarray可识别的格式）
    try:
        # 关键修复：用io.BytesIO处理上传文件
        file_bytes = uploaded_file.getvalue()  # 获取文件字节数据
        ds = xr.open_dataset(io.BytesIO(file_bytes))  # 转换为xarray可读取的流格式

        # 提取核心数据（若你的降水变量名不是precip，需改成实际名，如tp）
        precip = ds["precip"].values  # 降水数据（mm/h）
        lon = ds["lon"].values  # 经度
        lat = ds["lat"].values  # 纬度
        time = ds["time"].values  # 时间
        st.success("✅ 数据读取成功！（已修复文件格式问题）")
    except Exception as e:
        st.error(f"❌ 数据读取失败：{str(e)}")
        st.stop()  # 停止后续流程

    # 显示数据基本信息
    st.write(f"📅 时间范围：{pd.to_datetime(time[0])} ~ {pd.to_datetime(time[-1])}（共{len(time)}小时）")
    st.write(f"🌍 空间范围：经度{lon[0]:.1f}°E~{lon[-1]:.1f}°E，纬度{lat[0]:.1f}°N~{lat[-1]:.1f}°N")

    # --------------------------
    # 3. 侧边栏：DBSCAN参数设置
    # --------------------------
    st.sidebar.header("2. DBSCAN参数")
    eps = st.sidebar.slider(
        "eps（时空邻域半径）",
        min_value=1, max_value=10, value=5,
        help="值越大，事件合并越多；值越小，事件拆分越细"
    )
    min_samples = st.sidebar.slider(
        "min_samples（最小核心点数）",
        min_value=3, max_value=20, value=5,
        help="值越大，识别事件越严格（排除小噪声）"
    )

    # --------------------------
    # 4. 核心逻辑：极端点筛选+3D DBSCAN聚类
    # --------------------------
    if st.sidebar.button("3. 运行事件识别"):
        with st.spinner("🔍 正在计算..."):
            # ① 筛选极端点（固定阈值20mm/h）
            st.subheader("1. 极端点筛选结果")
            extreme_points = []
            for t_idx in range(len(time)):
                for lat_idx in range(len(lat)):
                    for lon_idx in range(len(lon)):
                        if precip[lat_idx, lon_idx, t_idx] > 20:
                            extreme_points.append([
                                lon[lon_idx],
                                lat[lat_idx],
                                t_idx / 24  # 时间转天数，适配3D可视化
                            ])

            # 检查极端点数量
            if not extreme_points:
                st.warning("⚠️ 未找到降水>20mm/h的极端点，请更换数据！")
                st.stop()
            extreme_points = np.array(extreme_points)
            st.write(f"✅ 共筛选极端点：{len(extreme_points)} 个")

            # ② 3D DBSCAN聚类
            st.subheader("2. 3D DBSCAN聚类结果")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = dbscan.fit_predict(extreme_points)  # 事件标签（-1=噪声）
            event_ids = [l for l in set(labels) if l != -1]
            n_events = len(event_ids)
            n_noise = list(labels).count(-1)
            st.write(f"✅ 识别极端事件：{n_events} 个")
            st.write(f"⚠️ 噪声点（孤立极端点）：{n_noise} 个")

            # --------------------------
            # 5. 结果可视化
            # --------------------------
            st.subheader("3. 结果可视化")

            # ① 3D时空分布图
            st.write("📊 3D时空分布（鼠标旋转查看，颜色=事件）")
            fig_3d = go.Figure()
            for e_id in event_ids:
                mask = labels == e_id
                fig_3d.add_trace(go.Scatter3d(
                    x=extreme_points[mask, 0], y=extreme_points[mask, 1], z=extreme_points[mask, 2],
                    mode="markers", name=f"事件{e_id + 1}", marker=dict(size=5)
                ))
            fig_3d.update_layout(
                scene=dict(xaxis_title="经度（°E）", yaxis_title="纬度（°N）", zaxis_title="时间（天）"),
                height=600, legend_title="事件列表"
            )
            st.plotly_chart(fig_3d)

            # ② 空间分布切片图
            st.write("🗺️ 空间分布（时间中点切片）")
            mid_time = extreme_points[:, 2].mean()
            time_mask = (extreme_points[:, 2] >= mid_time - 0.5) & (extreme_points[:, 2] <= mid_time + 0.5)
            plt.figure(figsize=(10, 6))
            for e_id in event_ids:
                mask = (labels == e_id) & time_mask
                plt.scatter(extreme_points[mask, 0], extreme_points[mask, 1], label=f"事件{e_id + 1}", s=50)
            plt.xlabel("经度（°E）"), plt.ylabel("纬度（°N）")
            plt.title(f"时间中点：{mid_time:.2f}天（约第{int(mid_time * 24)}小时）")
            plt.legend(), plt.grid(alpha=0.3)
            st.pyplot(plt)

            # --------------------------
            # 6. 结果下载
            # --------------------------
            st.subheader("4. 结果下载")
            events_df = pd.DataFrame({
                "经度（°E）": extreme_points[:, 0].round(2),
                "纬度（°N）": extreme_points[:, 1].round(2),
                "时间（天）": extreme_points[:, 2].round(2),
                "所属事件": [f"事件{l + 1}" if l != -1 else "噪声点" for l in labels]
            })
            st.download_button(
                "📥 下载事件列表（CSV）",
                data=events_df.to_csv(index=False, encoding="utf-8"),
                file_name="极端降水事件列表.csv",
                mime="text/csv"
            )
            events_df.to_csv(os.path.join(results_dir, "极端降水事件列表.csv"), index=False, encoding="utf-8")
            st.success(f"✅ 本地结果已保存到：{os.path.join(results_dir, '极端降水事件列表.csv')}")

# --------------------------
# 未上传数据时的引导
# --------------------------
else:
    st.info(f"ℹ️ 请在左侧上传数据（测试数据路径：{os.path.join(data_dir, 'test_data.nc')}）")
