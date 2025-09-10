import os
import numpy as np
import netCDF4 as nc
from netCDF4 import num2date
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib
import streamlit as st
from datetime import datetime
import pandas as pd

# 基础配置：适配Streamlit环境
matplotlib.use('Agg')  # 非交互式后端，避免绘图冲突
import matplotlib.pyplot as plt

# 设置中文字体（解决Streamlit中文乱码）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100  # 适配网页显示分辨率


# --------------------------
# 1. Streamlit页面初始化
# --------------------------
st.set_page_config(
    page_title="3D DBSCAN极端降水事件识别",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏标题
st.sidebar.title("参数配置")
st.sidebar.markdown("---")


# --------------------------
# 2. 核心工具函数（保留原逻辑，适配Streamlit）
# --------------------------
def read_era5_data(file_obj):
    """读取用户上传的ERA5 netCDF文件（适配Streamlit文件对象）"""
    try:
        # 从Streamlit上传的文件对象读取，无需保存到本地
        ds = nc.Dataset('temp.nc', mode='r', memory=file_obj.read())
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        time = ds.variables['valid_time'][:]
        
        # 读取降水变量（支持用户选择变量名）
        precip_var_name = st.session_state.get('precip_var_name', 'tp')
        if precip_var_name not in ds.variables:
            raise ValueError(f"文件中未找到变量 {precip_var_name}，可用变量：{list(ds.variables.keys())}")
        
        tp = ds.variables[precip_var_name][:] * 1000  # 转换为mm/h
        time_units = ds.variables['valid_time'].units
        ds.close()
        return lon, lat, time, tp, time_units
    except Exception as e:
        st.error(f"数据读取失败：{str(e)}")
        return None, None, None, None, None


def extract_extreme_precip(file_obj, config):
    """提取极端降水点（适配Streamlit状态管理）"""
    with st.spinner("正在读取并处理ERA5数据..."):
        lon, lat, time, tp, time_units = read_era5_data(file_obj)

        if lon is None:
            st.error("无法读取数据，请检查文件格式或变量名")
            return None

        # 稳健的时间转换（保留原逻辑）
        try:
            time_datetime_py = num2date(time, units=time_units)
            time_datetime = np.array(time_datetime_py, dtype='datetime64[h]')
            st.success("时间转换成功（使用num2date方法）")
        except Exception as e1:
            try:
                time_datetime = np.array(time, dtype='datetime64[h]')
                st.success("时间转换成功（使用直接转换方法）")
            except Exception as e2:
                try:
                    time_seconds = time.astype(float)
                    time_datetime = np.array('1970-01-01', dtype='datetime64[s]') + \
                                    np.array(time_seconds, dtype='timedelta64[s]')
                    time_datetime = time_datetime.astype('datetime64[h]')
                    st.success("时间转换成功（使用Unix时间戳解析）")
                except Exception as e3:
                    st.error(
                        f"时间格式转换失败：\n方法1: {str(e1)}\n方法2: {str(e2)}\n方法3: {str(e3)}"
                    )
                    return None

        # 计算相对于起始时间的天数
        time_delta_hours = (time_datetime - time_datetime[0]).astype('timedelta64[h]').astype(int)
        time_days = time_delta_hours / 24.0

        # 筛选研究区域（用户可配置）
        lon_mask = (lon >= config['domain'][0]) & (lon <= config['domain'][1])
        lat_mask = (lat >= config['domain'][2]) & (lat <= config['domain'][3])
        lon_indices = np.where(lon_mask)[0]
        lat_indices = np.where(lat_mask)[0]

        # 提取极端降水点
        st.info(f"根据阈值 {config['unified_threshold']} mm/h 筛选极端降水点...")
        extreme_points = []

        if tp.ndim == 3:
            total_time_steps = tp.shape[0]
            progress_bar = st.progress(0, text="筛选极端点中...")
            for t_idx in range(total_time_steps):
                # 更新进度条
                progress_bar.progress((t_idx + 1) / total_time_steps, 
                                     text=f"处理时间步 {t_idx+1}/{total_time_steps}")
                
                time_slice = tp[t_idx, :, :]
                for lat_idx in lat_indices:
                    for lon_idx in lon_indices:
                        if time_slice[lat_idx, lon_idx] > config['unified_threshold']:
                            extreme_points.append([
                                lon[lon_idx],  # 经度
                                lat[lat_idx],  # 纬度
                                time_days[t_idx]
                            ])
            progress_bar.empty()

        if not extreme_points:
            st.warning(f"未检测到极端降水点（当前阈值: {config['unified_threshold']} mm/h）")
            return None
        st.success(f"共筛选出 {len(extreme_points)} 个极端降水点")

        # 保存到会话状态，供后续步骤使用
        st.session_state['data_dict'] = {
            'features': np.array(extreme_points, dtype=float),
            'raw_precip': tp,
            'lon': lon,
            'lat': lat,
            'time': time,
            'time_days': time_days,
            'base_datetime': time_datetime[0],
            'time_datetime': time_datetime
        }
        return st.session_state['data_dict']


# --------------------------
# 3. 3D DBSCAN聚类类（适配Streamlit可视化）
# --------------------------
class ExtremeEventCluster:
    def __init__(self, config):
        self.config = config
        self.eps = config['dbscan_eps']
        self.min_pts = config['dbscan_min_pts']
        self.time_scale = config['time_scale']
        self.min_duration = config['min_duration']
        self.min_total_grid_count = config['min_total_grid_count']
        self.actual_time_eps = (self.eps / self.time_scale) * 24

        # 结果保存路径（Streamlit中优先内存展示，支持下载）
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.result_key = f"results_{self.timestamp}"

    def cluster(self, data_dict):
        """执行聚类（核心逻辑不变，添加Streamlit状态更新）"""
        with st.spinner("正在执行3D DBSCAN聚类..."):
            features = data_dict['features'].copy()
            features[:, 2] = features[:, 2] * self.time_scale  # 时间标准化

            # 执行DBSCAN
            db = DBSCAN(eps=self.eps, min_samples=self.min_pts, metric='euclidean').fit(features)
            labels = db.labels_
            label_counts = Counter(labels)

            n_noise = label_counts.get(-1, 0)
            n_initial_events = len(label_counts) - 1 if -1 in label_counts else len(label_counts)
            
            # 在页面显示聚类基础结果
            col1, col2, col3 = st.columns(3)
            col1.metric("初始事件数", n_initial_events)
            col2.metric("孤立点数", n_noise)
            col3.metric("总极端点数", len(features))

            if n_initial_events == 0:
                st.warning("未识别到有效事件，请调小eps或min_pts参数")
                return None

            # 筛选事件（保留原逻辑）
            st.subheader("事件筛选（基于持续时间和影响范围）")
            qualified_events = {}
            event_id = 0
            filter_log = []

            for initial_id in range(n_initial_events):
                if initial_id not in label_counts:
                    continue

                event_mask = (labels == initial_id)
                event_features = data_dict['features'][event_mask]

                # 检查持续时间
                time_vals = event_features[:, 2]
                duration_hours = (time_vals.max() - time_vals.min()) * 24 + 1
                # 检查影响网格数
                grid_points = np.round(event_features[:, :2], 2)
                total_grid_count = len(np.unique(grid_points, axis=0))

                if duration_hours <= self.min_duration or total_grid_count < self.min_total_grid_count:
                    filter_log.append(
                        f"❌ 事件{initial_id}：持续{duration_hours:.1f}h / 网格{total_grid_count}个 → 不满足条件"
                    )
                    continue

                qualified_events[event_id] = event_features
                filter_log.append(
                    f"✅ 事件{event_id}：持续{duration_hours:.1f}h / 网格{total_grid_count}个 → 保留"
                )
                event_id += 1

            # 显示筛选日志
            with st.expander("查看筛选详情", expanded=False):
                for log in filter_log:
                    st.write(log)

            n_qualified = len(qualified_events)
            st.success(f"事件筛选完成：共保留 {n_qualified} 个符合条件的极端降水事件")

            if n_qualified == 0:
                st.warning("未保留任何事件，请调整筛选条件（如降低min_duration）")
                return None

            # 保存结果到会话状态
            self.qualified_events = qualified_events
            st.session_state[self.result_key] = qualified_events
            st.session_state['cluster_obj'] = self  # 保存聚类对象供后续可视化
            return qualified_events

    def visualize_3d_spacetime(self, data_dict):
        """3D时空图（适配Streamlit显示）"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 颜色配置
        n_events = len(self.qualified_events)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_events, 20)))
        if n_events > 20:
            colors = np.tile(colors, (n_events // 20 + 1, 1))[:n_events]

        # 绘制每个事件
        for event_idx, (event_id, event_features) in enumerate(self.qualified_events.items()):
            ax.scatter(
                event_features[:, 1],  # X轴：纬度
                event_features[:, 0],  # Y轴：经度
                event_features[:, 2],  # Z轴：时间
                color=colors[event_idx],
                label=f'事件 {event_id}',
                alpha=0.7,
                s=30
            )

        # 坐标轴设置
        ax.set_xlim([self.config['domain'][3], self.config['domain'][2]])  # 纬度：右到左递增
        ax.set_xlabel('纬度 (°N)', fontsize=11)
        ax.set_ylim([self.config['domain'][0], self.config['domain'][1]])  # 经度：左到右递增
        ax.set_ylabel('经度 (°E)', fontsize=11)

        # 时间轴
        z_min, z_max = data_dict['time_days'].min(), data_dict['time_days'].max()
        z_ticks = np.arange(np.floor(z_min), np.ceil(z_max) + 1, 1)
        z_tick_labels = [f'{int(day)}日' for day in z_ticks]
        ax.set_zlim([z_min, z_max])
        ax.set_zticks(z_ticks)
        ax.set_zticklabels(z_tick_labels)
        ax.set_zlabel('时间', fontsize=11)

        # 标题和图例
        plt.title(
            f'极端降水事件3D时空分布（共{n_events}个事件）',
            fontsize=14, pad=20
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # 在Streamlit中显示图片
        st.pyplot(fig)
        plt.close()

        # 提供下载功能
        self.save_fig_to_download(fig, "3d_spacetime_distribution.png")

    def visualize_2d(self, data_dict, plot_type):
        """通用2D可视化（经度-时间/纬度-时间/空间投影）"""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # 颜色配置
        n_events = len(self.qualified_events)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_events, 20)))
        if n_events > 20:
            colors = np.tile(colors, (n_events // 20 + 1, 1))[:n_events]

        # 绘制每个事件
        for event_idx, (event_id, event_features) in enumerate(self.qualified_events.items()):
            if plot_type == 'lon_time':
                x, y = event_features[:, 0], event_features[:, 2]
                x_label, y_label = '经度 (°E)', '时间'
                title = f'极端降水事件经度-时间分布（共{n_events}个事件）'
                x_lim = [self.config['domain'][0], self.config['domain'][1]]
            elif plot_type == 'lat_time':
                x, y = event_features[:, 1], event_features[:, 2]
                x_label, y_label = '纬度 (°N)', '时间'
                title = f'极端降水事件纬度-时间分布（共{n_events}个事件）'
                x_lim = [self.config['domain'][2], self.config['domain'][3]]
            elif plot_type == 'spatial':
                x, y = event_features[:, 0], event_features[:, 1]
                x_label, y_label = '经度 (°E)', '纬度 (°N)'
                title = f'极端降水事件空间投影（共{n_events}个事件）'
                x_lim = [self.config['domain'][0], self.config['domain'][1]]
                y_lim = [self.config['domain'][2], self.config['domain'][3]]
            else:
                return

            ax.scatter(x, y, color=colors[event_idx], label=f'事件 {event_id}', alpha=0.7, s=30)

        # 坐标轴配置
        ax.set_xlim(x_lim)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        if plot_type == 'spatial':
            ax.set_ylim(y_lim)
        else:
            # 时间轴配置
            z_min, z_max = data_dict['time_days'].min(), data_dict['time_days'].max()
            z_ticks = np.arange(np.floor(z_min), np.ceil(z_max) + 1, 1)
            z_tick_labels = [f'{int(day)}日' for day in z_ticks]
            ax.set_ylim([z_min, z_max])
            ax.set_yticks(z_ticks)
            ax.set_yticklabels(z_tick_labels)

        # 网格和图例
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')
        plt.title(title, fontsize=14, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        # 在Streamlit中显示
        st.pyplot(fig)
        plt.close()

        # 提供下载
        filename = f"{plot_type}_distribution.png"
        self.save_fig_to_download(fig, filename)

    def save_fig_to_download(self, fig, filename):
        """将图片保存到内存，提供Streamlit下载按钮"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # 添加下载按钮
        st.download_button(
            label=f"下载 {filename}",
            data=buf,
            file_name=filename,
            mime="image/png",
            key=f"download_{filename}_{self.timestamp}"
        )

    def export_events_to_csv(self):
        """导出事件详情为CSV（支持批量下载）"""
        all_events_df = pd.DataFrame()
        for event_id, event_features in self.qualified_events.items():
            # 构建事件DataFrame
            df = pd.DataFrame({
                'longitude': event_features[:, 0].round(2),
                'latitude': event_features[:, 1].round(2),
                'time_days': event_features[:, 2].round(2),
                'event_id': event_id
            })
            # 转换时间为可读格式
            base_dt = self.config['base_datetime']
            df['datetime'] = [base_dt + np.timedelta64(int(day*24), 'h') for day in df['time_days']]
            all_events_df = pd.concat([all_events_df, df], ignore_index=True)

        # 保存到内存
        import io
        buf = io.StringIO()
        all_events_df.to_csv(buf, index=False, encoding='utf-8-sig')
        buf.seek(0)

        # 下载按钮
        st.download_button(
            label=f"下载所有事件详情（CSV）",
            data=buf,
            file_name=f"extreme_rain_events_{self.timestamp}.csv",
            mime="text/csv",
            key=f"download_csv_{self.timestamp}"
        )
        return all_events_df


# --------------------------
# 4. Streamlit交互式界面（核心流程）
# --------------------------
def main():
    # 主标题
    st.title("🌧️ 3D DBSCAN极端降水事件识别系统")
    st.markdown("基于ERA5小时降水数据，通过3D DBSCAN算法识别时空连续的极端降水事件")
    st.markdown("---")

    # 步骤1：上传数据
    st.subheader("步骤1：上传ERA5 NetCDF数据")
    uploaded_file = st.file_uploader("选择NC文件（支持ERA5降水数据）", type=["nc", "netcdf"])
    
    # 步骤2：配置参数（侧边栏）
    st.sidebar.subheader("数据参数")
    # 降水变量名（默认tp，用户可修改）
    precip_var_name = st.sidebar.text_input("降水变量名", value="tp", key="precip_var_name")
    # 研究区域（默认中国区域）
    st.sidebar.subheader("研究区域")
    domain = {
        'min_lon': st.sidebar.number_input("最小经度", value=73.0, step=0.5),
        'max_lon': st.sidebar.number_input("最大经度", value=135.0, step=0.5),
        'min_lat': st.sidebar.number_input("最小纬度", value=3.0, step=0.5),
        'max_lat': st.sidebar.number_input("最大纬度", value=54.0, step=0.5)
    }

    # DBSCAN参数
    st.sidebar.subheader("DBSCAN聚类参数")
    dbscan_params = {
        'unified_threshold': st.sidebar.slider("极端降水阈值（mm/h）", min_value=1.0, max_value=50.0, value=20.0, step=0.5),
        'dbscan_eps': st.sidebar.slider("EPS（空间邻域，度）", min_value=1.0, max_value=10.0, value=5.0, step=0.1),
        'dbscan_min_pts': st.sidebar.slider("MIN_PTS（核心点最小点数）", min_value=3, max_value=30, value=10, step=1),
        'time_scale': st.sidebar.slider("时间缩放系数", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    }

    # 事件筛选参数
    st.sidebar.subheader("事件筛选条件")
    filter_params = {
        'min_duration': st.sidebar.number_input("最小持续时间（小时）", min_value=6, max_value=100, value=48, step=6),
        'min_total_grid_count': st.sidebar.number_input("最小影响网格数", min_value=10, max_value=200, value=50, step=5)
    }

    # 合并所有配置
    config = {**domain, **dbscan_params, **filter_params}

    # 步骤3：处理数据（极端点提取）
    if uploaded_file is not None:
        st.subheader("步骤2：提取极端降水点")
        if st.button("开始提取极端点", key="extract_btn"):
            # 重置之前的结果
            st.session_state.pop('data_dict', None)
            st.session_state.pop('cluster_obj', None)
            
            # 执行极端点提取
            data_dict = extract_extreme_precip(uploaded_file, config)

    # 步骤4：聚类分析（数据提取完成后显示）
    if 'data_dict' in st.session_state:
        st.markdown("---")
        st.subheader("步骤3：3D DBSCAN聚类分析")
        data_dict = st.session_state['data_dict']
        
        if st.button("开始聚类", key="cluster_btn"):
            # 创建聚类对象并执行
            cluster_obj = ExtremeEventCluster(config)
            qualified_events = cluster_obj.cluster(data_dict)
            
            # 聚类完成后显示可视化
            if qualified_events is not None:
                st.markdown("---")
                st.subheader("步骤4：结果可视化")
                
                # 3D时空图
                st.subheader("3D时空分布")
                cluster_obj.visualize_3d_spacetime(data_dict)
                
                # 2D图表（分栏显示）
                st.subheader("2D分布图表")
                tab1, tab2, tab3 = st.tabs(["经度-时间", "纬度-时间", "空间投影"])
                with tab1:
                    cluster_obj.visualize_2d(data_dict, 'lon_time')
                with tab2:
                    cluster_obj.visualize_2d(data_dict, 'lat_time')
                with tab3:
                    cluster_obj.visualize_2d(data_dict, 'spatial')
                
                # 导出结果
                st.markdown("---")
                st.subheader("步骤5：导出结果")
                cluster_obj.export_events_to_csv()


if __name__ == "__main__":
    main()
