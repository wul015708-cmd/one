import sys
import subprocess
import importlib.util
import os
import warnings
import time

# --- 1. 自动依赖安装逻辑 ---
def check_and_install_packages():
    """检测并自动安装缺失的依赖库"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'openpyxl']
    install_needed = False
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            print(f"[{package}] 未检测到，正在自动安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                install_needed = True
            except Exception as e:
                print(f"安装 {package} 失败: {e}")
    
    if install_needed:
        print("依赖库安装完成，正在刷新环境...")
        importlib.invalidate_caches()

# 执行安装检查
check_and_install_packages()

# --- 2. 自动启动 Streamlit ---
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            print("正在启动 Streamlit 可视化界面...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
            sys.exit()
    except ImportError:
        pass

# --- 3. 导入核心库 ---
import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

# 忽略警告
warnings.filterwarnings('ignore')

# --- 4. 配置中文字体 ---
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 5. 核心算法：Mantel Test ---
def fast_mantel_test(X, Y, permutations=999):
    """Mantel Test 核心算法 (Numpy 加速版)"""
    x = np.array(X)
    y = np.array(Y)
    
    # 确保没有 NaN
    if np.isnan(x).any() or np.isnan(y).any():
        return 0.0, 1.0

    n = int(np.sqrt(len(x) * 2)) + 1
    X_mat = squareform(x)
    Y_mat = squareform(y)
    idx = np.tril_indices(n, k=-1)
    x_vec = X_mat[idx]
    y_vec = Y_mat[idx]
    
    x_mean = np.mean(x_vec)
    y_mean = np.mean(y_vec)
    x_std = np.std(x_vec)
    y_std = np.std(y_vec)
    
    if x_std == 0 or y_std == 0: return 0.0, 1.0

    r_obs = np.mean((x_vec - x_mean) * (y_vec - y_mean)) / (x_std * y_std)

    larger = 0
    Y_perm = Y_mat.copy()
    
    # 优化：如果在数据量极大的情况下，适当减少置换开销
    # 但为了准确性，这里还是保持标准算法
    for _ in range(permutations):
        perm_idx = np.random.permutation(n)
        Y_shuffled = Y_perm[perm_idx][:, perm_idx]
        y_vec_shuffled = Y_shuffled[idx]
        r_perm = np.mean((x_vec - x_mean) * (y_vec_shuffled - y_mean)) / (x_std * y_std)
        if r_perm >= r_obs: larger += 1

    p_value = (larger + 1) / (permutations + 1)
    return r_obs, p_value

# --- 6. 辅助函数：生成演示数据 ---
def generate_demo_data():
    np.random.seed(42)
    n = 150 # 增加演示数据量
    data = {
        'SampleID': np.repeat([f'S{i}' for i in range(1, 16)], 10), # 15个样本，每个10次重复
        '株高(cm)': np.random.uniform(100, 200, n),
        '冠幅(cm)': np.random.uniform(50, 150, n),
        '茎围(cm)': np.random.uniform(10, 30, n),
        '土壤pH': np.random.uniform(5, 8, n),
    }
    data['物种多样性'] = data['株高(cm)'] * 0.5 + np.random.normal(0, 10, n)
    data['生物量'] = data['冠幅(cm)'] * -0.3 + np.random.normal(0, 5, n)
    data['有机质含量'] = np.random.uniform(0, 100, n)
    data['丰富度指数'] = np.random.uniform(0, 10, n)
    return pd.DataFrame(data)

# --- 7. Streamlit 主程序 ---
st.set_page_config(page_title="Mantel Heatmap Generator", layout="wide")

st.title("生态学 Mantel Test 网络热图生成器")
st.markdown("""
**特别说明：**
* **大数据量支持**：已优化读取逻辑，确保读取所有行。
* **重复数据处理**：如果你每个样本有多行数据（重复测量），请使用下方的**“数据聚合”**功能求均值后再分析。
""")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 绘图参数")
    with st.expander("统计设置", expanded=False):
        permutations = st.number_input("置换次数", value=999, step=100)
    
    st.subheader("1. 颜色风格")
    color_map = st.selectbox("热图配色", ["RdBu_r", "coolwarm", "viridis", "PiYG"], index=0)
    
    st.subheader("2. 线条样式 (显著性)")
    col_w, col_c = st.columns([1, 1])
    with col_w:
        lw_p001 = st.number_input("P < 0.001 粗细", value=3.0, step=0.5)
        lw_p01 = st.number_input("P < 0.01 粗细", value=1.5, step=0.5)
        lw_p05 = st.number_input("P < 0.05 粗细", value=0.5, step=0.5)
    with col_c:
        c_p001 = st.color_picker("P<0.001 颜色", "#2E8B57") 
        c_p01 = st.color_picker("P<0.01 颜色", "#FFA500") 
        c_p05 = st.color_picker("P<0.05 颜色", "#D3D3D3") 

# --- 主界面：数据导入模块 ---
st.markdown("---")
st.subheader("📂 第一步：数据导入")

data_source = st.radio("选择数据来源:", ["上传本地文件", "使用演示数据 (Demo)"], horizontal=True)

raw_df = None

if data_source == "上传本地文件":
    uploaded_file = st.file_uploader("拖拽或点击上传文件 (CSV / Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            # 清理列名
            raw_df.columns = raw_df.columns.str.strip()
            st.success(f"✅ 文件读取成功！共检测到 **{raw_df.shape[0]}** 行数据。")
        except Exception as e:
            st.error(f"读取文件失败: {e}")
else:
    st.info("已加载内置演示数据（包含重复测量结构），用于展示功能效果。")
    raw_df = generate_demo_data()
    st.success(f"✅ 演示数据加载成功！共 **{raw_df.shape[0]}** 行数据。")

# --- 数据预处理模块 (关键更新) ---
df = None
if raw_df is not None:
    # 1. 完整数据预览
    with st.expander("📊 原始数据预览 (点击展开/折叠)", expanded=True):
        st.dataframe(raw_df, use_container_width=True)
        st.caption(f"当前显示所有 {raw_df.shape[0]} 行数据。如果行数很多，可以通过表格右侧滚动条查看。")

    st.markdown("---")
    st.subheader("🔧 第二步：数据预处理 (可选)")
    
    col_group1, col_group2 = st.columns([1, 2])
    
    with col_group1:
        st.info("💡 **提示**：如果您的数据每个样本有多行（例如每个植株测了100片叶子），建议先进行聚合（求均值），否则 Mantel Test 计算会非常慢且结果可能不准确。")
        need_aggregation = st.checkbox("我要对数据进行分组求均值 (Aggregation)", value=False)
    
    if need_aggregation:
        with col_group2:
            # 尝试自动识别非数值列作为分组列
            non_numeric_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
            # 同时也加入所有列供选择，以防样本ID是数字
            all_cols = raw_df.columns.tolist()
            
            group_col = st.selectbox("选择用于分组的列 (例如：样本编号/SampleID)", all_cols, index=0 if non_numeric_cols else 0)
            
            if group_col:
                try:
                    # 分组求均值
                    df_agg = raw_df.groupby(group_col).mean(numeric_only=True).reset_index()
                    st.success(f"聚合完成！数据从 **{raw_df.shape[0]}** 行合并为 **{df_agg.shape[0]}** 行样本。")
                    with st.expander("查看聚合后的数据"):
                        st.dataframe(df_agg)
                    df = df_agg
                except Exception as e:
                    st.error(f"聚合失败: {e}")
                    df = raw_df
    else:
        df = raw_df.copy()
        # 如果数据量过大，给予警告
        if df.shape[0] > 2000:
            st.warning(f"⚠️ 注意：当前数据量较大 ({df.shape[0]} 行)。Mantel Test 包含置换检验，计算可能需要较长时间，请耐心等待。")

    # --- 变量选择与分析 ---
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ 数据中未检测到数值型变量，无法进行分析。请检查数据格式。")
        else:
            st.markdown("---")
            st.subheader("🛠️ 第三步：变量选择")
            c1, c2 = st.columns(2)
            
            # 智能匹配
            potential_net_vars = [
                '株高(cm)', '冠幅(cm)', '茎围(cm)', '茎围（cm）', '土壤pH',
                '株高', '冠幅', '茎围', 'PC1', 'PC2', '第三枝长', '第三枝宽'
            ]
            default_net = [c for c in potential_net_vars if c in numeric_cols]
            
            with c1:
                st.markdown("**1. 网络节点变量 (左下角)**")
                st.caption("通常为环境因子，如：株高、土壤理化性质等")
                network_vars = st.multiselect("选择网络变量", numeric_cols, default=default_net)
                
            with c2:
                st.markdown("**2. 热图矩阵变量 (右上角)**")
                st.caption("通常为响应变量，如：物种多样性、生物量等")
                remaining = [c for c in numeric_cols if c not in network_vars]
                heatmap_vars = st.multiselect("选择热图变量", numeric_cols, default=remaining)

            # 校验与运行
            if network_vars and heatmap_vars:
                st.markdown("---")
                if st.button("🚀 开始分析并绘图", type="primary", use_container_width=True):
                    
                    # 进度条
                    progress = st.progress(0)
                    status = st.empty()
                    
                    try:
                        # 数据准备
                        combined_df = df[network_vars + heatmap_vars].dropna()
                        if len(combined_df) < 5:
                            st.error("有效样本量过少 (<5)，请检查数据是否有大量缺失值。")
                            st.stop()
                        
                        # 1. Pearson Matrix
                        status.text("Step 1/3: 计算 Pearson 相关性热图...")
                        heatmap_data = df[heatmap_vars]
                        corr_matrix = heatmap_data.corr(method='pearson')
                        progress.progress(30)
                        
                        # 2. Mantel Test
                        status.text(f"Step 2/3: 进行 Mantel Test ({permutations}次置换)...")
                        mantel_results = []
                        total_pairs = len(network_vars) * len(heatmap_vars)
                        count = 0
                        
                        for net_var in network_vars:
                            dist_A = pdist(combined_df[[net_var]], metric='euclidean')
                            for heat_var in heatmap_vars:
                                dist_B = pdist(combined_df[[heat_var]], metric='euclidean')
                                r_val, p_val = fast_mantel_test(dist_A, dist_B, permutations)
                                
                                mantel_results.append({'source': net_var, 'target': heat_var, 'r': r_val, 'p': p_val})
                                count += 1
                                    
                        mantel_df = pd.DataFrame(mantel_results)
                        progress.progress(70)
                        status.text("Step 3/3: 绘制组合图表...")
                        
                        # --- 绘图 ---
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111)
                        ax.set_aspect('equal')
                        ax.axis('off')
                        
                        n = len(heatmap_vars)
                        cmap = plt.get_cmap(color_map)
                        norm = plt.Normalize(-1, 1)
                        target_coords = {}
                        
                        # 绘制热图 (右上三角)
                        for i, row_var in enumerate(heatmap_vars):
                            for j, col_var in enumerate(heatmap_vars):
                                if j >= i:
                                    x = j
                                    y = n - 1 - i
                                    val = corr_matrix.loc[row_var, col_var]
                                    
                                    # A. 对角线坐标 (保留但不显示文字)
                                    if i == j:
                                        target_coords[row_var] = (x, y)
                                    
                                    # B. 顶部标签 (列名)
                                    if i == 0:
                                        ax.text(x, y + 0.6, col_var, ha='left', va='bottom', rotation=45, fontsize=10)

                                    # C. 右侧标签 (行名)
                                    if j == n - 1:
                                        ax.text(x + 0.6, y, row_var, ha='left', va='center', rotation=0, fontsize=10)

                                    # D. 绘制方格背景
                                    grid_rect = patches.Rectangle(
                                        (x - 0.5, y - 0.5), 1, 1, 
                                        fill=False, 
                                        edgecolor='#cccccc', 
                                        linewidth=0.5,
                                        linestyle='-'
                                    )
                                    ax.add_patch(grid_rect)

                                    # E. 绘制颜色块
                                    size = abs(val) * 0.92
                                    rect = patches.Rectangle((x - size/2, y - size/2), size, size, 
                                                        facecolor=cmap(norm(val)), edgecolor='none')
                                    ax.add_patch(rect)
                        
                        # 绘制 Colorbar
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
                        cbar.set_label("Pearson Correlation", rotation=270, labelpad=15)
                        
                        # 绘制网络节点 (左下角)
                        net_x = -3
                        # 优化节点间距，防止重叠
                        if len(network_vars) > 1:
                            net_y_coords = np.linspace(1, n-2, len(network_vars))[::-1]
                        else:
                            net_y_coords = [n/2]
                        
                        source_coords = {}
                        for idx, var in enumerate(network_vars):
                            y = net_y_coords[idx]
                            source_coords[var] = (net_x, y)
                            ax.scatter(net_x, y, s=120, color='#555555', zorder=10)
                            ax.text(net_x - 0.3, y, var, ha='right', va='center', fontsize=11, fontweight='bold')
                        
                        # 绘制连线
                        valid_links = mantel_df[mantel_df['p'] < 0.05].copy()
                        valid_links.sort_values('p', ascending=False, inplace=True)
                        
                        for _, row in valid_links.iterrows():
                            p_v = row['p']
                            if p_v < 0.001:   lw, c = lw_p001, c_p001
                            elif p_v < 0.01:  lw, c = lw_p01, c_p01
                            else:             lw, c = lw_p05, c_p05
                            
                            arrow = patches.FancyArrowPatch(
                                posA=source_coords[row['source']], 
                                posB=target_coords[row['target']],
                                connectionstyle="arc3,rad=0.2",
                                color=c, linewidth=lw, alpha=0.75, zorder=1
                            )
                            ax.add_patch(arrow)
                        
                        # 图例
                        legend_elements = [
                            Line2D([0], [0], color=c_p001, lw=lw_p001, label='P < 0.001'),
                            Line2D([0], [0], color=c_p01, lw=lw_p01, label='P < 0.01'),
                            Line2D([0], [0], color=c_p05, lw=lw_p05, label='P < 0.05'),
                        ]
                        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.2, 1.05),
                                frameon=False, title="Mantel's P")
                        
                        # 调整视图
                        ax.set_xlim(net_x - 5, n + 3)
                        ax.set_ylim(-1, n + 3)
                        
                        st.pyplot(fig)
                        progress.progress(100)
                        status.success("✅ 绘图完成！")
                        
                        # 结果下载
                        csv = mantel_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 下载 Mantel 分析结果 (CSV)", csv, "mantel_results.csv", "text/csv")

                    except Exception as e:
                        st.error(f"分析过程中发生错误: {e}")
                        st.markdown("建议检查：数据中是否包含非数值字符？是否所有列都已对齐？")

            elif not network_vars or not heatmap_vars:
                st.info("👈 请在上方选择变量以开始分析...")