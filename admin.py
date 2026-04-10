import streamlit as st
import pandas as pd
import sqlite3
import sys
import os

# --- 页面配置 ---
st.set_page_config(
    page_title="数据库管理后台",
    page_icon="🧬",
    layout="wide"  # 宽屏模式，方便展示多列数据
)

# --- 标题与侧边栏 ---
st.title("🧬 LatticeGPT 核心数据库管理")
st.markdown("---")

DB_PATH = "materials.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# --- 侧边栏：全局控制 ---
with st.sidebar:
    st.header("⚙️ 全局设置")
    limit_num = st.slider("最大显示行数 (Limit)", 10, 5000, 50)
    st.info("💡 提示：此面板用于辅助验证 Agent 的检索结果和数据质量。")

# --- 核心功能区 (Tabs) ---
# 🔧 修改：只保留两个 Tab，移除了可视化
tab1, tab2 = st.tabs(["📋 数据全貌 (All Data)", "🛠️ SQL 实验室 (Debug)"])

# ===============================================
# 功能 1: 数据浏览 (显示所有列)
# ===============================================
# --- Tab 1 修改版代码 ---
with tab1:
    st.subheader("数据库全字段浏览")

    # 1. 获取所有可选的结构类型 (为了做下拉菜单)
    conn = get_connection()
    try:
        # 查一下数据库里有哪些类型
        types_df = pd.read_sql("SELECT DISTINCT lattice_type1 FROM lattice_data", conn)
        all_types = types_df['lattice_type1'].tolist()
    except:
        all_types = []
    finally:
        conn.close()

    # 2. 布局筛选器 (增加一行)
    col1, col2, col3 = st.columns(3)

    with col1:
        # 多选框：允许用户选多个类型
        selected_types = st.multiselect("筛选: 结构类型", all_types, default=[])

    with col2:
        min_c11 = st.number_input("筛选: 最小 C11 (GPa)", value=0.0)

    with col3:
        max_density = st.number_input("筛选: 最大相对密度", value=1.0, step=0.01)

    # 3. 动态构建 SQL
    # 基础 SQL
    base_sql = "SELECT * FROM lattice_data WHERE C11 >= ? AND relative_density <= ?"
    params = [min_c11, max_density]

    # 如果用户选了类型，加到 SQL 里
    if selected_types:
        placeholders = ','.join(['?'] * len(selected_types))
        base_sql += f" AND lattice_type1 IN ({placeholders})"
        params.extend(selected_types)

    base_sql += f" LIMIT {limit_num}"

    # 4. 执行查询
    conn = get_connection()
    try:
        df = pd.read_sql(base_sql, conn, params=params)
        st.dataframe(df, use_container_width=True)
        st.caption(f"🔍 查询到 {len(df)} 条数据")
    finally:
        conn.close()
# ===============================================
# 功能 2: SQL 实验室 (调试 Agent 的 SQL)
# ===============================================
with tab2:
    st.subheader("SQL 调试控制台")
    st.markdown("""
    **功能作用**
    当 Agent 回答错误时，查看终端里 Agent 生成的 SQL，并粘贴到这里运行。
    - 如果这里报错：说明 Agent 生成的 SQL 语法有问题（Prompt 需要优化）。
    - 如果这里结果为空：说明数据库里确实没有符合条件的数据。
    """)

    # 默认给一个复杂的聚合查询例子，展示 SQL 的强大
    default_sql = "SELECT lattice_type1, COUNT(*) as count, AVG(C11) as avg_stiffness FROM lattice_data GROUP BY lattice_type1"
    sql_input = st.text_area("输入 SQL 语句", value=default_sql, height=150)

    run_btn = st.button("▶️ 执行 SQL", type="primary")

    if run_btn:
        conn = get_connection()
        try:
            results = pd.read_sql(sql_input, conn)
            st.success("✅ 执行成功")
            st.dataframe(results, use_container_width=True)
        except Exception as e:
            st.error(f"❌ SQL 语法错误: {e}")
        finally:
            conn.close()

# --- 页脚 ---
st.markdown("---")
st.caption("LatticeGPT Admin Dashboard | Powered by Streamlit")