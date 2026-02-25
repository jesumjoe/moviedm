import os
import sys
import ast
from io import StringIO

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from scipy.stats import chi2_contingency
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# ── Page config (only once) ──────────────────────────────────
st.set_page_config(
    page_title="CineMiner · TMDB 5000",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientBG {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Adjust padding for main content */
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

    /* Hide Streamlit toolbar (deploy, etc) but keep header for sidebar toggle */
    [data-testid="stToolbar"] { visibility: hidden !important; }

    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #111827, #020617);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
    }

    .main .block-container { animation: fadeIn 1.2s cubic-bezier(0.16, 1, 0.3, 1); }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(8px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px -10px rgba(139,92,246,0.5);
        border-color: rgba(139,92,246,0.5);
    }

    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        transition: all 0.3s ease-in-out;
        overflow: hidden;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 8px 25px -5px rgba(0,0,0,0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover  { transform: scale(1.05) !important; box-shadow: 0 0 20px rgba(139,92,246,0.6) !important; }
    .stButton > button:active { transform: scale(0.95) !important; }

    .stAlert {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px);
        color: #cbd5e1 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 12px !important;
        border-left: 4px solid #8b5cf6 !important;
        animation: fadeIn 0.8s ease-out;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stDataFrame"]:hover {
        border-color: rgba(139,92,246,0.3);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
    }

    h1 {
        background: linear-gradient(to right, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 { font-weight: 600 !important; color: #f8fafc !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track  { background: rgba(15,23,42,0.5); }
    ::-webkit-scrollbar-thumb  { background: rgba(99,102,241,0.5); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.8); }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
hero_path = os.path.join(current_dir, "hero_section.html")
with open(hero_path, 'r', encoding='utf-8') as f:
    hero_html = f.read()

# ── Navigation State ──────────────────────────────────────────
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "Dashboard"

def set_nav_section(section):
    st.session_state.nav_section = section

def go_back():
    st.session_state.nav_section = "Dashboard"
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Setup")
    uploaded_file = st.file_uploader("Upload tmdb_5000_movies.csv", type=['csv'])

    st.markdown("---")

    st.header("Navigation")
    sections = [
        "Dashboard",
        "Data Overview & EDA",
        "OLAP Analysis",
        "Data Transformation",
        "Association Rules (Apriori + FP-Growth)",
        "PCA",
        "Classification (Decision Tree, Naive Bayes, KNN)",
        "Linear Regression",
        "Custom Code Execution"
    ]
    
    # We assign to a distinct variable and don't tie the key directly to nav_section 
    # to avoid the "cannot be modified after widget instantiated" error.
    selected_section = st.radio(
        "Go to", 
        sections, 
        index=sections.index(st.session_state.nav_section) if st.session_state.nav_section in sections else 0,
        label_visibility="collapsed"
    )

    # Sync state so our buttons can still programmatically change the view
    if selected_section != st.session_state.nav_section:
        st.session_state.nav_section = selected_section
        st.rerun()

# ── Data Upload & Preprocessing ──────────────────────────────
if uploaded_file is not None:
    if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != uploaded_file.name:
        try:
            raw_df = pd.read_csv(uploaded_file)

            pre_metrics = {
                "total_rows": raw_df.shape[0],
                "total_cols": raw_df.shape[1],
                "duplicate_count": raw_df.duplicated().sum(),
                "missing_count": raw_df.isna().sum().sum(),
                "missing_per_col": raw_df.isna().sum()
            }

            with st.spinner("Running global data cleaning and preprocessing..."):
                df = raw_df.copy()

                numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if 'budget' in df.columns:  df['budget']  = df['budget'].fillna(0)
                if 'revenue' in df.columns: df['revenue'] = df['revenue'].fillna(0)
                if 'runtime' in df.columns: df['runtime'] = df['runtime'].fillna(df['runtime'].median())
                for col in ['homepage', 'tagline', 'overview']:
                    if col in df.columns: df[col] = df[col].fillna('')

                if 'release_date' in df.columns:
                    df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
                    df['release_year']  = df['release_date'].dt.year
                    df['release_month'] = df['release_date'].dt.month
                    df['release_date']  = df['release_date'].dt.strftime('%Y-%m-%d')

                if 'revenue' in df.columns and 'budget' in df.columns:
                    df['profit'] = df['revenue'] - df['budget']

                if 'profit' in df.columns and 'budget' in df.columns:
                    df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)

                if 'budget'  in df.columns: df = df[df['budget']  >= 0]
                if 'revenue' in df.columns: df = df[df['revenue'] >= 0]

                df = df.drop_duplicates()
                df = df.drop(columns=['id', 'imdb_id'], errors='ignore')

            post_metrics = {
                "total_rows": df.shape[0],
                "total_cols": df.shape[1],
                "duplicate_count": df.duplicated().sum(),
                "missing_count": df.isna().sum().sum(),
                "missing_per_col": df.isna().sum()
            }

            st.session_state["df"]                = df
            st.session_state["pre_metrics"]       = pre_metrics
            st.session_state["post_metrics"]      = post_metrics
            st.session_state["uploaded_filename"] = uploaded_file.name

        except Exception as e:
            st.error(f"Error loading and preprocessing file: {e}")

if uploaded_file is None:
    for key in ["df", "pre_metrics", "post_metrics", "uploaded_filename"]:
        if key in st.session_state:
            del st.session_state[key]

# ── Main Content ──────────────────────────────────────────────
if "df" not in st.session_state or uploaded_file is None:
    st.markdown("""
    <style>
        .block-container { 
            padding: 0 !important; 
            max-width: 100vw !important; 
            margin: 0 !important; 
            margin-top: -10px !important; 
        }
        .stApp { 
            background: #0a0a0f !important; 
        }
        iframe {
            height: 100vh !important;
            width: 100vw !important;
            border: none;
        }
        header[data-testid="stHeader"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    components.html(hero_html, height=1000, scrolling=False)
else:
    df = st.session_state["df"]

    if selected_section == "Dashboard":
        st.success("✅ Database uploaded and preprocessed successfully!")
        st.markdown("### Select an Analysis Module")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("📊 Data Overview & EDA", use_container_width=True, on_click=set_nav_section, args=("Data Overview & EDA",))
            st.button("🧬 Association Rules", use_container_width=True, on_click=set_nav_section, args=("Association Rules (Apriori + FP-Growth)",))
            st.button("📈 Linear Regression", use_container_width=True, on_click=set_nav_section, args=("Linear Regression",))
        with c2:
            st.button("🎲 OLAP Analysis", use_container_width=True, on_click=set_nav_section, args=("OLAP Analysis",))
            st.button("📉 PCA Dimensionality", use_container_width=True, on_click=set_nav_section, args=("PCA",))
            st.button("💻 Custom Code", use_container_width=True, on_click=set_nav_section, args=("Custom Code Execution",))
        with c3:
            st.button("📐 Data Transformation", use_container_width=True, on_click=set_nav_section, args=("Data Transformation",))
            st.button("🤖 Classification Models", use_container_width=True, on_click=set_nav_section, args=("Classification (Decision Tree, Naive Bayes, KNN)",))
        
        st.markdown("---")
        st.markdown("##### Quick Dataset Stats")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Rows", f"{df.shape[0]:,}")
        sc2.metric("Columns", f"{df.shape[1]:,}")
        if 'release_year' in df.columns:
            sc3.metric("Years Covered", f"{int(df['release_year'].min())} - {int(df['release_year'].max())}")
        if 'revenue' in df.columns:
            sc4.metric("Avg Revenue", f"${df['revenue'].mean()/1e6:.1f}M")

    else:
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("← Back", key="back_btn"):
                go_back()
                st.rerun()
        with col2:
            st.header(st.session_state.nav_section)
        st.markdown("---")

    # ── DATA OVERVIEW & EDA ───────────────────────────────────
    if st.session_state.nav_section == "Data Overview & EDA":

        pre_metrics  = st.session_state.get("pre_metrics", {})
        post_metrics = st.session_state.get("post_metrics", {})

        if pre_metrics and post_metrics:
            st.subheader("Data Summary (Before vs After Cleaning)")

            st.markdown("**Before Cleaning:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows",      pre_metrics.get("total_rows", 0))
            col2.metric("Total Columns",   pre_metrics.get("total_cols", 0))
            col3.metric("Duplicate Count", pre_metrics.get("duplicate_count", 0))
            col4.metric("Missing Values",  pre_metrics.get("missing_count", 0))

            st.markdown("---")

            st.markdown("**After Cleaning:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows",      post_metrics.get("total_rows", 0),
                        delta=post_metrics.get("total_rows", 0) - pre_metrics.get("total_rows", 0))
            col2.metric("Total Columns",   post_metrics.get("total_cols", 0),
                        delta=post_metrics.get("total_cols", 0) - pre_metrics.get("total_cols", 0))
            col3.metric("Duplicate Count", post_metrics.get("duplicate_count", 0),
                        delta=post_metrics.get("duplicate_count", 0) - pre_metrics.get("duplicate_count", 0),
                        delta_color="inverse")
            col4.metric("Missing Values",  post_metrics.get("missing_count", 0),
                        delta=post_metrics.get("missing_count", 0) - pre_metrics.get("missing_count", 0),
                        delta_color="inverse")

            st.markdown("---")
            st.markdown("##### Missing Values Per Column")
            fig_col1, fig_col2 = st.columns(2)

            with fig_col1:
                st.markdown("**Before Cleaning**")
                missing_pre_s = pre_metrics.get("missing_per_col")
                if missing_pre_s is not None:
                    mp = missing_pre_s.reset_index()
                    mp.columns = ["Column", "Missing Values"]
                    fig_pre = px.bar(mp, x="Column", y="Missing Values",
                                     color_discrete_sequence=["#ef4444"], height=350)
                    fig_pre.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#f8fafc", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pre, width='stretch')

            with fig_col2:
                st.markdown("**After Cleaning**")
                missing_post_s = post_metrics.get("missing_per_col")
                if missing_post_s is not None:
                    mp2 = missing_post_s.reset_index()
                    mp2.columns = ["Column", "Missing Values"]
                    fig_post = px.bar(mp2, x="Column", y="Missing Values",
                                      color_discrete_sequence=["#22c55e"], height=350)
                    fig_post.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           font_color="#f8fafc", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_post, width='stretch')

            st.success(f"Preprocessing complete! Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

        st.subheader("Dataset Preview")
        st.dataframe(df, width='stretch')

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width='stretch')

    # ── OLAP ─────────────────────────────────────────────────
    elif st.session_state.nav_section == "OLAP Analysis":
        st.subheader("OLAP Operations")

        if 'release_year' in df.columns:
            valid_years_df = df.dropna(subset=['release_year'])
            min_year = int(valid_years_df['release_year'].min())
            max_year = int(valid_years_df['release_year'].max())
        else:
            min_year, max_year = 1900, 2025

        with st.expander("🔪 SLICE: Filter by a single dimension (e.g., Year)", expanded=True):
            slice_year = st.slider("Select Release Year", min_value=min_year, max_value=max_year,
                                   value=max_year, key="slice_year")
            sliced_df = df[df['release_year'] == slice_year]
            st.write(f"**Movies released in {slice_year}:** {len(sliced_df)}")
            if not sliced_df.empty:
                cols_to_show = [c for c in ['title', 'original_title', 'revenue', 'vote_average', 'release_date']
                                if c in sliced_df.columns]
                st.dataframe(sliced_df[cols_to_show], width='stretch')
            else:
                st.info(f"No movies found for {slice_year}")

        with st.expander("🎲 DICE: Filter by multiple dimensions"):
            col1, col2, col3 = st.columns(3)
            with col1:
                dice_years = st.slider("Select Year Range", min_value=min_year, max_value=max_year,
                                       value=(min_year, max_year), key="dice_years")
            with col2:
                min_rev = float(df['revenue'].min()) if 'revenue' in df.columns else 0.0
                max_rev = float(df['revenue'].max()) if 'revenue' in df.columns else 1000000.0
                dice_min_revenue = st.number_input("Minimum Revenue", min_value=min_rev, max_value=max_rev,
                                                   value=min_rev, step=100000.0)
            with col3:
                dice_min_vote = st.number_input("Minimum Vote Average", min_value=0.0, max_value=10.0,
                                                value=0.0, step=0.5)
            diced_df = df[
                (df['release_year'] >= dice_years[0]) &
                (df['release_year'] <= dice_years[1]) &
                (df['revenue'] >= dice_min_revenue) &
                (df['vote_average'] >= dice_min_vote)
            ]
            st.write(f"**Filtered Results:** {len(diced_df)} movies match these criteria.")
            if not diced_df.empty:
                cols_to_show = [c for c in ['title', 'original_title', 'release_year', 'revenue', 'vote_average']
                                if c in diced_df.columns]
                st.dataframe(diced_df[cols_to_show], width='stretch')

        with st.expander("📈 ROLL-UP: Aggregate data across dimensions"):
            st.markdown("**(a) Total Revenue by Year**")
            rollup_yearly = df.groupby('release_year')['revenue'].sum().reset_index()
            fig_rollup_line = px.line(rollup_yearly, x='release_year', y='revenue', markers=True,
                                      title="Total Revenue by Year",
                                      color_discrete_sequence=["#eab308"])
            fig_rollup_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#f8fafc")
            st.plotly_chart(fig_rollup_line, width='stretch')

            st.markdown("**(b) Revenue Heatmap by Year & Month**")
            if 'release_month' in df.columns:
                heatmap_data  = df.groupby(['release_year', 'release_month'])['revenue'].sum().reset_index()
                heatmap_pivot = heatmap_data.pivot(index="release_year", columns="release_month",
                                                   values="revenue").fillna(0)
                fig_heatmap = px.imshow(heatmap_pivot,
                                        labels=dict(x="Month", y="Year", color="Revenue"),
                                        x=heatmap_pivot.columns, y=heatmap_pivot.index,
                                        aspect="auto", color_continuous_scale="Viridis",
                                        title="Revenue Heatmap by Year & Month")
                fig_heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#f8fafc")
                st.plotly_chart(fig_heatmap, width='stretch')
            else:
                st.warning("Month dimensions missing from dataset.")

        with st.expander("🔍 DRILL-DOWN: Break down aggregated data"):
            drill_year = st.selectbox("Select Year to Drill-down into Months",
                                      options=sorted(df['release_year'].dropna().unique(), reverse=True))
            if 'release_month' in df.columns:
                drill_df    = df[df['release_year'] == drill_year]
                monthly_rev = drill_df.groupby('release_month')['revenue'].sum().reset_index()
                all_months  = pd.DataFrame({'release_month': range(1, 13)})
                monthly_rev = pd.merge(all_months, monthly_rev, on='release_month', how='left').fillna({'revenue': 0})
                fig_drill = px.bar(monthly_rev, x='release_month', y='revenue',
                                   title=f"Monthly Revenue for {int(drill_year)}",
                                   labels={"release_month": "Month", "revenue": "Revenue"},
                                   color_discrete_sequence=["#06b6d4"])
                fig_drill.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#f8fafc")
                st.plotly_chart(fig_drill, width='stretch')
            else:
                st.warning("Month dimensions missing from dataset.")

    # ── DATA TRANSFORMATION ───────────────────────────────────
    elif st.session_state.nav_section == "Data Transformation":
        st.subheader("Data Transformation & Statistical Testing")

        with st.expander("📏 1. EQUAL WIDTH BINNING", expanded=True):
            st.markdown("Bin `revenue` into 5 equal-width intervals.")
            if 'revenue' in df.columns:
                df['revenue_bin_equal_width'] = pd.cut(df['revenue'], bins=5)
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['revenue', 'revenue_bin_equal_width']].head(10), width='stretch')
                with col2:
                    bin_counts = df['revenue_bin_equal_width'].value_counts().sort_index().reset_index()
                    bin_counts.columns = ['Bin', 'Count']
                    bin_counts['Bin'] = bin_counts['Bin'].astype(str)
                    fig_ew = px.bar(bin_counts, x='Bin', y='Count', title="Equal Width Bin Counts",
                                    color_discrete_sequence=["#3b82f6"])
                    fig_ew.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         font_color="#f8fafc")
                    st.plotly_chart(fig_ew, width='stretch')
            else:
                st.warning("Revenue column missing.")

        with st.expander("⚖️ 2. EQUAL FREQUENCY BINNING"):
            st.markdown("Bin `revenue` into 5 quantiles (equal number of records).")
            if 'revenue' in df.columns:
                try:
                    df['revenue_bin_equal_freq'] = pd.qcut(df['revenue'], q=5, duplicates='drop')
                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        st.dataframe(df[['revenue', 'revenue_bin_equal_freq']].head(10), width='stretch')
                    with col2:
                        freq_counts = df['revenue_bin_equal_freq'].value_counts().sort_index().reset_index()
                        freq_counts.columns = ['Bin', 'Count']
                        freq_counts['Bin'] = freq_counts['Bin'].astype(str)
                        fig_ef = px.bar(freq_counts, x='Bin', y='Count', title="Equal Frequency Bin Counts",
                                        color_discrete_sequence=["#10b981"])
                        fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                             font_color="#f8fafc")
                        st.plotly_chart(fig_ef, width='stretch')
                except Exception as e:
                    st.error(f"Error producing quantiles: {e}")
            else:
                st.warning("Revenue column missing.")

        with st.expander("🎨 3. CUSTOM BINNING"):
            st.markdown("Bin `revenue` into custom tiers: Low, Medium, High, Blockbuster.")
            if 'revenue' in df.columns:
                max_rev   = df['revenue'].max()
                safe_max  = max(500_000_000, max_rev) + 1
                bins      = [-1, 50_000_000, 200_000_000, 500_000_000, safe_max]
                labels    = ['Low', 'Medium', 'High', 'Blockbuster']
                df['revenue_custom_tier'] = pd.cut(df['revenue'], bins=bins, labels=labels)
                tier_counts = df['revenue_custom_tier'].value_counts().reset_index()
                tier_counts.columns = ['Tier', 'Count']
                fig_pie = px.pie(tier_counts, names='Tier', values='Count',
                                 title="Custom Revenue Tiers Distribution",
                                 color_discrete_sequence=px.colors.sequential.Plasma)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#f8fafc")
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.warning("Revenue column missing.")

        with st.expander("📉 4. MIN-MAX NORMALIZATION"):
            st.markdown("Scale `budget` to a range of [0, 1].")
            if 'budget' in df.columns:
                scaler_minmax = MinMaxScaler()
                df['budget_minmax'] = scaler_minmax.fit_transform(df[['budget']])
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['budget', 'budget_minmax']].head(10), width='stretch')
                with col2:
                    fig_minmax = px.histogram(df, x='budget_minmax', nbins=30,
                                              title="Distribution of Budget (Min-Max)",
                                              color_discrete_sequence=["#8b5cf6"])
                    fig_minmax.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                             font_color="#f8fafc")
                    st.plotly_chart(fig_minmax, width='stretch')
            else:
                st.warning("Budget column missing.")

        with st.expander("📊 5. Z-SCORE NORMALIZATION"):
            st.markdown("Standardize `budget` (mean=0, std=1).")
            if 'budget' in df.columns:
                scaler_z = StandardScaler()
                df['budget_zscore'] = scaler_z.fit_transform(df[['budget']])
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['budget', 'budget_zscore']].head(10), width='stretch')
                with col2:
                    fig_z = px.histogram(df, x='budget_zscore', nbins=30,
                                         title="Distribution of Budget (Z-Score)",
                                         color_discrete_sequence=["#f43f5e"])
                    fig_z.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#f8fafc")
                    st.plotly_chart(fig_z, width='stretch')
            else:
                st.warning("Budget column missing.")

        st.markdown("---")
        st.subheader("🧪 Chi-Square Test of Independence")
        st.markdown("Testing independence between **Revenue Category** and **Vote Category**.")
        if 'revenue' in df.columns and 'vote_average' in df.columns:
            df['chi2_rev_cat']  = pd.qcut(df['revenue'],      q=3, labels=['Low', 'Medium', 'High'],
                                          duplicates='drop')
            df['chi2_vote_cat'] = pd.qcut(df['vote_average'], q=3,
                                          labels=['Low Rating', 'Medium Rating', 'High Rating'],
                                          duplicates='drop')
            contingency_table = pd.crosstab(df['chi2_rev_cat'], df['chi2_vote_cat'])
            st.markdown("**Contingency Table:**")
            st.dataframe(contingency_table, width='stretch')
            chi2, p_val, dof, _ = chi2_contingency(contingency_table)
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-Square Statistic", f"{chi2:.2f}")
            col2.metric("P-value", f"{p_val:.4e}")
            col3.metric("Degrees of Freedom", dof)
            if p_val < 0.05:
                st.success("**Significant Result (p < 0.05):** We reject the null hypothesis. "
                           "There is a statistically significant association between Revenue Category and Vote Category.")
            else:
                st.warning("**Not Significant (p >= 0.05):** We fail to reject the null hypothesis.")
        else:
            st.warning("Revenue or Vote Average columns missing.")

    # ── ASSOCIATION RULES ─────────────────────────────────────
    elif st.session_state.nav_section == "Association Rules (Apriori + FP-Growth)":
        st.subheader("Associations & Frequent Itemsets")

        def extract_names(json_str):
            try:
                lst = ast.literal_eval(json_str)
                return [item['name'] for item in lst] if isinstance(lst, list) else []
            except (ValueError, SyntaxError):
                return []

        # Pre-extract genre lists (shared between sections A & C)
        genre_lists = []
        if 'genres' in df.columns:
            genre_lists = [g for g in df['genres'].dropna().apply(extract_names).tolist() if len(g) > 0]

        with st.expander("📚 SECTION A: Apriori on Genres", expanded=True):
            if genre_lists:
                st.markdown("**Frequent Genres using Apriori Algorithm**")
                col1, col2 = st.columns(2)
                with col1: min_support_g    = st.slider("Minimum Support (Genres)",     0.01, 0.50, 0.05, 0.01, key='msg')
                with col2: min_confidence_g = st.slider("Minimum Confidence (Genres)", 0.10, 1.00, 0.60, 0.05, key='mcg')

                te = TransactionEncoder()
                te_ary = te.fit(genre_lists).transform(genre_lists)
                df_genres = pd.DataFrame(te_ary, columns=te.columns_)

                with st.spinner("Running Apriori algorithm on genres..."):
                    freq_items_apriori = apriori(df_genres, min_support=min_support_g, use_colnames=True)

                if not freq_items_apriori.empty:
                    freq_items_apriori['itemsets_str'] = freq_items_apriori['itemsets'].apply(
                        lambda x: ', '.join(list(x)))
                    top10_genres = freq_items_apriori.sort_values('support', ascending=False).head(10)
                    fig_g = px.bar(top10_genres, x='itemsets_str', y='support',
                                   title="Top 10 Genres (Support)",
                                   color_discrete_sequence=["#8b5cf6"])
                    fig_g.update_layout(xaxis_title="Genre", yaxis_title="Support",
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#f8fafc")
                    st.plotly_chart(fig_g, width='stretch')

                    rules_g = association_rules(freq_items_apriori, metric="confidence",
                                               min_threshold=min_confidence_g)
                    if not rules_g.empty:
                        rules_g['antecedents'] = rules_g['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_g['consequents'] = rules_g['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_g = rules_g.sort_values('lift', ascending=False)[
                            ['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        st.markdown(f"**Discovered {len(rules_g)} rules (Sorted by Lift Descending)**")
                        st.dataframe(rules_g.style.background_gradient(subset=['lift'], cmap='Blues'),
                                     width='stretch')
                    else:
                        st.info("No explicit rules found matching the confidence threshold.")
                else:
                    st.warning("No frequent itemsets found. Try lowering the support threshold.")
            else:
                st.warning("Genres column missing or empty.")

        with st.expander("🔑 SECTION B: Apriori on Keywords"):
            if 'keywords' in df.columns:
                st.markdown("**Frequent Keywords using Apriori Algorithm**")
                col1, col2 = st.columns(2)
                with col1: min_support_k    = st.slider("Minimum Support (Keywords)",     0.01, 0.50, 0.03, 0.01, key='msk')
                with col2: min_confidence_k = st.slider("Minimum Confidence (Keywords)", 0.10, 1.00, 0.60, 0.05, key='mck')

                keyword_lists = [k for k in df['keywords'].dropna().apply(extract_names).tolist() if len(k) > 0]
                if keyword_lists:
                    te_k = TransactionEncoder()
                    te_ary_k = te_k.fit(keyword_lists).transform(keyword_lists)
                    df_keywords = pd.DataFrame(te_ary_k, columns=te_k.columns_)

                    with st.spinner("Running Apriori algorithm on keywords..."):
                        freq_items_k = apriori(df_keywords, min_support=min_support_k, use_colnames=True)

                    if not freq_items_k.empty:
                        freq_items_k['itemsets_str'] = freq_items_k['itemsets'].apply(
                            lambda x: ', '.join(list(x)))
                        top10_keys = freq_items_k.sort_values('support', ascending=False).head(10)
                        fig_k = px.bar(top10_keys, x='itemsets_str', y='support',
                                       title="Top 10 Keywords (Support)",
                                       color_discrete_sequence=["#10b981"])
                        fig_k.update_layout(xaxis_title="Keyword", yaxis_title="Support",
                                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#f8fafc")
                        st.plotly_chart(fig_k, width='stretch')

                        rules_k = association_rules(freq_items_k, metric="confidence",
                                                   min_threshold=min_confidence_k)
                        if not rules_k.empty:
                            rules_k['antecedents'] = rules_k['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_k['consequents'] = rules_k['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules_k = rules_k.sort_values('lift', ascending=False)[
                                ['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                            st.markdown(f"**Discovered {len(rules_k)} rules (Sorted by Lift Descending)**")
                            st.dataframe(rules_k.style.background_gradient(subset=['lift'], cmap='Greens'),
                                         width='stretch')
                        else:
                            st.info("No explicit rules found matching the confidence threshold.")
                    else:
                        st.warning("No frequent itemsets found. Try lowering the support threshold.")
                else:
                    st.warning("Could not extract any keywords.")
            else:
                st.warning("Keywords column missing.")

        with st.expander("⚡ SECTION C: FP-Growth on Genres"):
            if genre_lists:
                st.markdown("**Frequent Genres using FP-Growth Algorithm**")
                st.info("💡 **Comparison Note:** FP-Growth is faster than Apriori for large datasets "
                        "as it avoids candidate generation by compressing the data into an FP-tree.")
                fp_min_support = 0.05
                fp_min_conf    = 0.6

                te_fp   = TransactionEncoder()
                te_ary_fp = te_fp.fit(genre_lists).transform(genre_lists)
                df_genres_fp = pd.DataFrame(te_ary_fp, columns=te_fp.columns_)

                with st.spinner("Running FP-Growth algorithm on genres..."):
                    freq_items_fp = fpgrowth(df_genres_fp, min_support=fp_min_support, use_colnames=True)

                if not freq_items_fp.empty:
                    rules_fp = association_rules(freq_items_fp, metric="confidence", min_threshold=fp_min_conf)
                    if not rules_fp.empty:
                        rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_fp = rules_fp.sort_values('lift', ascending=False)[
                            ['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        st.markdown(f"**Discovered {len(rules_fp)} rules "
                                    f"(Support >= {fp_min_support}, Confidence >= {fp_min_conf})**")
                        st.dataframe(rules_fp.style.background_gradient(subset=['lift'], cmap='Oranges'),
                                     width='stretch')
                    else:
                        st.info("No explicit rules found matching the confidence threshold.")
                else:
                    st.warning("No frequent itemsets found with FP-Growth.")
            else:
                st.warning("Genres column missing or empty.")

    # ── PCA ───────────────────────────────────────────────────
    elif st.session_state.nav_section == "PCA":
        st.subheader("Principal Component Analysis (PCA)")
        st.markdown("Reducing dimensionality for: `budget`, `popularity`, `revenue`, `runtime`, "
                    "`vote_average`, `vote_count`")

        features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
        missing_features = [f for f in features if f not in df.columns]

        if not missing_features:
            X = df[features].copy()
            for col in features:
                X[col] = X[col].fillna(X[col].median())

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            with st.spinner("Running Principal Component Analysis..."):
                pca_full    = PCA()
                pca_full.fit(X_scaled)
                explained_var  = pca_full.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            n_components_85 = int(np.argmax(cumulative_var >= 0.85) + 1)

            st.markdown("### 1. Variance Analysis")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_var = go.Figure()
                fig_var.add_trace(go.Scatter(x=list(range(1, len(explained_var)+1)), y=explained_var,
                                             mode='lines+markers', name='Individual Explained Variance',
                                             line=dict(color='#3b82f6')))
                fig_var.add_trace(go.Scatter(x=list(range(1, len(cumulative_var)+1)), y=cumulative_var,
                                             mode='lines+markers', name='Cumulative Explained Variance',
                                             line=dict(color='#10b981')))
                fig_var.add_hline(y=0.85, line_dash="dash", line_color="#ef4444",
                                  annotation_text="85% Threshold")
                fig_var.update_layout(title="PCA Explained Variance",
                                      xaxis_title="Principal Component",
                                      yaxis_title="Explained Variance Ratio",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#f8fafc",
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                  xanchor="right", x=1))
                st.plotly_chart(fig_var, width='stretch')
            with col2:
                st.metric("Components needed for 85% variance", n_components_85)
                var_df = pd.DataFrame({
                    'Component': [f"PC{i+1}" for i in range(len(explained_var))],
                    'Individual Variance': np.round(explained_var, 4),
                    'Cumulative Variance': np.round(cumulative_var, 4)
                })
                st.dataframe(var_df, width='stretch')

            st.markdown("---")
            st.markdown("### 2. 2D Projection")
            pca_2d   = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            pca_df   = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
            pca_df.index = df.index

            try:
                rating_cat = pd.qcut(df['vote_average'], q=3,
                                     labels=['Low Rating', 'Medium Rating', 'High Rating'],
                                     duplicates='drop')
            except ValueError:
                rating_cat = pd.cut(df['vote_average'], bins=[-1, 5, 7, 10],
                                    labels=['Low Rating', 'Medium Rating', 'High Rating'])

            pca_df['Rating Category'] = rating_cat
            pca_df['Title'] = (df['original_title'] if 'original_title' in df.columns
                               else df['title'] if 'title' in df.columns
                               else "Movie " + pca_df.index.astype(str))

            col3, col4 = st.columns([2, 1])
            with col3:
                fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color='Rating Category',
                                         hover_name='Title',
                                         title="2D PCA Scatter Plot Colored by Vote Rating",
                                         color_discrete_sequence=px.colors.qualitative.Vivid)
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#f8fafc")
                st.plotly_chart(fig_scatter, width='stretch')
            with col4:
                st.markdown("**First 10 rows of mapped PC dimensions**")
                st.dataframe(pca_df[['PC1', 'PC2', 'Rating Category']].head(10), width='stretch')

            total_explained_2d = np.sum(pca_2d.explained_variance_ratio_) * 100
            st.info(f"💡 Dataset reduced from 6 dimensions to 2 dimensions. "
                    f"PC1 + PC2 explain {total_explained_2d:.2f}% of variance.")
        else:
            st.warning(f"Cannot perform PCA. Missing features: {', '.join(missing_features)}")

    # ── CLASSIFICATION ────────────────────────────────────────
    elif st.session_state.nav_section == "Classification (Decision Tree, Naive Bayes, KNN)":
        st.subheader("Classification Models")
        st.markdown("Predicting whether a movie is a **Hit** (revenue >= median revenue) based on features.")

        features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
        if all(f in df.columns for f in features) and 'revenue' in df.columns:
            ml_df = df.dropna(subset=features + ['revenue']).copy()
            for col in features:
                ml_df[col] = ml_df[col].fillna(ml_df[col].median())

            ml_df['hit'] = (ml_df['revenue'] >= ml_df['revenue'].median()).astype(int)
            X = ml_df[features]
            y = ml_df['hit']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.write(f"**Dataset:** {len(X_train)} training samples, {len(X_test)} testing samples")

            tab1, tab2, tab3 = st.tabs(["Decision Tree", "Naive Bayes", "KNN"])

            with tab1:
                st.markdown("### Decision Tree Classifier")
                dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
                dt.fit(X_train, y_train)
                y_pred_dt = dt.predict(X_test)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Accuracy", f"{acc_dt * 100:.2f}%")
                    st.markdown("**Classification Report:**")
                    st.text(classification_report(y_test, y_pred_dt))
                with col2:
                    cm_dt = confusion_matrix(y_test, y_pred_dt)
                    fig_cm_dt = px.imshow(cm_dt, text_auto=True, color_continuous_scale='Blues',
                                          labels=dict(x="Predicted", y="Actual", color="Count"),
                                          x=['Not Hit (0)', 'Hit (1)'], y=['Not Hit (0)', 'Hit (1)'],
                                          title="Confusion Matrix")
                    fig_cm_dt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#f8fafc")
                    st.plotly_chart(fig_cm_dt, width='stretch')
                st.markdown("**Tree Visualization (Text)**")
                st.code(export_text(dt, feature_names=features), language="text")

            with tab2:
                st.markdown("### Gaussian Naive Bayes")
                st.info("No scaling required for Gaussian Naive Bayes.")
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                y_pred_nb = nb.predict(X_test)
                acc_nb = accuracy_score(y_test, y_pred_nb)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Accuracy", f"{acc_nb * 100:.2f}%")
                    st.markdown("**Classification Report:**")
                    st.text(classification_report(y_test, y_pred_nb))
                with col2:
                    cm_nb = confusion_matrix(y_test, y_pred_nb)
                    fig_cm_nb = px.imshow(cm_nb, text_auto=True, color_continuous_scale='Greens',
                                          labels=dict(x="Predicted", y="Actual", color="Count"),
                                          x=['Not Hit (0)', 'Hit (1)'], y=['Not Hit (0)', 'Hit (1)'],
                                          title="Confusion Matrix")
                    fig_cm_nb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#f8fafc")
                    st.plotly_chart(fig_cm_nb, width='stretch')

            with tab3:
                st.markdown("### K-Nearest Neighbors (KNN)")
                scaler_knn     = StandardScaler()
                X_train_scaled = scaler_knn.fit_transform(X_train)
                X_test_scaled  = scaler_knn.transform(X_test)

                k_values   = range(1, 21)
                accuracies = []
                for k in k_values:
                    knn_temp = KNeighborsClassifier(n_neighbors=k)
                    knn_temp.fit(X_train_scaled, y_train)
                    accuracies.append(accuracy_score(y_test, knn_temp.predict(X_test_scaled)))

                best_k   = list(k_values)[int(np.argmax(accuracies))]
                best_acc = max(accuracies)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(f"Best K (1–20)", f"{best_k}", f"Accuracy: {best_acc * 100:.2f}%")
                    st.markdown("---")
                    st.markdown("**Test a Custom K**")
                    custom_k = st.number_input("Enter K value:", min_value=1, max_value=100,
                                               value=best_k, step=1)
                    knn_custom = KNeighborsClassifier(n_neighbors=custom_k)
                    knn_custom.fit(X_train_scaled, y_train)
                    y_pred_custom = knn_custom.predict(X_test_scaled)
                    acc_custom = accuracy_score(y_test, y_pred_custom)
                    st.success(f"Accuracy with K={custom_k}: **{acc_custom * 100:.2f}%**")
                    st.text(classification_report(y_test, y_pred_custom))
                with col2:
                    fig_k = px.line(x=list(k_values), y=accuracies, markers=True,
                                    title="Accuracy vs K Value",
                                    labels={'x': 'Number of Neighbors (K)', 'y': 'Accuracy'},
                                    color_discrete_sequence=["#eab308"])
                    fig_k.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#f8fafc", xaxis=dict(tickmode='linear', dtick=1))
                    fig_k.add_scatter(x=[best_k], y=[best_acc], mode='markers',
                                      marker=dict(color='red', size=12, symbol='star'), name='Best K')
                    st.plotly_chart(fig_k, width='stretch')

                    cm_knn = confusion_matrix(y_test, y_pred_custom)
                    fig_cm_knn = px.imshow(cm_knn, text_auto=True, color_continuous_scale='Oranges',
                                           labels=dict(x="Predicted", y="Actual", color="Count"),
                                           x=['Not Hit (0)', 'Hit (1)'], y=['Not Hit (0)', 'Hit (1)'],
                                           title=f"Confusion Matrix (K={custom_k})")
                    fig_cm_knn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                             font_color="#f8fafc")
                    st.plotly_chart(fig_cm_knn, width='stretch')
        else:
            st.warning("Missing required features for Classification. Check dataset completion.")

    # ── LINEAR REGRESSION ─────────────────────────────────────
    elif st.session_state.nav_section == "Linear Regression":
        st.subheader("Linear Regression Analysis")
        st.markdown("Predicting numerical **Revenue** based on standardized features.")

        features_reg = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
        if all(f in df.columns for f in features_reg) and 'revenue' in df.columns:
            with st.spinner("Training Linear Regression model..."):
                reg_df = df.dropna(subset=features_reg + ['revenue']).copy()
                for col in features_reg:
                    reg_df[col] = reg_df[col].fillna(reg_df[col].median())

                X_reg = reg_df[features_reg]
                y_reg = reg_df['revenue']
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                    X_reg, y_reg, test_size=0.2, random_state=42)

                scaler_reg       = StandardScaler()
                X_train_reg_sc   = scaler_reg.fit_transform(X_train_reg)
                X_test_reg_sc    = scaler_reg.transform(X_test_reg)
                lr_model         = LinearRegression()
                lr_model.fit(X_train_reg_sc, y_train_reg)
                y_pred_reg = lr_model.predict(X_test_reg_sc)
                mse = mean_squared_error(y_test_reg, y_pred_reg)
                r2  = r2_score(y_test_reg, y_pred_reg)

            st.write(f"**Dataset:** {len(X_train_reg)} training, {len(X_test_reg)} testing samples")
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:,.0f}")
            col2.metric("R² Score", f"{r2:.4f}", help="Close to 1 is better")

            st.markdown("---")
            col3, col4 = st.columns([1, 1.2])
            with col3:
                st.markdown("**Feature Coefficients**")
                coef_df = pd.DataFrame({'Feature': features_reg, 'Coefficient': lr_model.coef_})
                coef_df = coef_df.sort_values(by='Coefficient', ascending=True)
                coef_df['Color'] = np.where(coef_df['Coefficient'] > 0, "#3b82f6", "#ef4444")
                fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                  color='Color', color_discrete_map="identity",
                                  title="Impact of Features on Revenue")
                fig_coef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="#f8fafc")
                st.plotly_chart(fig_coef, width='stretch')
            with col4:
                st.markdown("**Actual vs Predicted Revenue**")
                avsp_df = pd.DataFrame({'Actual Revenue': y_test_reg, 'Predicted Revenue': y_pred_reg})
                fig_scatter = px.scatter(avsp_df, x='Actual Revenue', y='Predicted Revenue',
                                         opacity=0.6, color_discrete_sequence=["#8b5cf6"],
                                         title="Actual vs Predicted Revenue")
                max_val = max(avsp_df['Actual Revenue'].max(), avsp_df['Predicted Revenue'].max())
                min_val = min(avsp_df['Actual Revenue'].min(), avsp_df['Predicted Revenue'].min())
                fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                                 mode='lines', name='Perfect Prediction',
                                                 line=dict(color='#ef4444', dash='dash')))
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#f8fafc", showlegend=False)
                st.plotly_chart(fig_scatter, width='stretch')
        else:
            st.warning("Missing required features for Linear Regression.")

    # ── CUSTOM CODE ───────────────────────────────────────────
    elif st.session_state.nav_section == "Custom Code Execution":
        st.subheader("💻 Custom Code Execution")
        st.warning("⚠️ **Security Warning:** This section allows executing arbitrary Python code. "
                   "Added strictly for college project demonstration purposes.")

        default_code = '''# Example 1: Print a simple message
print("Hello, Faculty!")

# Example 2: Access the current dataframe
print(f"Current Dataset Shape: {df.shape}")

# Example 3: Draw a custom plot
fig = px.scatter(df.head(100), x='budget', y='revenue', title="Custom Plot")
st.plotly_chart(fig)
'''
        st.markdown("Write Python below. You have access to all imported libraries "
                    "(`pd`, `px`, `st`, etc.) and the dataset variable `df`.")
        user_code = st.text_area("Enter Python Code:", value=default_code, height=300)

        if st.button("▶️ Run Code", type="primary"):
            st.markdown("### Execution Result")
            old_stdout = sys.stdout
            try:
                redirected_output = sys.stdout = StringIO()
                exec_globals = globals().copy()
                exec_locals  = locals().copy()
                result = None

                tree = ast.parse(user_code)
                if tree.body:
                    last_stmt = tree.body[-1]
                    if isinstance(last_stmt, ast.Expr):
                        exec_body = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(exec_body, '<string>', 'exec'), exec_globals, exec_locals)
                        expr   = ast.Expression(last_stmt.value)
                        result = eval(compile(expr, '<string>', 'eval'), exec_globals, exec_locals)
                    else:
                        exec(user_code, exec_globals, exec_locals)

                sys.stdout = old_stdout
                output_str = redirected_output.getvalue()
                
                if not tree.body:
                    st.warning("⚠️ No executable code found. Did you forget to uncomment the code block?")
                else:
                    if output_str.strip():
                        st.code(output_str, language="text")
                    if result is not None and type(result).__name__ != 'DeltaGenerator':
                        st.write(result)
                    elif not output_str.strip():
                        st.success("Code executed successfully with no textual output.")
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"Error executing code:\n{str(e)}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.9em;'>"
    "<strong>&copy; 2026 Built by Jesu Mariya Joe D (2361018) & Keya (2361023)</strong><br>"
    "CIA 3.1 Mini Project | TMDB 5000 Dataset | Techniques: EDA, OLAP, Binning, Normalization, "
    "Chi-Square, Apriori, FP-Growth, PCA, Decision Tree, Naive Bayes, KNN, Linear Regression"
    "</p>",
    unsafe_allow_html=True
)