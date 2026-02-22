import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from scipy.stats import chi2_contingency
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Set page config
st.set_page_config(
    page_title="Movies & OTT Data Mining Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Tailwind-style dark theme (slate/gray palette)
st.markdown("""
<style>
    /* Backgrounds */
    .stApp {
        background-color: #0f172a; /* Tailwind slate-900 */
        color: #f8fafc; /* Tailwind slate-50 */
    }
    /* Adjust Streamlit specific containers for a cohesive dark theme */
    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Tailwind slate-800 */
    }
    [data-testid="stHeader"] {
        background-color: #0f172a;
    }
    /* Info box styling */
    .stAlert {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movies & OTT Data Mining Dashboard")

# Sidebar
with st.sidebar:
    st.header("Data Setup")
    uploaded_file = st.file_uploader("Upload tmdb_5000_movies.csv", type=['csv'])
    
    st.markdown("---")
    
    st.header("Navigation")
    sections = [
        "Data Overview & EDA",
        "OLAP Analysis",
        "Data Transformation",
        "Association Rules (Apriori + FP-Growth)",
        "PCA",
        "Classification (Decision Tree, Naive Bayes, KNN)",
        "Linear Regression"
    ]
    selected_section = st.radio("Go to", sections, label_visibility="collapsed")

# Handle Data Upload & Session State
if uploaded_file is not None:
    if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != uploaded_file.name:
        try:
            # Load the dataframe
            raw_df = pd.read_csv(uploaded_file)
            
            # metrics before cleaning
            pre_metrics = {
                "total_rows": raw_df.shape[0],
                "total_cols": raw_df.shape[1],
                "duplicate_count": raw_df.duplicated().sum(),
                "missing_count": raw_df.isna().sum().sum(),
                "missing_per_col": raw_df.isna().sum()
            }
            
            # Preprocessing Pipeline
            with st.spinner("Running global data cleaning and preprocessing..."):
                df = raw_df.copy()
                
                # 1. Convert to numeric
                numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # 2. Fill missing
            if 'budget' in df.columns: df['budget'] = df['budget'].fillna(0)
            if 'revenue' in df.columns: df['revenue'] = df['revenue'].fillna(0)
            if 'runtime' in df.columns: df['runtime'] = df['runtime'].fillna(df['runtime'].median())
            for col in ['homepage', 'tagline', 'overview']:
                if col in df.columns: df[col] = df[col].fillna('')
                    
            # 3. Convert release_date to datetime, extract year & month
            if 'release_date' in df.columns:
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                df['release_year'] = df['release_date'].dt.year
                df['release_month'] = df['release_date'].dt.month
                
            # 4. Create profit = revenue - budget
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['profit'] = df['revenue'] - df['budget']
                
            # 5. Create roi = profit/budget
            if 'profit' in df.columns and 'budget' in df.columns:
                df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)
                
            # 6. Remove rows where budget<0 or revenue<0
            if 'budget' in df.columns: df = df[df['budget'] >= 0]
            if 'revenue' in df.columns: df = df[df['revenue'] >= 0]
                
            # 7. Remove outliers: keep revenue <= revenue.quantile(0.99)
            if 'revenue' in df.columns and not df.empty:
                q99 = df['revenue'].quantile(0.99)
                df = df[df['revenue'] <= q99]
                
            # 8. Drop duplicates
            df = df.drop_duplicates()
            
            # 9. Drop columns: id, imdb_id
            df = df.drop(columns=['id', 'imdb_id'], errors='ignore')
            
            # metrics after cleaning
            post_metrics = {
                "total_rows": df.shape[0],
                "total_cols": df.shape[1],
                "duplicate_count": df.duplicated().sum(),
                "missing_count": df.isna().sum().sum(),
                "missing_per_col": df.isna().sum()
            }
            
            st.session_state["df"] = df
            st.session_state["pre_metrics"] = pre_metrics
            st.session_state["post_metrics"] = post_metrics
            st.session_state["uploaded_filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading and preprocessing file: {e}")

# Check if file has been removed
if uploaded_file is None:
    for key in ["df", "pre_metrics", "post_metrics", "uploaded_filename"]:
        if key in st.session_state:
            del st.session_state[key]

# Main Layout
if "df" not in st.session_state or uploaded_file is None:
    st.info("Please upload tmdb_5000_movies.csv to begin", icon="ℹ️")
else:
    df = st.session_state["df"]
    
    st.header(selected_section)
    st.markdown("---")
    
    if selected_section == "Data Overview & EDA":
        
        # Metric cards
        pre_metrics = st.session_state.get("pre_metrics", {})
        post_metrics = st.session_state.get("post_metrics", {})
        
        if pre_metrics and post_metrics:
            st.subheader("Data Summary (Before vs After Cleaning)")
            
            # Row 1: Before Cleaning Metrics
            st.markdown("**Before Cleaning:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", pre_metrics.get("total_rows", 0))
            col2.metric("Total Columns", pre_metrics.get("total_cols", 0))
            col3.metric("Duplicate Count", pre_metrics.get("duplicate_count", 0))
            col4.metric("Missing Values", pre_metrics.get("missing_count", 0))
            
            st.markdown("---")
            
            # Row 2: After Cleaning Metrics
            st.markdown("**After Cleaning:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", post_metrics.get("total_rows", 0), 
                        delta=post_metrics.get("total_rows", 0) - pre_metrics.get("total_rows", 0))
            col2.metric("Total Columns", post_metrics.get("total_cols", 0),
                        delta=post_metrics.get("total_cols", 0) - pre_metrics.get("total_cols", 0))
            col3.metric("Duplicate Count", post_metrics.get("duplicate_count", 0),
                        delta=post_metrics.get("duplicate_count", 0) - pre_metrics.get("duplicate_count", 0), delta_color="inverse")
            col4.metric("Missing Values", post_metrics.get("missing_count", 0),
                        delta=post_metrics.get("missing_count", 0) - pre_metrics.get("missing_count", 0), delta_color="inverse")
            
            st.markdown("---")
            
            st.markdown("##### Missing Values Per Column")
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                st.markdown("**Before Cleaning**")
                missing_pre_s = pre_metrics.get("missing_per_col")
                if missing_pre_s is not None:
                    missing_pre_df = missing_pre_s.reset_index()
                    missing_pre_df.columns = ["Column", "Missing Values"]
                    fig_pre = px.bar(missing_pre_df, x="Column", y="Missing Values", 
                                 color_discrete_sequence=["#ef4444"],
                                 height=350)
                    fig_pre.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pre, use_container_width=True)
                    
            with fig_col2:
                st.markdown("**After Cleaning**")
                missing_post_s = post_metrics.get("missing_per_col")
                if missing_post_s is not None:
                    missing_post_df = missing_post_s.reset_index()
                    missing_post_df.columns = ["Column", "Missing Values"]
                    fig_post = px.bar(missing_post_df, x="Column", y="Missing Values", 
                                 color_discrete_sequence=["#22c55e"],
                                 height=350)
                    fig_post.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_post, use_container_width=True)
                
            st.success(f"Preprocessing complete! Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
    elif selected_section == "OLAP Analysis":
        st.subheader("OLAP Operations")
        
        # Make sure release_year exists and drop naive NaNs for year sliders
        if 'release_year' in df.columns:
            valid_years_df = df.dropna(subset=['release_year'])
            min_year = int(valid_years_df['release_year'].min())
            max_year = int(valid_years_df['release_year'].max())
        else:
            min_year, max_year = 1900, 2025 # Fallbacks
        
        # 1. SLICE
        with st.expander("🔪 SLICE: Filter by a single dimension (e.g., Year)", expanded=True):
            slice_year = st.slider("Select Release Year", min_value=min_year, max_value=max_year, value=max_year, key="slice_year")
            sliced_df = df[df['release_year'] == slice_year]
            st.write(f"**Movies released in {slice_year}:** {len(sliced_df)}")
            if not sliced_df.empty:
                cols_to_show = [c for c in ['title', 'original_title', 'revenue', 'vote_average', 'release_date'] if c in sliced_df.columns]
                st.dataframe(sliced_df[cols_to_show].head(50), use_container_width=True)
            else:
                st.info(f"No movies found for {slice_year}")

        # 2. DICE
        with st.expander("🎲 DICE: Filter by multiple dimensions"):
            col1, col2, col3 = st.columns(3)
            with col1:
                dice_years = st.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="dice_years")
            with col2:
                min_rev = float(df['revenue'].min()) if 'revenue' in df.columns else 0.0
                max_rev = float(df['revenue'].max()) if 'revenue' in df.columns else 1000000.0
                dice_min_revenue = st.number_input("Minimum Revenue", min_value=min_rev, max_value=max_rev, value=min_rev, step=100000.0)
            with col3:
                dice_min_vote = st.number_input("Minimum Vote Average", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
            
            diced_df = df[
                (df['release_year'] >= dice_years[0]) & 
                (df['release_year'] <= dice_years[1]) & 
                (df['revenue'] >= dice_min_revenue) & 
                (df['vote_average'] >= dice_min_vote)
            ]
            st.write(f"**Filtered Results:** {len(diced_df)} movies match these criteria.")
            if not diced_df.empty:
                cols_to_show = [c for c in ['title', 'original_title', 'release_year', 'revenue', 'vote_average'] if c in diced_df.columns]
                st.dataframe(diced_df[cols_to_show].head(50), use_container_width=True)

        # 3. ROLL-UP
        with st.expander("📈 ROLL-UP: Aggregate data across dimensions"):
            st.markdown("**(a) Total Revenue by Year**")
            rollup_yearly = df.groupby('release_year')['revenue'].sum().reset_index()
            fig_rollup_line = px.line(rollup_yearly, x='release_year', y='revenue', markers=True, 
                                      title="Total Revenue by Year", color_discrete_sequence=["#eab308"])
            fig_rollup_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_rollup_line, use_container_width=True)
            
            st.markdown("**(b) Revenue Heatmap by Year & Month**")
            if 'release_month' in df.columns:
                heatmap_data = df.groupby(['release_year', 'release_month'])['revenue'].sum().reset_index()
                heatmap_pivot = heatmap_data.pivot(index="release_year", columns="release_month", values="revenue").fillna(0)
                fig_heatmap = px.imshow(heatmap_pivot, 
                                        labels=dict(x="Month", y="Year", color="Revenue"),
                                        x=heatmap_pivot.columns, 
                                        y=heatmap_pivot.index,
                                        aspect="auto",
                                        color_continuous_scale="Viridis",
                                        title="Revenue Heatmap by Year & Month")
                fig_heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Month dimensions missing from dataset.")

        # 4. DRILL-DOWN
        with st.expander("🔍 DRILL-DOWN: Break down aggregated data"):
            drill_year = st.selectbox("Select Year to Drill-down into Months", 
                                      options=sorted(df['release_year'].dropna().unique(), reverse=True))
            if 'release_month' in df.columns:
                drill_df = df[df['release_year'] == drill_year]
                monthly_rev = drill_df.groupby('release_month')['revenue'].sum().reset_index()
                
                # Make sure all 12 months present
                all_months = pd.DataFrame({'release_month': range(1, 13)})
                monthly_rev = pd.merge(all_months, monthly_rev, on='release_month', how='left').fillna({'revenue': 0})
                
                fig_drill = px.bar(monthly_rev, x='release_month', y='revenue', 
                                   title=f"Monthly Revenue for {int(drill_year)}",
                                   labels={"release_month": "Month", "revenue": "Revenue"},
                                   color_discrete_sequence=["#06b6d4"])
                fig_drill.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_drill, use_container_width=True)
            else:
                st.warning("Month dimensions missing from dataset.")
        
    elif selected_section == "Data Transformation":
        st.subheader("Data Transformation & Statistical Testing")
        
        # 1. Equal Width Binning
        with st.expander("📏 1. EQUAL WIDTH BINNING", expanded=True):
            st.markdown("Bin `revenue` into 5 equal-width intervals.")
            if 'revenue' in df.columns:
                df['revenue_bin_equal_width'] = pd.cut(df['revenue'], bins=5)
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['revenue', 'revenue_bin_equal_width']].head(10), use_container_width=True)
                with col2:
                    bin_counts = df['revenue_bin_equal_width'].value_counts().sort_index().reset_index()
                    bin_counts.columns = ['Bin', 'Count']
                    bin_counts['Bin'] = bin_counts['Bin'].astype(str)
                    fig_ew = px.bar(bin_counts, x='Bin', y='Count', title="Equal Width Bin Counts", color_discrete_sequence=["#3b82f6"])
                    fig_ew.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_ew, use_container_width=True)
            else:
                st.warning("Revenue column missing.")

        # 2. Equal Frequency Binning
        with st.expander("⚖️ 2. EQUAL FREQUENCY BINNING"):
            st.markdown("Bin `revenue` into 5 quantiles (equal number of records).")
            if 'revenue' in df.columns:
                try:
                    df['revenue_bin_equal_freq'] = pd.qcut(df['revenue'], q=5, duplicates='drop')
                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        st.dataframe(df[['revenue', 'revenue_bin_equal_freq']].head(10), use_container_width=True)
                    with col2:
                        freq_counts = df['revenue_bin_equal_freq'].value_counts().sort_index().reset_index()
                        freq_counts.columns = ['Bin', 'Count']
                        freq_counts['Bin'] = freq_counts['Bin'].astype(str)
                        fig_ef = px.bar(freq_counts, x='Bin', y='Count', title="Equal Frequency Bin Counts", color_discrete_sequence=["#10b981"])
                        fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_ef, use_container_width=True)
                except Exception as e:
                    st.error(f"Error producing quantiles: {e}")
            else:
                st.warning("Revenue column missing.")

        # 3. Custom Binning
        with st.expander("🎨 3. CUSTOM BINNING"):
            st.markdown("Bin `revenue` into custom tiers: Low, Medium, High, Blockbuster.")
            if 'revenue' in df.columns:
                max_rev = df['revenue'].max()
                # If maximum revenue happens to be smaller than 500M, ensure right bound of 'Blockbuster' goes beyond it
                safe_max = max(500000000, max_rev) + 1
                bins = [-1, 50000000, 200000000, 500000000, safe_max]
                labels = ['Low', 'Medium', 'High', 'Blockbuster']
                
                df['revenue_custom_tier'] = pd.cut(df['revenue'], bins=bins, labels=labels)
                
                tier_counts = df['revenue_custom_tier'].value_counts().reset_index()
                tier_counts.columns = ['Tier', 'Count']
                fig_pie = px.pie(tier_counts, names='Tier', values='Count', title="Custom Revenue Tiers Distribution",
                                 color_discrete_sequence=px.colors.sequential.Plasma)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Revenue column missing.")

        # 4. Min-Max Normalization
        with st.expander("📉 4. MIN-MAX NORMALIZATION"):
            st.markdown("Scale `budget` to a range of [0, 1].")
            if 'budget' in df.columns:
                scaler_minmax = MinMaxScaler()
                df['budget_minmax'] = scaler_minmax.fit_transform(df[['budget']])
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['budget', 'budget_minmax']].head(10), use_container_width=True)
                with col2:
                    fig_minmax = px.histogram(df, x='budget_minmax', nbins=30, title="Distribution of Budget (Min-Max)", 
                                              color_discrete_sequence=["#8b5cf6"])
                    fig_minmax.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_minmax, use_container_width=True)
            else:
                st.warning("Budget column missing.")

        # 5. Z-Score Normalization
        with st.expander("📊 5. Z-SCORE NORMALIZATION"):
            st.markdown("Standardize `budget` (mean=0, std=1).")
            if 'budget' in df.columns:
                scaler_z = StandardScaler()
                df['budget_zscore'] = scaler_z.fit_transform(df[['budget']])
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df[['budget', 'budget_zscore']].head(10), use_container_width=True)
                with col2:
                    fig_z = px.histogram(df, x='budget_zscore', nbins=30, title="Distribution of Budget (Z-Score)", 
                                         color_discrete_sequence=["#f43f5e"])
                    fig_z.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_z, use_container_width=True)
            else:
                st.warning("Budget column missing.")

        # Chi-Square Test
        st.markdown("---")
        st.subheader("🧪 Chi-Square Test of Independence")
        st.markdown("Testing independence between **Revenue Category** and **Vote Category**.")
        if 'revenue' in df.columns and 'vote_average' in df.columns:
            # Create Revenue Category (Low/Medium/High)
            df['chi2_rev_cat'] = pd.qcut(df['revenue'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            
            # Create Vote Category (Low Rating/Medium Rating/High Rating)
            df['chi2_vote_cat'] = pd.qcut(df['vote_average'], q=3, labels=['Low Rating', 'Medium Rating', 'High Rating'], duplicates='drop')
            
            # Contingency table
            contingency_table = pd.crosstab(df['chi2_rev_cat'], df['chi2_vote_cat'])
            
            st.markdown("**Contingency Table:**")
            st.dataframe(contingency_table, use_container_width=True)
            
            # Perform Chi-Square Test
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-Square Statistic", f"{chi2:.2f}")
            col2.metric("P-value", f"{p_val:.4e}")
            col3.metric("Degrees of Freedom", dof)
            
            if p_val < 0.05:
                st.success(f"**Significant Result (p < 0.05):** We reject the null hypothesis. There is a statistically significant association between Revenue Category and Vote Category.")
            else:
                st.warning(f"**Not Significant (p >= 0.05):** We fail to reject the null hypothesis. There is no statistically significant association between Revenue Category and Vote Category.")
        else:
            st.warning("Revenue or Vote Average columns missing.")
        
    elif selected_section == "Association Rules (Apriori + FP-Growth)":
        st.subheader("Associations & Frequent Itemsets")
        
        # Helper function to extract names from JSON-like list of dicts string
        def extract_names(json_str):
            try:
                lst = ast.literal_eval(json_str)
                return [item['name'] for item in lst] if isinstance(lst, list) else []
            except (ValueError, SyntaxError):
                return []
                
        # SECTION A: Apriori on Genres
        with st.expander("📚 SECTION A: Apriori on Genres", expanded=True):
            if 'genres' in df.columns:
                st.markdown("**Frequent Genres using Apriori Algorithm**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_support_g = st.slider("Minimum Support (Genres)", 0.01, 0.50, 0.05, 0.01, key='msg')
                with col2:
                    min_confidence_g = st.slider("Minimum Confidence (Genres)", 0.1, 1.0, 0.6, 0.05, key='mcg')
                
                # Extract genres
                genre_lists = df['genres'].dropna().apply(extract_names).tolist()
                genre_lists = [g for g in genre_lists if len(g) > 0]
                
                if genre_lists:
                    # One-hot encode
                    te = TransactionEncoder()
                    te_ary = te.fit(genre_lists).transform(genre_lists)
                    df_genres = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # Run apriori
                    with st.spinner("Running Apriori algorithm on genres..."):
                        freq_items_apriori = apriori(df_genres, min_support=min_support_g, use_colnames=True)
                    
                    if not freq_items_apriori.empty:
                        # Top 10 by support chart
                        freq_items_apriori['itemsets_str'] = freq_items_apriori['itemsets'].apply(lambda x: ', '.join(list(x)))
                        top10_genres = freq_items_apriori.sort_values('support', ascending=False).head(10)
                        
                        fig_g = px.bar(top10_genres, x='itemsets_str', y='support', title="Top 10 Genres (Support)", color_discrete_sequence=["#8b5cf6"])
                        fig_g.update_layout(xaxis_title="Genre", yaxis_title="Support", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_g, use_container_width=True)
                        
                        # Generate rules
                        rules_g = association_rules(freq_items_apriori, metric="confidence", min_threshold=min_confidence_g)
                        
                        if not rules_g.empty:
                            rules_g['antecedents'] = rules_g['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_g['consequents'] = rules_g['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules_g = rules_g.sort_values('lift', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                            st.markdown(f"**Discovered {len(rules_g)} rules (Sorted by Lift Descending)**")
                            st.dataframe(rules_g.style.background_gradient(subset=['lift'], cmap='Blues'), use_container_width=True)
                        else:
                            st.info("No explicit rules found matching the confidence threshold.")
                    else:
                        st.warning("No frequent itemsets found. Try lowering the support threshold.")
                else:
                    st.warning("Could not extract any genres.")
            else:
                st.warning("Genres column missing.")

        # SECTION B: Apriori on Keywords
        with st.expander("🔑 SECTION B: Apriori on Keywords"):
            if 'keywords' in df.columns:
                st.markdown("**Frequent Keywords using Apriori Algorithm**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_support_k = st.slider("Minimum Support (Keywords)", 0.01, 0.50, 0.03, 0.01, key='msk')
                with col2:
                    min_confidence_k = st.slider("Minimum Confidence (Keywords)", 0.1, 1.0, 0.6, 0.05, key='mck')
                
                # Extract keywords
                keyword_lists = df['keywords'].dropna().apply(extract_names).tolist()
                keyword_lists = [k for k in keyword_lists if len(k) > 0]
                
                if keyword_lists:
                    # One-hot encode
                    te_k = TransactionEncoder()
                    te_ary_k = te_k.fit(keyword_lists).transform(keyword_lists)
                    df_keywords = pd.DataFrame(te_ary_k, columns=te_k.columns_)
                    
                    # Run apriori
                    with st.spinner("Running Apriori algorithm on keywords..."):
                        freq_items_k = apriori(df_keywords, min_support=min_support_k, use_colnames=True)
                    
                    if not freq_items_k.empty:
                        # Top 10 by support chart
                        freq_items_k['itemsets_str'] = freq_items_k['itemsets'].apply(lambda x: ', '.join(list(x)))
                        top10_keys = freq_items_k.sort_values('support', ascending=False).head(10)
                        
                        fig_k = px.bar(top10_keys, x='itemsets_str', y='support', title="Top 10 Keywords (Support)", color_discrete_sequence=["#10b981"])
                        fig_k.update_layout(xaxis_title="Keyword", yaxis_title="Support", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_k, use_container_width=True)
                        
                        # Generate rules
                        rules_k = association_rules(freq_items_k, metric="confidence", min_threshold=min_confidence_k)
                        
                        if not rules_k.empty:
                            rules_k['antecedents'] = rules_k['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_k['consequents'] = rules_k['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules_k = rules_k.sort_values('lift', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                            st.markdown(f"**Discovered {len(rules_k)} rules (Sorted by Lift Descending)**")
                            st.dataframe(rules_k.style.background_gradient(subset=['lift'], cmap='Greens'), use_container_width=True)
                        else:
                            st.info("No explicit rules found matching the confidence threshold.")
                    else:
                        st.warning("No frequent itemsets found. Try lowering the support threshold.")
                else:
                    st.warning("Could not extract any keywords.")
            else:
                st.warning("Keywords column missing.")

        # SECTION C: FP-Growth on Genres
        with st.expander("⚡ SECTION C: FP-Growth on Genres"):
            if 'genres' in df.columns:
                st.markdown("**Frequent Genres using FP-Growth Algorithm**")
                st.info("💡 **Comparison Note:** FP-Growth is faster than Apriori for large datasets as it avoids candidate generation by compressing the data into an FP-tree.")
                
                # Hardcoded params to match requirements
                fp_min_support = 0.05
                fp_min_conf = 0.6
                
                if genre_lists:
                    # Reuse genre_lists and df_genres from section A if available
                    if 'df_genres' not in locals() or df_genres.empty:
                        te = TransactionEncoder()
                        te_ary = te.fit(genre_lists).transform(genre_lists)
                        df_genres = pd.DataFrame(te_ary, columns=te.columns_)
                        
                    # Run FP-Growth
                    with st.spinner("Running FP-Growth algorithm on genres..."):
                        freq_items_fp = fpgrowth(df_genres, min_support=fp_min_support, use_colnames=True)
                    
                    if not freq_items_fp.empty:
                        rules_fp = association_rules(freq_items_fp, metric="confidence", min_threshold=fp_min_conf)
                        
                        if not rules_fp.empty:
                            rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules_fp = rules_fp.sort_values('lift', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                            st.markdown(f"**Discovered {len(rules_fp)} rules (Support >= {fp_min_support}, Confidence >= {fp_min_conf})**")
                            st.dataframe(rules_fp.style.background_gradient(subset=['lift'], cmap='Oranges'), use_container_width=True)
                        else:
                            st.info("No explicit rules found matching the confidence threshold.")
                    else:
                        st.warning("No frequent itemsets found with FP-Growth.")
                else:
                    st.warning("Could not extract any genres.")
            else:
                st.warning("Genres column missing.")
        
    elif selected_section == "PCA":
        st.subheader("Principal Component Analysis (PCA)")
        st.markdown("Reducing dimensionality for numerical features: `budget`, `popularity`, `revenue`, `runtime`, `vote_average`, `vote_count`")
        
        features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if not missing_features:
            # Prepare data
            X = df[features].copy()
            
            # Formally fill NaNs with median just in case (most were filled in Preprocessing)
            for col in features:
                X[col] = X[col].fillna(X[col].median())
                
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # --- FULL PCA (6 Components) ---
            with st.spinner("Running Principal Component Analysis..."):
                pca_full = PCA()
                pca_full.fit(X_scaled)
                
                explained_var = pca_full.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Find components needed for 85% variance
            n_components_85 = np.argmax(cumulative_var >= 0.85) + 1
            
            st.markdown("### 1. Variance Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plotly line chart with two lines
                fig_var = go.Figure()
                fig_var.add_trace(go.Scatter(
                    x=list(range(1, len(explained_var) + 1)),
                    y=explained_var,
                    mode='lines+markers',
                    name='Individual Explained Variance',
                    line=dict(color='#3b82f6')
                ))
                fig_var.add_trace(go.Scatter(
                    x=list(range(1, len(cumulative_var) + 1)),
                    y=cumulative_var,
                    mode='lines+markers',
                    name='Cumulative Explained Variance',
                    line=dict(color='#10b981')
                ))
                
                # Add 0.85 threshold line
                fig_var.add_hline(y=0.85, line_dash="dash", line_color="#ef4444", annotation_text="85% Threshold")
                
                fig_var.update_layout(
                    title="PCA Explained Variance",
                    xaxis_title="Principal Component",
                    yaxis_title="Explained Variance Ratio",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_var, use_container_width=True)
                
            with col2:
                st.metric("Components needed for 85% variance", n_components_85)
                
                var_df = pd.DataFrame({
                    'Component': [f"PC{i+1}" for i in range(len(explained_var))],
                    'Individual Variance': np.round(explained_var, 4),
                    'Cumulative Variance': np.round(cumulative_var, 4)
                })
                st.dataframe(var_df, use_container_width=True)
                
            # --- REDUCED PCA (2 Components) ---
            st.markdown("---")
            st.markdown("### 2. 2D Projection")
            
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            
            pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
            
            # Keep original indices for matching
            pca_df.index = df.index
            
            # Create vote category for color
            # Use fixed bins: <6 (Low), 6-7 (Medium), >7 (High) or dynamic qcut
            try:
                rating_cat = pd.qcut(df['vote_average'], q=3, labels=['Low Rating', 'Medium Rating', 'High Rating'], duplicates='drop')
            except ValueError:
                rating_cat = pd.cut(df['vote_average'], bins=[-1, 5, 7, 10], labels=['Low Rating', 'Medium Rating', 'High Rating'])
                
            pca_df['Rating Category'] = rating_cat
            
            # Map original titles for hover info if available
            if 'original_title' in df.columns:
                pca_df['Title'] = df['original_title']
            elif 'title' in df.columns:
                pca_df['Title'] = df['title']
            else:
                pca_df['Title'] = "Movie " + pca_df.index.astype(str)
                
            col3, col4 = st.columns([2, 1])
            
            with col3:
                fig_scatter = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color='Rating Category', 
                    hover_name='Title',
                    title="2D PCA Scatter Plot Colored by Vote Rating",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col4:
                st.markdown("**First 10 rows of mapped PC dimensions**")
                # Exclude Title from displayed dataframe to match shape requirements more strictly
                display_df = pca_df[['PC1', 'PC2', 'Rating Category']]
                st.dataframe(display_df.head(10), use_container_width=True)
                
            total_explained_2d = np.sum(pca_2d.explained_variance_ratio_) * 100
            st.info(f"💡 Dataset reduced from 6 dimensions to 2 dimensions. PC1 + PC2 explain {total_explained_2d:.2f}% of variance.")
            
        else:
            st.warning(f"Cannot perform PCA. Missing features: {', '.join(missing_features)}")
        
    elif selected_section == "Classification (Decision Tree, Naive Bayes, KNN)":
        st.subheader("Classification Models")
        st.markdown("Predicting whether a movie is a **Hit** (revenue >= median revenue) based on features.")
        
        # SHARED SETUP
        features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
        if all(f in df.columns for f in features) and 'revenue' in df.columns:
            # Drop rows missing target/features
            ml_df = df.dropna(subset=features + ['revenue']).copy()
            
            # Formally fill remaining feature NaNs with median just in case
            for col in features:
                ml_df[col] = ml_df[col].fillna(ml_df[col].median())
                
            # Create Target: hit = 1 if revenue >= median_revenue else 0
            median_revenue = ml_df['revenue'].median()
            ml_df['hit'] = (ml_df['revenue'] >= median_revenue).astype(int)
            
            X = ml_df[features]
            y = ml_df['hit']
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.write(f"**Dataset:** {len(X_train)} training samples, {len(X_test)} testing samples")
            
            # Tabs for models
            tab1, tab2, tab3 = st.tabs(["Decision Tree", "Naive Bayes", "KNN"])
            
            # TAB 1: Decision Tree
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
                    fig_cm_dt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_cm_dt, use_container_width=True)
                    
                st.markdown("**Tree Visualization (Text)**")
                tree_rules = export_text(dt, feature_names=features)
                st.code(tree_rules, language="text")

            # TAB 2: Naive Bayes
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
                    fig_cm_nb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_cm_nb, use_container_width=True)

            # TAB 3: KNN
            with tab3:
                st.markdown("### K-Nearest Neighbors (KNN)")
                
                # KNN needs scaling
                scaler_knn = StandardScaler()
                X_train_scaled = scaler_knn.fit_transform(X_train)
                X_test_scaled = scaler_knn.transform(X_test)
                
                # Default K=5
                knn_5 = KNeighborsClassifier(n_neighbors=5)
                knn_5.fit(X_train_scaled, y_train)
                acc_5 = accuracy_score(y_test, knn_5.predict(X_test_scaled))
                st.write(f"Initial accuracy with **K=5**: {acc_5 * 100:.2f}%")
                
                # Find best K
                k_values = range(1, 21)
                accuracies = []
                for k in k_values:
                    knn_temp = KNeighborsClassifier(n_neighbors=k)
                    knn_temp.fit(X_train_scaled, y_train)
                    accuracies.append(accuracy_score(y_test, knn_temp.predict(X_test_scaled)))
                
                best_k = k_values[np.argmax(accuracies)]
                best_acc = max(accuracies)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(f"Best K (1-20)", f"{best_k}", f"Accuracy: {best_acc * 100:.2f}%", delta_color="normal")
                    
                    st.markdown("---")
                    st.markdown("**Test a Custom K**")
                    custom_k = st.number_input("Enter K value to train:", min_value=1, max_value=100, value=best_k, step=1)
                    
                    knn_custom = KNeighborsClassifier(n_neighbors=custom_k)
                    knn_custom.fit(X_train_scaled, y_train)
                    y_pred_custom = knn_custom.predict(X_test_scaled)
                    acc_custom = accuracy_score(y_test, y_pred_custom)
                    
                    st.success(f"Accuracy with K={custom_k}: **{acc_custom * 100:.2f}%**")
                    st.text(classification_report(y_test, y_pred_custom))
                    
                with col2:
                    # Line chart of Acc vs K
                    fig_k = px.line(x=list(k_values), y=accuracies, markers=True, 
                                    title="Accuracy vs K Value", labels={'x': 'Number of Neighbors (K)', 'y': 'Accuracy'},
                                    color_discrete_sequence=["#eab308"])
                    fig_k.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", xaxis=dict(tickmode='linear', dtick=1))
                    
                    # Highlight best K
                    fig_k.add_scatter(x=[best_k], y=[best_acc], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Best K')
                    
                    st.plotly_chart(fig_k, use_container_width=True)
                    
                    # Confusion matrix for Custom K
                    cm_knn = confusion_matrix(y_test, y_pred_custom)
                    fig_cm_knn = px.imshow(cm_knn, text_auto=True, color_continuous_scale='Oranges',
                                          labels=dict(x="Predicted", y="Actual", color="Count"),
                                          x=['Not Hit (0)', 'Hit (1)'], y=['Not Hit (0)', 'Hit (1)'],
                                          title=f"Confusion Matrix (K={custom_k})")
                    fig_cm_knn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_cm_knn, use_container_width=True)
                
        else:
            st.warning("Missing required features for Classification. Check dataset completion.")
        
    elif selected_section == "Linear Regression":
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
                
                # Split
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
                
                # Scale Features
                scaler_reg = StandardScaler()
                X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
                X_test_reg_scaled = scaler_reg.transform(X_test_reg)
                
                # Train Model
                lr_model = LinearRegression()
                lr_model.fit(X_train_reg_scaled, y_train_reg)
                y_pred_reg = lr_model.predict(X_test_reg_scaled)
                
                # Metrics
                mse = mean_squared_error(y_test_reg, y_pred_reg)
                r2 = r2_score(y_test_reg, y_pred_reg)
            
            st.write(f"**Dataset:** {len(X_train_reg)} training samples, {len(X_test_reg)} testing samples")
            
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:,.0f}")
            col2.metric("R² Score", f"{r2:.4f}", help="Close to 1 is better, close to 0 is worse")
            
            st.markdown("---")
            
            col3, col4 = st.columns([1, 1.2])
            
            with col3:
                # Coefficients Bar Chart
                st.markdown("**Feature Coefficients**")
                coef_df = pd.DataFrame({
                    'Feature': features_reg,
                    'Coefficient': lr_model.coef_
                })
                coef_df = coef_df.sort_values(by='Coefficient', ascending=True)
                # Color code positive (blue) and negative (red)
                coef_df['Color'] = np.where(coef_df['Coefficient'] > 0, "#3b82f6", "#ef4444")
                
                fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                  color='Color', color_discrete_map="identity",
                                  title="Impact of Features on Revenue")
                fig_coef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_coef, use_container_width=True)
                
            with col4:
                # Actual vs Predicted Scatter
                st.markdown("**Actual vs Predicted Revenue**")
                avsp_df = pd.DataFrame({
                    'Actual Revenue': y_test_reg,
                    'Predicted Revenue': y_pred_reg
                })
                
                fig_scatter = px.scatter(avsp_df, x='Actual Revenue', y='Predicted Revenue', 
                                         opacity=0.6,
                                         title="Actual vs Predicted Revenue — points near diagonal = good model",
                                         color_discrete_sequence=["#8b5cf6"])
                
                # Diagonal Reference Line
                max_val = max(avsp_df['Actual Revenue'].max(), avsp_df['Predicted Revenue'].max())
                min_val = min(avsp_df['Actual Revenue'].min(), avsp_df['Predicted Revenue'].min())
                fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                                 mode='lines', name='Perfect Prediction',
                                                 line=dict(color='#ef4444', dash='dash')))
                
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.warning("Missing required features for Linear Regression. Check dataset completion.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9em;'><strong>&copy; 2026 Built by Jesu Mariya Joe D (2361018) & Keya (2361023)</strong><br>CIA 3.1 Mini Project | TMDB 5000 Dataset | Techniques: EDA, OLAP, Binning, Normalization, Chi-Square, Apriori, FP-Growth, PCA, Decision Tree, Naive Bayes, KNN, Linear Regression</p>", unsafe_allow_html=True)
