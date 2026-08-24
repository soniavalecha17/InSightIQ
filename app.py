import streamlit as st
import pandas as pd
import os

# Import your Phase 1 Analytics modules
from analytics.loader import DatasetLoader
from analytics.profiler import DatasetProfiler
from analytics.cleaner import DataCleaner
from analytics.statistics import StatisticalAnalyzer
from analytics.insights import InsightGenerator

# Page Configuration
st.set_page_config(
    page_title="InsightIQ | Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

def main():
    # --- SIDEBAR ---
    st.sidebar.title("INSIGHTIQ")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Dataset")
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    # Fallback to sample dataset if nothing is uploaded
    default_path = "data/sample.csv"
    df = None
    dataset_name = "No dataset selected"

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    elif os.path.exists(default_path):
        if st.sidebar.checkbox("Use default sample.csv", value=True):
            df = pd.read_csv(default_path)
            dataset_name = "sample.csv"

    # --- MAIN PAGE HEADER ---
    st.title("📊 INSIGHTIQ")
    st.markdown("### **Data Intelligence Platform**")
    st.markdown("---")

    if df is not None:
        # 1. Profile Data
        profiler = DatasetProfiler(df)
        profile = profiler.generate_profile()
        
        rows, cols = df.shape
        total_cells = rows * cols
        total_missing = profile['quality_report']['Missing Values'].sum()
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        st.markdown(f"**Active Dataset:** `{dataset_name}`")

        # --- KPI METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{rows:,}")
        col2.metric("Columns", f"{cols}")
        col3.metric("Missing Values", f"{missing_pct:.1f}%")
        col4.metric("Duplicate Rows", f"{profile['duplicates']}")

        st.markdown("---")

        # --- TABS FOR ORGANIZATION ---
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Dataset Preview", "📈 Statistics", "📉 Visualizations", "💡 Key Insights"])

        with tab1:
            st.subheader("Dataset Preview (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            st.subheader("Data Quality Report")
            st.dataframe(profile['quality_report'], use_container_width=True)

        with tab2:
            st.subheader("Automated Statistical Analysis")
            analyzer = StatisticalAnalyzer(df)
            
            num_stats = analyzer.compute_numerical_stats()
            if num_stats:
                st.markdown("#### Numerical Columns")
                num_df = pd.DataFrame(num_stats).T
                st.dataframe(num_df, use_container_width=True)
                
            cat_stats = analyzer.compute_categorical_stats()
            if cat_stats:
                st.markdown("#### Categorical Summary")
                for col, metrics in cat_stats.items():
                    with st.expander(f"Column: {col} (Unique: {metrics['unique_values']})"):
                        st.write(f"**Most Frequent:** {metrics['most_frequent']}")
                        st.write("**Distribution Percentage:**")
                        st.json(metrics['distribution_percentage'])

        with tab3:
            st.subheader("Automated Visualizations")
            st.info("Charts generated from your analytics engine rules:")
            
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

            # Interactive Streamlit Native Charts
            if num_cols:
                selected_num_col = st.selectbox("Select numerical column for distribution (Histogram)", num_cols)
                st.bar_chart(df[selected_num_col].value_counts().sort_index())

            if len(num_cols) >= 2:
                st.markdown("#### Correlation Matrix")
                corr = df[num_cols].corr()
                st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)

        with tab4:
            st.subheader("💡 Automated Insight Generator")
            insight_engine = InsightGenerator(df, profile)
            key_insights = insight_engine.generate_insights()

            if key_insights:
                for insight in key_insights:
                    st.success(f"• {insight}")
            else:
                st.info("No major anomalies or rule-based triggers found for this dataset layout.")

    else:
        st.info("👈 Please upload a CSV file via the sidebar or place a `sample.csv` inside your `data/` folder to get started!")

if __name__ == "__main__":
    main()