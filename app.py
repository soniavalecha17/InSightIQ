import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Import your Phase 1 Analytics modules
from analytics.loader import DatasetLoader
from analytics.profiler import DatasetProfiler
from analytics.cleaner import DataCleaner
from analytics.statistics import StatisticalAnalyzer
from analytics.insights import InsightGenerator

# Phase 2A/2B Imports
from analytics.ai_analyst import AIAnalyst
from analytics.query_engine import QueryEngine

# Page Configuration
st.set_page_config(
    page_title="InsightIQ | Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

def find_column_by_keywords(df, keywords):
    """Semantic column detector using case-insensitive keywords and substring matching."""
    for col in df.columns:
        col_name = col.lower().replace("_", " ").strip()
        for keyword in keywords:
            if keyword in col_name:
                return col
    return None

def find_best_categorical(df):
    """Selects a useful categorical column avoiding IDs or unique keys (like Name or ID)."""
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    candidates = []
    
    for col in categorical_cols:
        unique = df[col].nunique(dropna=True)
        # Avoid columns where almost every row is unique (e.g. IDs, names)
        if 2 <= unique <= min(30, len(df) * 0.3):
            candidates.append(col)
            
    if candidates:
        return candidates[0]
    return categorical_cols[0] if len(categorical_cols) > 0 else None

def get_dynamic_kpis(df):
    """
    Automatically detects meaningful semantic and statistical KPIs based on the uploaded dataset.
    """
    kpis = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # KPI 1: Total Records / Items / Customers / Properties
    row_count = len(df)
    rec_label = "📋 TOTAL RECORDS"
    if find_column_by_keywords(df, ["customer", "client", "user"]):
        rec_label = "👥 TOTAL CUSTOMERS"
    elif find_column_by_keywords(df, ["student", "pupil"]):
        rec_label = "🎓 TOTAL STUDENTS"
    elif find_column_by_keywords(df, ["employee", "staff"]):
        rec_label = "👔 TOTAL EMPLOYEES"
    elif find_column_by_keywords(df, ["property", "house", "building"]):
        rec_label = "🏠 TOTAL PROPERTIES"
    elif find_column_by_keywords(df, ["product", "item", "sku"]):
        rec_label = "📦 TOTAL PRODUCTS"

    kpis.append({
        "label": rec_label,
        "value": f"{row_count:,}"
    })

    # KPI 2: Semantic Numeric Match (Age, Salary, Price, Sales, Balance, Score)
    age_col = find_column_by_keywords(df, ["age", "customer_age", "user_age"])
    salary_col = find_column_by_keywords(df, ["salary", "income", "revenue", "sales", "balance", "amount", "price", "cost", "score", "rating"])
    
    selected_num = age_col if age_col else salary_col
    if not selected_num and numeric_cols:
        selected_num = numeric_cols[0]

    if selected_num and pd.api.types.is_numeric_dtype(df[selected_num]):
        mean_val = df[selected_num].mean()
        col_lower = selected_num.lower()
        if "age" in col_lower:
            label = "📊 AVERAGE AGE"
            val_str = f"{mean_val:,.1f}"
        elif any(k in col_lower for k in ["salary", "income", "revenue", "sales", "balance", "amount", "price", "cost"]):
            label = f"💰 AVG {selected_num.replace('_', ' ').upper()}"
            val_str = f"{mean_val:,.2f}"
        elif any(k in col_lower for k in ["score", "rating"]):
            label = f"⭐ AVG {selected_num.replace('_', ' ').upper()}"
            val_str = f"{mean_val:,.2f}"
        else:
            label = f"📊 AVG {selected_num.replace('_', ' ').upper()}"
            val_str = f"{mean_val:,.2f}"

        kpis.append({"label": label, "value": val_str})

    # KPI 3: Second Numeric / Median or Rate
    second_num = None
    for col in numeric_cols:
        if col != selected_num:
            second_num = col
            break

    if second_num and pd.api.types.is_numeric_dtype(df[second_num]):
        med_val = df[second_num].median()
        kpis.append({
            "label": f"📈 MEDIAN {second_num.replace('_', ' ').upper()}",
            "value": f"{med_val:,.2f}"
        })

    # KPI 4: Binary/Outcome or Unique Category Rate
    binary_col = None
    for col in categorical_cols:
        if df[col].dropna().nunique() == 2:
            binary_col = col
            break

    if binary_col:
        vc = df[binary_col].dropna().value_counts()
        pct = (vc.iloc[0] / vc.sum()) * 100
        kpis.append({
            "label": f"📌 {binary_col.replace('_', ' ').upper()} RATE",
            "value": f"{pct:.1f}%"
        })
    else:
        best_cat = find_best_categorical(df)
        if best_cat:
            unique_cnt = df[best_cat].nunique()
            kpis.append({
                "label": f"🔹 UNIQUE {best_cat.replace('_', ' ').upper()}",
                "value": f"{unique_cnt:,}"
            })

    return kpis[:4]

def get_dynamic_suggestions(df):
    """Generates context-aware sample questions for the AI Analyst based on dataset schema."""
    suggestions = ["Total records?", "Summary statistics?"]
    
    if find_column_by_keywords(df, ["age"]):
        suggestions.append("Average age?")
    if find_column_by_keywords(df, ["balance", "salary", "income", "price", "amount", "revenue"]):
        suggestions.append("Average balance or salary?")
    if find_column_by_keywords(df, ["default", "subscribed", "target", "churn", "status", "outcome"]):
        suggestions.append("What is the success or subscription rate?")
    if find_column_by_keywords(df, ["job", "department", "category", "education"]):
        suggestions.append("Breakdown by category?")
        
    while len(suggestions) < 6:
        suggestions.append("Tell me about this dataset.")
        
    return suggestions[:6]

def render_visualization(instruction: dict, raw_result: any, df: pd.DataFrame):
    """Renders Plotly charts based on the AI analyst's instructions and execution results."""
    chart_type = instruction.get("chart_type", "none")
    chart_title = instruction.get("chart_title", "Data Visualization")
    group_col = instruction.get("group_col")
    target_col = instruction.get("target_col")

    if chart_type == "none" or not chart_type:
        return

    try:
        if isinstance(raw_result, dict) and group_col:
            val_col = target_col if target_col and target_col in df.columns else "Value"
            chart_df = pd.DataFrame(list(raw_result.items()), columns=[group_col, val_col])
            
            if chart_type == "bar":
                fig = px.bar(chart_df, x=group_col, y=val_col, title=chart_title, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "pie":
                fig = px.pie(chart_df, names=group_col, values=val_col, title=chart_title)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "line":
                fig = px.line(chart_df, x=group_col, y=val_col, title=chart_title, markers=True)
                st.plotly_chart(fig, use_container_width=True)

        elif isinstance(raw_result, list) and len(raw_result) > 0:
            chart_df = pd.DataFrame(raw_result)
            if target_col and len(chart_df.columns) >= 2:
                x_col = chart_df.columns[0] if group_col not in chart_df.columns else group_col
                if chart_type == "bar":
                    fig = px.bar(chart_df, x=x_col, y=target_col, title=chart_title, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "line":
                    fig = px.line(chart_df, x=x_col, y=target_col, title=chart_title, markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "scatter" and len(chart_df.columns) >= 2:
                    fig = px.scatter(chart_df, x=x_col, y=target_col, title=chart_title)
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render automated visualization: {e}")

def main():
    # --- SIDEBAR (PHASE 3.1) ---
    st.sidebar.markdown("### **INSIGHTIQ**")
    st.sidebar.markdown("Data Intelligence Platform")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("📂 **DATASET**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
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

    # --- EXPORT REPORT (SIDEBAR) ---
    if df is not None:
        current_df = st.session_state.get("cleaned_df", df)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("📄 **EXPORT**")
        
        profiler_temp = DatasetProfiler(current_df)
        profile_temp = profiler_temp.generate_profile()
        insight_engine_temp = InsightGenerator(current_df, profile_temp)
        insights_temp = insight_engine_temp.generate_insights()
        
        report_content = f"""========================================
   INSIGHTIQ ANALYTICS REPORT
========================================
Dataset Name: {dataset_name}
Total Rows: {current_df.shape[0]:,}
Total Columns: {current_df.shape[1]}
Duplicate Rows: {profile_temp['duplicates']}

----------------------------------------
KEY INSIGHTS & FINDINGS:
----------------------------------------
"""
        for ins in insights_temp:
            report_content += f"• {ins}\n"
            
        report_content += "\n========================================\nGenerated by InsightIQ Engine"

        st.sidebar.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"InsightIQ_Report_{dataset_name.split('.')[0]}.txt",
            mime="text/plain"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About InsightIQ**")
    st.sidebar.markdown("InsightIQ is an advanced AI-powered data intelligence platform built to streamline analytics, data hygiene, and automated exploration.")

    # --- MAIN PAGE HEADER (PHASE 3.1) ---
    st.title("📊 INSIGHTIQ")
    st.markdown("### **Data Intelligence Platform**")
    st.markdown("Turn raw data into actionable insights with AI-powered analytics.")
    st.markdown("---")

    if df is not None:
        if "cleaned_df" not in st.session_state or dataset_name != st.session_state.get("current_dataset"):
            st.session_state.cleaned_df = df.copy()
            st.session_state.current_dataset = dataset_name
            st.session_state.chat_history = []
            st.session_state.pending_suggestion = ""

        active_df = st.session_state.cleaned_df

        # Profile Active Data
        profiler = DatasetProfiler(active_df)
        profile = profiler.generate_profile()
        
        rows, cols = active_df.shape
        total_cells = rows * cols
        total_missing = profile['quality_report']['Missing Values'].sum()
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        st.markdown(f"**Active Dataset:** `{dataset_name}`")

        # --- CONSISTENT METRIC CARDS (PHASE 3.1) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROWS", f"{rows:,}")
        col2.metric("COLUMNS", f"{cols}")
        col3.metric("MISSING", f"{missing_pct:.1f}%")
        col4.metric("DUPLICATES", f"{profile['duplicates']}")

        # --- SEMANTIC COLUMN DETECTION FOR VISUALIZATIONS ---
        numeric_cols = active_df.select_dtypes(include="number").columns.tolist()
        best_cat_col = find_best_categorical(active_df)
        
        # Second category or alternative numeric column for visualizations
        categorical_cols = active_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        secondary_cat_col = categorical_cols[1] if len(categorical_cols) > 1 else best_cat_col
        primary_num_col = numeric_cols[0] if numeric_cols else None
        secondary_num_col = numeric_cols[1] if len(numeric_cols) > 1 else primary_num_col

        # --- PHASE 3.3: FULLY DYNAMIC KPI DASHBOARD ---
        st.markdown("---")
        st.markdown("### 📊 DASHBOARD OVERVIEW")

        dynamic_kpis = get_dynamic_kpis(active_df)
        kpi_columns = st.columns(4)

        for i, kpi in enumerate(dynamic_kpis):
            with kpi_columns[i]:
                st.metric(
                    kpi["label"],
                    kpi["value"]
                )

        # --- FULLY DYNAMIC VISUALIZATIONS SECTION ---
        st.markdown("---")
        st.markdown("#### 📈 Key Business Visualizations")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if best_cat_col:
                cat_counts = active_df[best_cat_col].value_counts().head(10).reset_index()
                cat_counts.columns = [best_cat_col, "Count"]
                fig_cat = px.bar(cat_counts, x=best_cat_col, y="Count", title=f"📊 Distribution of {best_cat_col.replace('_', ' ').title()}", text_auto=True)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Categorical dimension column not available for distribution visualization.")

            if primary_num_col and pd.api.types.is_numeric_dtype(active_df[primary_num_col]):
                fig_num = px.histogram(active_df, x=primary_num_col, title=f"📈 Distribution of {primary_num_col.replace('_', ' ').title()}", nbins=30)
                st.plotly_chart(fig_num, use_container_width=True)

        with chart_col2:
            if secondary_cat_col and secondary_cat_col != best_cat_col:
                sec_counts = active_df[secondary_cat_col].value_counts().head(10).reset_index()
                sec_counts.columns = [secondary_cat_col, "Count"]
                fig_sec = px.bar(sec_counts, x=secondary_cat_col, y="Count", title=f"📌 Distribution of {secondary_cat_col.replace('_', ' ').title()}", text_auto=True)
                st.plotly_chart(fig_sec, use_container_width=True)
            elif best_cat_col and secondary_num_col and pd.api.types.is_numeric_dtype(active_df[secondary_num_col]):
                agg_df = active_df.groupby(best_cat_col)[secondary_num_col].mean().reset_index().head(10)
                fig_agg = px.bar(agg_df, x=best_cat_col, y=secondary_num_col, title=f"📉 Avg {secondary_num_col.replace('_', ' ').title()} by {best_cat_col.replace('_', ' ').title()}", text_auto=True)
                st.plotly_chart(fig_agg, use_container_width=True)
            else:
                st.info("Secondary categorical or numerical dimension not available for secondary breakdown.")

            if secondary_num_col and pd.api.types.is_numeric_dtype(active_df[secondary_num_col]) and secondary_num_col != primary_num_col:
                fig_num2 = px.histogram(active_df, x=secondary_num_col, title=f"📊 Distribution of {secondary_num_col.replace('_', ' ').title()}", nbins=30)
                st.plotly_chart(fig_num2, use_container_width=True)

        # --- DYNAMIC KEY METRICS / DATA HIGHLIGHTS ---
        st.markdown("---")
        st.markdown("### 🏆 KEY METRICS & DATA HIGHLIGHTS")
        hp1, hp2, hp3, hp4 = st.columns(4)

        highlight_1_label = "Top Category"
        highlight_1_val = "N/A"
        if best_cat_col and not active_df.empty:
            top_cat_mode = active_df[best_cat_col].mode()
            if not top_cat_mode.empty:
                highlight_1_label = f"Most Common {best_cat_col.replace('_', ' ').title()}"
                highlight_1_val = str(top_cat_mode.iloc[0])

        highlight_2_label = "Secondary Category"
        highlight_2_val = "N/A"
        if secondary_cat_col and secondary_cat_col != best_cat_col and not active_df.empty:
            sec_mode = active_df[secondary_cat_col].mode()
            if not sec_mode.empty:
                highlight_2_label = f"Most Common {secondary_cat_col.replace('_', ' ').title()}"
                highlight_2_val = str(sec_mode.iloc[0])

        highlight_3_label = "Max Numeric Value"
        highlight_3_val = "N/A"
        if primary_num_col and pd.api.types.is_numeric_dtype(active_df[primary_num_col]) and not active_df.empty:
            max_val = active_df[primary_num_col].max()
            highlight_3_label = f"Max {primary_num_col.replace('_', ' ').title()}"
            highlight_3_val = f"{max_val:,.1f}" if isinstance(max_val, (int, float)) else str(max_val)

        highlight_4_label = "Total Records"
        highlight_4_val = f"{len(active_df):,}"
        if secondary_num_col and pd.api.types.is_numeric_dtype(active_df[secondary_num_col]) and not active_df.empty:
            mean_val = active_df[secondary_num_col].mean()
            highlight_4_label = f"Avg {secondary_num_col.replace('_', ' ').title()}"
            highlight_4_val = f"{mean_val:,.1f}"

        hp1.metric(highlight_1_label, highlight_1_val)
        hp2.metric(highlight_2_label, highlight_2_val)
        hp3.metric(highlight_3_label, highlight_3_val)
        hp4.metric(highlight_4_label, highlight_4_val)

        # --- BUSINESS ALERTS ---
        st.markdown("---")
        st.markdown("### ⚠️ BUSINESS ALERTS & INSIGHTS")

        if primary_num_col and pd.api.types.is_numeric_dtype(active_df[primary_num_col]):
            mean_p1 = active_df[primary_num_col].mean()
            st.info(f"💡 **DATA INSIGHT:** The dataset contains **{len(active_df):,}** total records with an average {primary_num_col.replace('_', ' ')} of **{mean_p1:,.2f}**.")

        if best_cat_col and not active_df.empty:
            unique_cats = active_df[best_cat_col].nunique()
            st.info(f"📌 **CATEGORY INSIGHT:** `{best_cat_col}` contains **{unique_cats}** distinct unique values.")

        st.markdown("---")

        # --- ORGANIZED TABS (PHASE 3.1) ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Dataset Preview", 
            "🧹 Data Cleaning", 
            "📊 Statistics", 
            "📈 Visualizations", 
            "💡 Key Insights",
            "🤖 AI Data Analyst"
        ])

        with tab1:
            st.subheader("Dataset Preview (First 5 Rows)")
            st.dataframe(active_df.head(5), use_container_width=True)
            
            st.subheader("Data Quality Report")
            st.dataframe(profile['quality_report'], use_container_width=True)

        with tab2:
            st.subheader("🧹 Dataset Cleaning Suite")
            cleaner = DataCleaner(active_df)

            st.markdown("#### 1. Handle Missing Values")
            missing_quality = profile['quality_report'][profile['quality_report']['Missing Values'] > 0]
            
            if not missing_quality.empty:
                st.dataframe(missing_quality, use_container_width=True)
                
                selected_col = st.selectbox("Select column to fix missing values", missing_quality.index.tolist(), key="missing_col_sel")
                strategy = st.selectbox(
                    "Choose cleaning strategy",
                    ["-- Select Strategy --", "drop", "mean", "median", "mode"],
                    key="missing_strat_sel"
                )

                if st.button("Apply Missing Value Strategy"):
                    if strategy != "-- Select Strategy --":
                        st.session_state.cleaned_df = cleaner.handle_missing_values(selected_col, strategy)
                        st.success(f"Successfully applied '{strategy}' to column '{selected_col}'!")
                        st.rerun()
                    else:
                        st.warning("Please choose a valid strategy.")
            else:
                st.success("No missing values found in this dataset!")

            st.markdown("---")

            st.markdown("#### 2. Remove Duplicate Rows")
            st.write(f"Current duplicate rows: **{profile['duplicates']}**")
            if profile['duplicates'] > 0:
                if st.button("Remove Duplicates"):
                    st.session_state.cleaned_df = cleaner.remove_duplicates()
                    st.success("Duplicates successfully removed!")
                    st.rerun()
            else:
                st.info("No duplicates found.")

            st.markdown("---")

            st.markdown("#### 3. Convert Data Types")
            conv_col = st.selectbox("Select column to convert", active_df.columns.tolist(), key="conv_col")
            current_type = str(active_df[conv_col].dtype)
            st.write(f"Current data type for `{conv_col}`: `{current_type}`")

            target_type = st.selectbox("Select target type", ["-- Select Type --", "numeric", "datetime", "category"], key="target_type_sel")
            
            if st.button("Convert Data Type"):
                if target_type != "-- Select Type --":
                    st.session_state.cleaned_df = cleaner.convert_data_types(conv_col, target_type)
                    st.success(f"Converted column `{conv_col}` to `{target_type}`!")
                    st.rerun()
                else:
                    st.warning("Please select a target type.")

            st.markdown("---")

            st.markdown("#### 👁️ Show Cleaned Dataset (Preview)")
            st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)

            csv_data = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Cleaned CSV",
                data=csv_data,
                file_name=f"cleaned_{dataset_name}",
                mime="text/csv"
            )

        with tab3:
            st.subheader("Automated Statistical Analysis")
            analyzer = StatisticalAnalyzer(active_df)
            
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

        with tab4:
            st.subheader("📉 Automated Visualization Suite")
            sns.set_theme(style="whitegrid")
            
            num_cols = active_df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = active_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            datetime_cols = active_df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()

            chart_type = st.selectbox(
                "Select Visualization Type",
                ["Histogram (Distribution)", "Boxplot (Outliers)", "Bar Chart (Categorical)", "Scatter Plot (Numerical vs Numerical)", "Correlation Heatmap", "Line Chart (Time Series / Sequence)"]
            )

            st.markdown("---")

            if chart_type == "Histogram (Distribution)":
                if num_cols:
                    col_choice = st.selectbox("Select Numerical Column", num_cols, key="hist_col")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(active_df[col_choice].dropna(), kde=True, ax=ax, color='royalblue')
                    ax.set_title(f"Histogram & Distribution of {col_choice}")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available.")

            elif chart_type == "Boxplot (Outliers)":
                if num_cols:
                    col_choice = st.selectbox("Select Numerical Column", num_cols, key="box_col")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.boxplot(x=active_df[col_choice].dropna(), ax=ax, color='orange')
                    ax.set_title(f"Boxplot of {col_choice}")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available.")

            elif chart_type == "Bar Chart (Categorical)":
                if cat_cols:
                    col_choice = st.selectbox("Select Categorical Column", cat_cols, key="bar_col")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    top_cats = active_df[col_choice].value_counts().head(10).index
                    sns.countplot(data=active_df[active_df[col_choice].isin(top_cats)], x=col_choice, order=top_cats, ax=ax, palette="viridis")
                    ax.set_title(f"Top Categories in {col_choice}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("No categorical columns available.")

            elif chart_type == "Scatter Plot (Numerical vs Numerical)":
                if len(num_cols) >= 2:
                    col_x = st.selectbox("Select X-axis Column", num_cols, key="scatter_x")
                    col_y = st.selectbox("Select Y-axis Column", num_cols, key="scatter_y")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.scatterplot(data=active_df, x=col_x, y=col_y, ax=ax, alpha=0.7, color='purple')
                    ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
                    st.pyplot(fig)
                else:
                    st.warning("At least 2 numerical columns are required for a scatter plot.")

            elif chart_type == "Correlation Heatmap":
                if len(num_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    corr = active_df[num_cols].corr()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title("Numerical Correlation Heatmap")
                    st.pyplot(fig)
                else:
                    st.warning("At least 2 numerical columns are required to generate a correlation heatmap.")

            elif chart_type == "Line Chart (Time Series / Sequence)":
                if datetime_cols and num_cols:
                    dt_col = st.selectbox("Select Date/Time Column", datetime_cols, key="line_dt")
                    num_col = st.selectbox("Select Numerical Metric", num_cols, key="line_num")
                    temp_df = active_df.sort_values(by=dt_col)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=temp_df, x=dt_col, y=num_col, ax=ax, marker='o', color='green')
                    ax.set_title(f"Trend of {num_col} over {dt_col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                elif len(num_cols) >= 2:
                    st.info("No datetime column detected. Plotting numerical sequence instead.")
                    num_x = st.selectbox("Select X-axis Sequence/Index Column", num_cols, key="seq_x")
                    num_y = st.selectbox("Select Y-axis Metric Column", num_cols, key="seq_y")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=active_df, x=num_x, y=num_y, ax=ax, marker='o', color='teal')
                    ax.set_title(f"Line Chart: {num_y} vs {num_x}")
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient columns for a line chart. Ensure you have numerical columns.")

        with tab5:
            st.subheader("💡 Automated Insight Generator")
            insight_engine = InsightGenerator(active_df, profile)
            key_insights = insight_engine.generate_insights()

            if key_insights:
                for insight in key_insights:
                    st.success(f"• {insight}")
            else:
                st.info("No major anomalies or rule-based triggers found for this dataset layout.")

        # --- Tab 6: AI Data Analyst (Clearly Separated per Phase 3.1) ---
        with tab6:
            st.markdown("---")
            st.markdown("### 🤖 AI DATA ANALYST")
            st.markdown("Ask questions about your dataset in natural language.")
            st.markdown("")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            if "pending_suggestion" not in st.session_state:
                st.session_state.pending_suggestion = ""

            ai_analyst = AIAnalyst()
            query_engine = QueryEngine(active_df)
            dataset_summary = query_engine.get_dataset_summary()

            # Display chat history container
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Dynamically generated suggested questions based on schema
            st.markdown("##### 💡 Suggested Questions")
            dynamic_suggestions = get_dynamic_suggestions(active_df)
            s_cols = st.columns(3)

            for idx, suggestion in enumerate(dynamic_suggestions):
                col_target = s_cols[idx % 3]
                with col_target:
                    if st.button(suggestion, use_container_width=True, key=f"sug_{idx}"):
                        st.session_state.pending_suggestion = suggestion
                        st.rerun()

            st.markdown("---")

            # Capture pending suggestion value if set
            default_input = st.session_state.pending_suggestion
            st.session_state.pending_suggestion = "" # Reset immediately

            user_question = st.chat_input("Ask InsightIQ anything about your dataset...", key="chat_input_val")

            # Final query determination
            final_query = user_question if user_question else (default_input if default_input else None)

            if final_query and not user_question and default_input:
                st.session_state.chat_history.append({"role": "user", "content": final_query})
                with st.chat_message("user"):
                    st.markdown(final_query)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your dataset and generating visuals..."):
                        instruction = ai_analyst.interpret_question(final_query, dataset_summary)
                        raw_result = query_engine.execute_query(instruction)
                        final_answer = ai_analyst.generate_natural_answer(final_query, raw_result, instruction)
                        
                        st.markdown(final_answer)
                        render_visualization(instruction, raw_result, active_df)

                        with st.expander("🔍 View Technical Details"):
                            st.json({
                                "AI Instruction": instruction,
                                "Raw Calculation Result": raw_result
                            })

                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                st.rerun()

            elif user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your dataset and generating visuals..."):
                        instruction = ai_analyst.interpret_question(user_question, dataset_summary)
                        raw_result = query_engine.execute_query(instruction)
                        final_answer = ai_analyst.generate_natural_answer(user_question, raw_result, instruction)
                        
                        st.markdown(final_answer)
                        render_visualization(instruction, raw_result, active_df)

                        with st.expander("🔍 View Technical Details"):
                            st.json({
                                "AI Instruction": instruction,
                                "Raw Calculation": raw_result
                            })

                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                st.rerun()
            
            st.markdown("---")

    else:
        # --- POLISHED EMPTY STATE (PHASE 3.1) ---
        st.info("📂 **No dataset loaded**\n\nUpload a CSV file via the sidebar to start analyzing your data with InsightIQ.")

if __name__ == "__main__":
    main()