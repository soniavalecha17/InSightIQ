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

def find_best_column(df, keywords):
    """Helper function to dynamically find a relevant column using case-insensitive keywords."""
    cols = df.columns.tolist()
    for kw in keywords:
        for col in cols:
            if kw.lower() in col.lower():
                return col
    return None

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

        # --- PHASE 3.2: DYNAMIC COLUMN & SALES DETECTION ---
        price_col = find_best_column(active_df, ["unit_price", "price", "cost"])
        qty_col = find_best_column(active_df, ["sales_volume", "quantity", "qty", "volume", "units"])
        single_sales_col = find_best_column(active_df, ["purchase_amount", "sales", "amount", "revenue", "total", "value"])

        sales_col = None
        if price_col and qty_col and pd.api.types.is_numeric_dtype(active_df[price_col]) and pd.api.types.is_numeric_dtype(active_df[qty_col]):
            active_df["_calculated_sales"] = active_df[price_col] * active_df[qty_col]
            sales_col = "_calculated_sales"
        else:
            sales_col = single_sales_col

        cust_col = find_best_column(active_df, ["customer_id", "customerid", "customer", "user_id", "userid", "client"])
        prod_col = find_best_column(active_df, ["product_name", "product", "item", "good", "article", "sku"])
        city_col = find_best_column(active_df, ["city", "location", "region", "store", "branch", "country", "state", "zone"])
        rating_col = find_best_column(active_df, ["rating", "score", "stars", "review"])
        stock_col = find_best_column(active_df, ["stock", "inventory", "quantity_in_stock", "qty"])

        total_sales = active_df[sales_col].sum() if sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]) else None
        total_customers = active_df[cust_col].nunique() if cust_col else active_df.shape[0]
        average_purchase = active_df[sales_col].mean() if sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]) else None
        unique_products = active_df[prod_col].nunique() if prod_col else active_df.select_dtypes(include=['number']).shape[1]

        # --- PHASE 3.2: DASHBOARD OVERVIEW SECTION ---
        st.markdown("---")
        st.markdown("### 📊 DASHBOARD OVERVIEW")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("💰 TOTAL SALES", f"₹{total_sales:,.0f}" if total_sales is not None else "Not available")
        kpi2.metric("👥 CUSTOMERS", f"{total_customers:,}")
        kpi3.metric("🛒 AVG PURCHASE", f"₹{average_purchase:,.1f}" if average_purchase is not None else "Not available")
        kpi4.metric("📦 PRODUCTS", f"{unique_products}")

        st.markdown("---")
        st.markdown("#### 📈 Key Business Visualizations")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if city_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]):
                city_sales = active_df.groupby(city_col)[sales_col].sum().reset_index()
                fig_city = px.bar(city_sales, x=city_col, y=sales_col, title=f"💰 Total Sales by {city_col.title()}", text_auto=True)
                st.plotly_chart(fig_city, use_container_width=True)
            elif city_col:
                city_counts = active_df[city_col].value_counts().reset_index()
                city_counts.columns = [city_col, "Count"]
                fig_city = px.bar(city_counts, x=city_col, y="Count", title=f"Records by {city_col.title()}", text_auto=True)
                st.plotly_chart(fig_city, use_container_width=True)
            else:
                st.info("Location/City dimension column not available for regional visualization.")

            if city_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]):
                city_avg = active_df.groupby(city_col)[sales_col].mean().reset_index()
                fig_avg = px.bar(city_avg, x=city_col, y=sales_col, title=f"🛒 Average Purchase by {city_col.title()}", text_auto=True)
                st.plotly_chart(fig_avg, use_container_width=True)

        with chart_col2:
            if city_col and cust_col:
                city_cust = active_df.groupby(city_col)[cust_col].nunique().reset_index()
                fig_cust = px.bar(city_cust, x=city_col, y=cust_col, title=f"👥 Customers by {city_col.title()}", text_auto=True)
                st.plotly_chart(fig_cust, use_container_width=True)
            elif city_col:
                st.info("Customer identifier column not found for customer distribution.")

            if rating_col and pd.api.types.is_numeric_dtype(active_df[rating_col]):
                rating_counts = active_df[rating_col].value_counts().sort_index(ascending=False).reset_index()
                rating_counts.columns = [rating_col, "Count"]
                fig_rating = px.bar(rating_counts, x=rating_col, y="Count", title="⭐ Rating Distribution", text_auto=True)
                st.plotly_chart(fig_rating, use_container_width=True)
            else:
                st.info("Rating column not available for rating distribution chart.")

        if prod_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]):
            prod_sales = active_df.groupby(prod_col)[sales_col].sum().reset_index()
            fig_prod = px.bar(prod_sales, x=prod_col, y=sales_col, title="📦 Sales by Product", text_auto=True)
            st.plotly_chart(fig_prod, use_container_width=True)

        # --- TOP PERFORMERS ---
        st.markdown("---")
        st.markdown("### 🏆 TOP PERFORMERS")
        tp1, tp2, tp3, tp4 = st.columns(4)

        top_city = active_df.groupby(city_col)[sales_col].sum().idxmax() if city_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]) and not active_df.empty else "N/A"
        top_product = active_df.groupby(prod_col)[sales_col].sum().idxmax() if prod_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]) and not active_df.empty else "N/A"
        highest_purchase = active_df[sales_col].max() if sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]) and not active_df.empty else None
        best_rated_prod = active_df.groupby(prod_col)[rating_col].mean().idxmax() if prod_col and rating_col and pd.api.types.is_numeric_dtype(active_df[rating_col]) and not active_df.empty else "N/A"

        tp1.metric("Top Location", str(top_city))
        tp2.metric("Top Product", str(top_product))
        tp3.metric("Highest Purchase", f"₹{highest_purchase:,.0f}" if highest_purchase is not None else "Not available")
        tp4.metric("Best Rated Product", str(best_rated_prod))

        # --- BUSINESS ALERTS ---
        st.markdown("---")
        st.markdown("### ⚠️ BUSINESS ALERTS & INSIGHTS")

        if stock_col and pd.api.types.is_numeric_dtype(active_df[stock_col]):
            low_stock_count = active_df[active_df[stock_col] < 5].shape[0]
            if low_stock_count > 0:
                st.warning(f"⚠️ **INVENTORY ALERT:** {low_stock_count} products are below reorder level (stock < 5).")
            else:
                st.success("✅ **Inventory Status:** All products have healthy stock levels.")

        if city_col and sales_col and pd.api.types.is_numeric_dtype(active_df[sales_col]):
            top_city_val = active_df.groupby(city_col)[sales_col].sum().max()
            st.info(f"💡 **KEY INSIGHT:** **{top_city}** generates the highest total purchase amount (₹{top_city_val:,.0f}).")

        if rating_col and pd.api.types.is_numeric_dtype(active_df[rating_col]):
            high_ratings = active_df[active_df[rating_col] > 4].shape[0]
            st.info(f"⭐ **CUSTOMER INSIGHT:** {high_ratings} transactions received ratings greater than 4.")

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

            # Suggested Questions UI placed ABOVE the input box
            st.markdown("##### 💡 Suggested Questions")
            s_col1, s_col2, s_col3 = st.columns(3)

            with s_col1:
                if st.button("Total sales volume?", use_container_width=True):
                    st.session_state.pending_suggestion = "What is the total sales volume?"
                    st.rerun()
                if st.button("Products with low stock?", use_container_width=True):
                    st.session_state.pending_suggestion = "What products have low stock?"
                    st.rerun()

            with s_col2:
                if st.button("Average value by category?", use_container_width=True):
                    st.session_state.pending_suggestion = "What is the average value by category?"
                    st.rerun()
                if st.button("Products expiring soon?", use_container_width=True):
                    st.session_state.pending_suggestion = "Which products are expiring soon?"
                    st.rerun()

            with s_col3:
                if st.button("Category with highest sales?", use_container_width=True):
                    st.session_state.pending_suggestion = "Which category has the highest sales?"
                    st.rerun()
                if st.button("City with highest sales?", use_container_width=True):
                    st.session_state.pending_suggestion = "Which city has the highest sales?"
                    st.rerun()

            st.markdown("---")

            # Capture pending suggestion value if set
            default_input = st.session_state.pending_suggestion
            st.session_state.pending_suggestion = "" # Reset immediately

            user_question = st.chat_input("Ask InsightIQ anything about your dataset (e.g., 'Show total sales by city')", key="chat_input_val")

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
                                "Raw Calculation Result": raw_result
                            })

                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                st.rerun()
            
            st.markdown("---")

    else:
        # --- POLISHED EMPTY STATE (PHASE 3.1) ---
        st.info("📂 **No dataset loaded**\n\nUpload a CSV file via the sidebar to start analyzing your data with InsightIQ.")

if __name__ == "__main__":
    main()