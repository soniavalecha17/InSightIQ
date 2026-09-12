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


def _matching_column(df: pd.DataFrame, column_name):
    """Return the real dataframe column matching a name case-insensitively."""
    if not column_name:
        return None
    name = str(column_name).strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == name:
            return col
    return None


def render_visualization(instruction: dict, raw_result, df: pd.DataFrame):
    """Render a Plotly chart safely from the AI instruction and query result."""
    chart_type = instruction.get("chart_type", "none")
    chart_title = instruction.get("chart_title") or "Data Visualization"
    group_col = _matching_column(df, instruction.get("group_col"))
    target_col = _matching_column(df, instruction.get("target_col"))
    operation_type = instruction.get("operation_type")

    if chart_type == "none" or not chart_type:
        return

    try:
        # -------------------------------------------------------------
        # 1. highest_group / lowest_group returns a special dictionary:
        #    {item, value, group_col, target_col}
        # -------------------------------------------------------------
        if (
            isinstance(raw_result, dict)
            and operation_type in ["highest_group", "lowest_group"]
            and "item" in raw_result
            and "value" in raw_result
        ):
            x_name = group_col or "Group"
            y_name = target_col or "Value"
            chart_df = pd.DataFrame({
                x_name: [raw_result["item"]],
                y_name: [raw_result["value"]]
            })

            if chart_type == "bar":
                fig = px.bar(
                    chart_df,
                    x=x_name,
                    y=y_name,
                    title=chart_title,
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "pie":
                fig = px.pie(
                    chart_df,
                    names=x_name,
                    values=y_name,
                    title=chart_title
                )
                st.plotly_chart(fig, use_container_width=True)
            return

        # -------------------------------------------------------------
        # 2. Grouped dictionary results:
        #    group_count / group_mean / group_sum
        # -------------------------------------------------------------
        if isinstance(raw_result, dict) and group_col:
            # Only treat a dictionary as grouped data when it is a simple
            # key -> value mapping. This avoids plotting metadata dictionaries.
            simple_group_dict = all(
                not isinstance(v, (dict, list, tuple, set))
                for v in raw_result.values()
            )

            if simple_group_dict and raw_result:
                value_name = target_col if target_col else "Value"
                chart_df = pd.DataFrame(
                    list(raw_result.items()),
                    columns=[group_col, value_name]
                )

                if chart_type == "bar":
                    fig = px.bar(
                        chart_df,
                        x=group_col,
                        y=value_name,
                        title=chart_title,
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "pie":
                    fig = px.pie(
                        chart_df,
                        names=group_col,
                        values=value_name,
                        title=chart_title
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "line":
                    fig = px.line(
                        chart_df,
                        x=group_col,
                        y=value_name,
                        title=chart_title,
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                return

        # -------------------------------------------------------------
        # 3. top_n returns a list of dataframe records
        # -------------------------------------------------------------
        if isinstance(raw_result, list) and len(raw_result) > 0:
            chart_df = pd.DataFrame(raw_result)
            if target_col and target_col in chart_df.columns and len(chart_df.columns) >= 2:
                x_col = group_col if group_col in chart_df.columns else chart_df.columns[0]

                if chart_type == "bar":
                    fig = px.bar(
                        chart_df,
                        x=x_col,
                        y=target_col,
                        title=chart_title,
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "line":
                    fig = px.line(
                        chart_df,
                        x=x_col,
                        y=target_col,
                        title=chart_title,
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "scatter":
                    fig = px.scatter(
                        chart_df,
                        x=x_col,
                        y=target_col,
                        title=chart_title
                    )
                    st.plotly_chart(fig, use_container_width=True)
                return

        # -------------------------------------------------------------
        # 4. For a simple scalar result, do not invent a chart.
        #    A line chart for one aggregate value is misleading.
        # -------------------------------------------------------------
        if isinstance(raw_result, (int, float)):
            st.info("A chart was requested, but this question produced a single aggregate value, so no meaningful chart was rendered.")

    except Exception as e:
        st.info(f"Could not render automated visualization: {e}")


def generate_suggested_questions(df: pd.DataFrame):
    """Generate useful suggestions from the actual uploaded dataset."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    suggestions = []

    # Prefer business-looking numeric columns when available.
    preferred_numeric = next(
        (
            col for col in numeric_cols
            if any(word in str(col).lower() for word in [
                "sales", "sale", "purchase", "revenue", "amount", "price", "value", "quantity", "volume"
            ])
        ),
        numeric_cols[0] if numeric_cols else None
    )

    rating_col = next(
        (col for col in numeric_cols if "rating" in str(col).lower()),
        None
    )

    city_col = next(
        (col for col in categorical_cols if "city" in str(col).lower()),
        categorical_cols[0] if categorical_cols else None
    )

    product_col = next(
        (col for col in categorical_cols if "product" in str(col).lower()),
        None
    )

    # Current sample-style dataset gets especially natural questions.
    if preferred_numeric:
        display_numeric = str(preferred_numeric).replace("_", " ")
        suggestions.append(f"What is the total {display_numeric}?")
        suggestions.append(f"What is the average {display_numeric}?")

    if city_col:
        suggestions.append(f"How many customers are in each {city_col}?")

        if rating_col:
            suggestions.append(f"What is the average {rating_col} by {city_col}?")
        elif preferred_numeric:
            suggestions.append(f"What is the average {preferred_numeric} by {city_col}?")

        if preferred_numeric:
            suggestions.append(f"Which {city_col} has the highest total {preferred_numeric}?")

    if product_col:
        suggestions.append(f"How many unique {product_col.lower()}s are there?")
    elif categorical_cols:
        cat = categorical_cols[0]
        suggestions.append(f"How many unique {cat.lower()}s are there?")

    # Add one useful generic group question if we have a categorical column.
    if categorical_cols and preferred_numeric:
        cat = categorical_cols[0]
        question = f"Show {preferred_numeric} by {cat}."
        suggestions.append(question)

    # Remove duplicates while preserving order and limit the UI to six.
    unique_suggestions = []
    for question in suggestions:
        if question not in unique_suggestions:
            unique_suggestions.append(question)

    return unique_suggestions[:6]


def process_ai_question(question, ai_analyst, query_engine, dataset_summary, active_df):
    """Run one AI question and return everything needed for display/history."""
    instruction = ai_analyst.interpret_question(question, dataset_summary)
    raw_result = query_engine.execute_query(instruction)
    final_answer = ai_analyst.generate_natural_answer(
        question,
        raw_result,
        instruction
    )

    return instruction, raw_result, final_answer


def display_ai_result(question, instruction, raw_result, final_answer, active_df):
    """Display one AI result including its chart and technical details."""
    st.markdown(final_answer)
    render_visualization(instruction, raw_result, active_df)

    with st.expander("🔍 View Technical Details"):
        st.json({
            "AI Instruction": instruction,
            "Raw Calculation Result": raw_result
        })


def main():
    # --- SIDEBAR ---
    st.sidebar.title("INSIGHTIQ")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

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
        st.sidebar.subheader("📋 Export Analytics Report")

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
            label="📄 Download Full Report (.txt)",
            data=report_content,
            file_name=f"InsightIQ_Report_{dataset_name.split('.')[0]}.txt",
            mime="text/plain"
        )

    # --- MAIN PAGE HEADER ---
    st.title("📊 INSIGHTIQ")
    st.markdown("### **Data Intelligence Platform**")
    st.markdown("---")

    if df is not None:
        # Reset dataset-specific state when a new dataset is selected.
        if (
            "cleaned_df" not in st.session_state
            or dataset_name != st.session_state.get("current_dataset")
        ):
            st.session_state.cleaned_df = df.copy()
            st.session_state.current_dataset = dataset_name
            st.session_state.chat_history = []
            st.session_state.pending_suggestion = None

        active_df = st.session_state.cleaned_df

        # Profile Active Data
        profiler = DatasetProfiler(active_df)
        profile = profiler.generate_profile()

        rows, cols = active_df.shape
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Dataset Preview",
            "🧹 Data Cleaning",
            "📈 Statistics",
            "📉 Visualizations",
            "💡 Key Insights",
            "💬 AI Data Analyst"
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
            missing_quality = profile['quality_report'][
                profile['quality_report']['Missing Values'] > 0
            ]

            if not missing_quality.empty:
                st.dataframe(missing_quality, use_container_width=True)

                selected_col = st.selectbox(
                    "Select column to fix missing values",
                    missing_quality.index.tolist(),
                    key="missing_col_sel"
                )
                strategy = st.selectbox(
                    "Choose cleaning strategy",
                    ["-- Select Strategy --", "drop", "mean", "median", "mode"],
                    key="missing_strat_sel"
                )

                if st.button("Apply Missing Value Strategy"):
                    if strategy != "-- Select Strategy --":
                        st.session_state.cleaned_df = cleaner.handle_missing_values(
                            selected_col,
                            strategy
                        )
                        st.success(
                            f"Successfully applied '{strategy}' to column '{selected_col}'!"
                        )
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
            conv_col = st.selectbox(
                "Select column to convert",
                active_df.columns.tolist(),
                key="conv_col"
            )
            current_type = str(active_df[conv_col].dtype)
            st.write(f"Current data type for `{conv_col}`: `{current_type}`")

            target_type = st.selectbox(
                "Select target type",
                ["-- Select Type --", "numeric", "datetime", "category"],
                key="target_type_sel"
            )

            if st.button("Convert Data Type"):
                if target_type != "-- Select Type --":
                    st.session_state.cleaned_df = cleaner.convert_data_types(
                        conv_col,
                        target_type
                    )
                    st.success(f"Converted column `{conv_col}` to `{target_type}`!")
                    st.rerun()
                else:
                    st.warning("Please select a target type.")

            st.markdown("---")

            st.markdown("#### 👁️ Show Cleaned Dataset (Preview)")
            st.dataframe(
                st.session_state.cleaned_df.head(10),
                use_container_width=True
            )

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
                    with st.expander(
                        f"Column: {col} (Unique: {metrics['unique_values']})"
                    ):
                        st.write(f"**Most Frequent:** {metrics['most_frequent']}")
                        st.write("**Distribution Percentage:**")
                        st.json(metrics['distribution_percentage'])

        with tab4:
            st.subheader("📉 Automated Visualization Suite")
            sns.set_theme(style="whitegrid")

            num_cols = active_df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = active_df.select_dtypes(
                include=['object', 'category', 'bool']
            ).columns.tolist()
            datetime_cols = active_df.select_dtypes(
                include=['datetime64', 'datetimetz']
            ).columns.tolist()

            chart_type = st.selectbox(
                "Select Visualization Type",
                [
                    "Histogram (Distribution)",
                    "Boxplot (Outliers)",
                    "Bar Chart (Categorical)",
                    "Scatter Plot (Numerical vs Numerical)",
                    "Correlation Heatmap",
                    "Line Chart (Time Series / Sequence)"
                ]
            )

            st.markdown("---")

            if chart_type == "Histogram (Distribution)":
                if num_cols:
                    col_choice = st.selectbox(
                        "Select Numerical Column",
                        num_cols,
                        key="hist_col"
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(
                        active_df[col_choice].dropna(),
                        kde=True,
                        ax=ax,
                        color='royalblue'
                    )
                    ax.set_title(f"Histogram & Distribution of {col_choice}")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available.")

            elif chart_type == "Boxplot (Outliers)":
                if num_cols:
                    col_choice = st.selectbox(
                        "Select Numerical Column",
                        num_cols,
                        key="box_col"
                    )
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.boxplot(
                        x=active_df[col_choice].dropna(),
                        ax=ax,
                        color='orange'
                    )
                    ax.set_title(f"Boxplot of {col_choice}")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available.")

            elif chart_type == "Bar Chart (Categorical)":
                if cat_cols:
                    col_choice = st.selectbox(
                        "Select Categorical Column",
                        cat_cols,
                        key="bar_col"
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    top_cats = active_df[col_choice].value_counts().head(10).index
                    sns.countplot(
                        data=active_df[active_df[col_choice].isin(top_cats)],
                        x=col_choice,
                        order=top_cats,
                        ax=ax,
                        palette="viridis"
                    )
                    ax.set_title(f"Top Categories in {col_choice}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("No categorical columns available.")

            elif chart_type == "Scatter Plot (Numerical vs Numerical)":
                if len(num_cols) >= 2:
                    col_x = st.selectbox(
                        "Select X-axis Column",
                        num_cols,
                        key="scatter_x"
                    )
                    col_y = st.selectbox(
                        "Select Y-axis Column",
                        num_cols,
                        key="scatter_y"
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.scatterplot(
                        data=active_df,
                        x=col_x,
                        y=col_y,
                        ax=ax,
                        alpha=0.7,
                        color='purple'
                    )
                    ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
                    st.pyplot(fig)
                else:
                    st.warning("At least 2 numerical columns are required for a scatter plot.")

            elif chart_type == "Correlation Heatmap":
                if len(num_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    corr = active_df[num_cols].corr()
                    sns.heatmap(
                        corr,
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f",
                        linewidths=0.5,
                        ax=ax
                    )
                    ax.set_title("Numerical Correlation Heatmap")
                    st.pyplot(fig)
                else:
                    st.warning("At least 2 numerical columns are required to generate a correlation heatmap.")

            elif chart_type == "Line Chart (Time Series / Sequence)":
                if datetime_cols and num_cols:
                    dt_col = st.selectbox(
                        "Select Date/Time Column",
                        datetime_cols,
                        key="line_dt"
                    )
                    num_col = st.selectbox(
                        "Select Numerical Metric",
                        num_cols,
                        key="line_num"
                    )
                    temp_df = active_df.sort_values(by=dt_col)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(
                        data=temp_df,
                        x=dt_col,
                        y=num_col,
                        ax=ax,
                        marker='o',
                        color='green'
                    )
                    ax.set_title(f"Trend of {num_col} over {dt_col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                elif len(num_cols) >= 2:
                    st.info("No datetime column detected. Plotting numerical sequence instead.")
                    num_x = st.selectbox(
                        "Select X-axis Sequence/Index Column",
                        num_cols,
                        key="seq_x"
                    )
                    num_y = st.selectbox(
                        "Select Y-axis Metric Column",
                        num_cols,
                        key="seq_y"
                    )
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(
                        data=active_df,
                        x=num_x,
                        y=num_y,
                        ax=ax,
                        marker='o',
                        color='teal'
                    )
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

        # --- Tab 6: AI Data Analyst (Phase 2A & 2B) ---
        with tab6:
            st.subheader("💬 AI Data Analyst")
            st.markdown(
                "Ask questions about your active dataset in normal English. "
                "InsightIQ will query the data, provide explanations, and render dynamic charts."
            )

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            if "pending_suggestion" not in st.session_state:
                st.session_state.pending_suggestion = None

            ai_analyst = AIAnalyst()
            query_engine = QueryEngine(active_df)
            dataset_summary = query_engine.get_dataset_summary()

            # ---------------------------------------------------------
            # FIX 3: Store instruction + raw_result with chat history.
            # This allows previous charts to be recreated after reruns.
            # ---------------------------------------------------------
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    if message["role"] == "assistant":
                        stored_instruction = message.get("instruction")
                        stored_raw_result = message.get("raw_result")

                        if stored_instruction is not None and stored_raw_result is not None:
                            render_visualization(
                                stored_instruction,
                                stored_raw_result,
                                active_df
                            )

                            with st.expander("🔍 View Technical Details"):
                                st.json({
                                    "AI Instruction": stored_instruction,
                                    "Raw Calculation Result": stored_raw_result
                                })

            # ---------------------------------------------------------
            # FIX 1: Dataset-aware suggested questions.
            # ---------------------------------------------------------
            suggestions = generate_suggested_questions(active_df)

            st.markdown("##### 💡 Suggested Questions")

            if suggestions:
                suggestion_columns = st.columns(3)
                for index, suggestion in enumerate(suggestions):
                    with suggestion_columns[index % 3]:
                        if st.button(
                            suggestion,
                            key=f"suggestion_{index}_{suggestion}",
                            use_container_width=True
                        ):
                            st.session_state.pending_suggestion = suggestion
                            st.rerun()
            else:
                st.info("No automatic suggestions are available for this dataset. Try asking your own question below.")

            st.markdown("---")

            # A button click cannot directly populate st.chat_input, so a
            # pending suggestion is executed as a normal question.
            pending_question = st.session_state.pending_suggestion
            st.session_state.pending_suggestion = None

            user_question = st.chat_input(
                "Ask InsightIQ anything about your dataset (e.g., 'Show total sales by category')",
                key="chat_input_val"
            )

            final_query = user_question or pending_question

            if final_query:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": final_query
                })

                with st.chat_message("user"):
                    st.markdown(final_query)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your dataset and generating visuals..."):
                        instruction, raw_result, final_answer = process_ai_question(
                            final_query,
                            ai_analyst,
                            query_engine,
                            dataset_summary,
                            active_df
                        )

                        display_ai_result(
                            final_query,
                            instruction,
                            raw_result,
                            final_answer,
                            active_df
                        )

                # Store everything needed to recreate the result on rerun.
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "instruction": instruction,
                    "raw_result": raw_result
                })

                st.rerun()

    else:
        st.info(
            "👈 Please upload a CSV file via the sidebar or place a `sample.csv` "
            "inside your `data/` folder to get started!"
        )


if __name__ == "__main__":
    main()
