import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq

# Import your Phase 1 Analytics modules
from analytics.loader import DatasetLoader
from analytics.profiler import DatasetProfiler
from analytics.cleaner import DataCleaner
from analytics.statistics import StatisticalAnalyzer
from analytics.insights import InsightGenerator

# Phase 2A/2B Imports
from analytics.ai_analyst import AIAnalyst
from analytics.query_engine import QueryEngine

# Scikit-Learn Imports for AutoML Engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="InsightIQ | Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# PHASE 2B: INSIGHT ENGINE (Fact-backed Narratives)
# ==========================================
class InsightEngine:
    """
    Computes statistical facts, data quality flags, and correlations using Pandas,
    then uses Groq to synthesize them into clean, professional insight cards.
    """
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        self.df = df.copy()
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def compute_deterministic_facts(self) -> dict:
        """Extracts 5 distinct raw analytical facts using Pandas."""
        facts = {}
        rows, cols = self.df.shape
        missing_cells = int(self.df.isna().sum().sum())
        duplicates = int(self.df.duplicated().sum())
        
        facts["quality"] = {
            "rows": rows,
            "columns": cols,
            "missing_cells": missing_cells,
            "duplicates": duplicates
        }

        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            top_cat_col = cat_cols[0]
            vc = self.df[top_cat_col].value_counts()
            facts["top_category"] = {
                "column": top_cat_col,
                "dominant_value": str(vc.index[0]),
                "count": int(vc.iloc[0]),
                "percentage": float((vc.iloc[0] / rows) * 100) if rows > 0 else 0
            }

        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            primary_num = num_cols[0]
            series = pd.to_numeric(self.df[primary_num], errors="coerce").dropna()
            facts["numeric_summary"] = {
                "column": primary_num,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std())
            }

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
            facts["outliers"] = {
                "column": primary_num,
                "count": int(len(outliers))
            }

        if len(num_cols) >= 2:
            corr_matrix = self.df[num_cols].corr().abs()
            corr_values = corr_matrix.values.copy() # Make writable copy to avoid read-only error
            np.fill_diagonal(corr_values, 0)
            max_corr = corr_values.max()
            if not np.isnan(max_corr) and max_corr > 0.3:
                unstacked = corr_matrix.unstack()
                filtered = unstacked[unstacked < 0.999]
                if not filtered.empty:
                    col1, col2 = filtered.idxmax()
                    facts["strongest_correlation"] = {
                        "col1": col1,
                        "col2": col2,
                        "value": float(self.df[col1].corr(self.df[col2]))
                    }

        return facts

    def generate_insight_narratives(self, facts: dict) -> list:
        """Sends verified facts to Groq to generate clean, readable bullet points."""
        prompt = f"""
        You are a senior data analyst. Based strictly on the following verified computational facts extracted from a dataset, generate exactly 5 distinct, engaging, and professional insight bullet points.
        
        Verified Facts:
        {facts}

        Rules:
        - Do not modify or invent numbers. Use the exact values provided in the facts.
        - Format each insight clearly starting with an appropriate emoji and bold title.
        - Keep each insight concise and actionable.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            text = response.choices[0].message.content
            if text:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                cleaned = []
                for l in lines:
                    cleaned_line = l.lstrip("•-*0123456789. ").strip()
                    if cleaned_line:
                        cleaned.append(f"💡 {cleaned_line}")
                return cleaned[:5] if cleaned else [text]
        except Exception as e:
            logger.exception("Failed to generate insight narrative via Groq")

        return [
            f"📊 Dataset contains {facts.get('quality', {}).get('rows', 0):,} rows and {facts.get('quality', {}).get('columns', 0)} columns.",
            f"🧹 Data quality check found {facts.get('quality', {}).get('missing_cells', 0)} missing values and {facts.get('quality', {}).get('duplicates', 0)} duplicate rows."
        ]

# ==========================================
# SEMANTIC COLUMN ROLE DETECTOR ENGINE
# ==========================================
def classify_column_role(col, df):
    col_lower = col.lower().replace("_", " ").replace("-", " ").strip()
    series = df[col]
    n_total = len(df)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0

    id_keywords = ["id", "identifier", "uuid", "key", "customer no", "user no", "serial no", "passid", "passengerid", "show_id", "invoice id", "transaction id"]
    if any(kw in col_lower for kw in id_keywords) or unique_ratio >= 0.95:
        return "Identifier"

    date_keywords = ["date", "year", "time", "timestamp", "dt", "created", "posted", "dob"]
    if any(kw in col_lower for kw in date_keywords) or pd.api.types.is_datetime64_any_dtype(series):
        return "Date/year"

    target_keywords = ["survived", "churn", "default", "target", "label", "class", "status", "fraud", "success"]
    if any(kw in col_lower for kw in target_keywords):
        return "Target variable"

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        unique_vals = non_null.unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            return "Binary categorical"
    elif n_unique == 2:
        return "Binary categorical"

    if series.dtype == "object":
        sample_vals = series.dropna().astype(str)
        if len(sample_vals) > 0:
            avg_len = sample_vals.str.len().mean()
            if avg_len > 40 or (unique_ratio > 0.8 and not pd.api.types.is_numeric_dtype(series)):
                return "Text"

    if pd.api.types.is_numeric_dtype(series):
        return "Numeric measurement"

    ordinal_keywords = ["pclass", "rating", "grade", "rank", "level", "priority", "score", "star"]
    if any(kw in col_lower for kw in ordinal_keywords) or (pd.api.types.is_numeric_dtype(series) and n_unique <= 10):
        return "Ordinal coded variable"

    return "Nominal categorical"

def get_column_roles_summary(df):
    return {col: classify_column_role(col, df) for col in df.columns}

def get_filtered_numeric_cols(df):
    roles = get_column_roles_summary(df)
    return [col for col, role in roles.items() if role in ["Numeric measurement", "Ordinal coded variable"]]

def find_best_categorical(df):
    roles = get_column_roles_summary(df)
    cats = [col for col, role in roles.items() if role in ["Nominal categorical", "Ordinal coded variable"]]
    return cats[0] if cats else None

# ==========================================
# SEMANTICALLY PRECISE & REFINED KPI ENGINE
# ==========================================
def get_dynamic_kpis(df):
    kpis = []
    roles = get_column_roles_summary(df)
    
    row_count = len(df)
    rec_label = "📋 TOTAL RECORDS"
    if any("customer" in c.lower() for c in df.columns):
        rec_label = "👥 TOTAL CUSTOMERS"
    elif any("passenger" in c.lower() for c in df.columns):
        rec_label = "📋 TOTAL PASSENGERS"
    elif any("student" in c.lower() for c in df.columns):
        rec_label = "🎓 TOTAL STUDENTS"
    elif any(k in c.lower() for c in df.columns for k in ["title", "show_id"]):
        rec_label = "🎬 TOTAL TITLES"
    elif any(k in c.lower() for c in df.columns for k in ["invoice", "transaction", "sale", "ticket", "booking"]):
        rec_label = "🧾 TOTAL TRANSACTIONS"

    kpis.append({"label": rec_label, "value": f"{row_count:,}"})

    financial_col = None
    for col, role in roles.items():
        if role == "Numeric measurement":
            col_l = col.lower()
            if any(kw in col_l for kw in ["total", "sales", "revenue", "income", "amount", "fare", "price", "age", "lead_time"]):
                financial_col = col
                break

    if financial_col and pd.api.types.is_numeric_dtype(df[financial_col]):
        col_l = financial_col.lower()
        if any(kw in col_l for kw in ["total", "sales", "revenue", "income", "amount"]):
            total_sum = df[financial_col].sum()
            kpis.append({"label": f"💰 TOTAL {financial_col.replace('_', ' ').upper()}", "value": f"${total_sum:,.2f}"})
        elif "lead_time" in col_l:
            avg_val = df[financial_col].mean()
            kpis.append({"label": "⏳ AVG LEAD TIME", "value": f"{avg_val:,.1f} days"})
        elif "price" in col_l or "fare" in col_l or "unit" in col_l:
            avg_val = df[financial_col].mean()
            kpis.append({"label": f"💵 AVG {financial_col.replace('_', ' ').upper()}", "value": f"${avg_val:,.2f}"})
        else:
            avg_val = df[financial_col].mean()
            kpis.append({"label": f"📊 AVG {financial_col.replace('_', ' ').upper()}", "value": f"${avg_val:,.2f}"})

    for col, role in roles.items():
        col_l = col.lower()
        if role in ["Target variable", "Binary categorical"]:
            vc = df[col].dropna().value_counts()
            pos_vals = [v for v in vc.index if v in [1, 1.0, True, "1", "True", "Movie"]]
            if pos_vals:
                pct = (vc[pos_vals[0]] / vc.sum()) * 100
                if "surviv" in col_l:
                    kpis.append({"label": "❤️ SURVIVAL RATE", "value": f"{pct:.1f}%"})
                elif "churn" in col_l:
                    kpis.append({"label": "⚡ CHURN RATE", "value": f"{pct:.1f}%"})
                elif "cancel" in col_l:
                    kpis.append({"label": "❌ CANCELLATION RATE", "value": f"{pct:.1f}%"})
                else:
                    kpis.append({"label": f"📌 {col.replace('_', ' ').upper()} RATE", "value": f"{pct:.1f}%"})
            break
        elif any(kw in col_l for kw in ["rating", "score"]):
            mean_val = df[col].mean()
            kpis.append({"label": f"⭐ AVERAGE {col.replace('_', ' ').upper()}", "value": f"{mean_val:,.2f}"})
            break

    return kpis[:4]

def main():
    # --- SIDEBAR ---
    st.sidebar.markdown("### **INSIGHTIQ**")
    st.sidebar.markdown("AI-Powered Data Intelligence & AutoML Platform")
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

    if df is not None:
        current_df = st.session_state.get("cleaned_df", df)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("📄 **EXPORT**")
        
        profiler_temp = DatasetProfiler(current_df)
        profile_temp = profiler_temp.generate_profile()
        
        insight_engine_temp = InsightEngine(current_df)
        raw_facts_temp = insight_engine_temp.compute_deterministic_facts()
        insights_temp = insight_engine_temp.generate_insight_narratives(raw_facts_temp)
        
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
            report_content += f"{ins}\n"
            
        report_content += "\n========================================\nGenerated by InsightIQ Engine"

        st.sidebar.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"InsightIQ_Report_{dataset_name.split('.')[0]}.txt",
            mime="text/plain"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About InsightIQ**")
    st.sidebar.markdown("InsightIQ is an advanced AI-powered data intelligence platform built to streamline analytics, AutoML modeling, and automated exploration.")

    # --- MAIN PAGE HEADER ---
    st.title("📊 INSIGHTIQ")
    st.markdown("### **AI-Powered Data Intelligence & AutoML Platform**")
    st.markdown("Turn raw data into actionable insights and predictive models.")
    st.markdown("---")

    if df is not None:
        if "cleaned_df" not in st.session_state or dataset_name != st.session_state.get("current_dataset"):
            st.session_state.cleaned_df = df.copy()
            st.session_state.current_dataset = dataset_name
            st.session_state.chat_history = []
            st.session_state.pending_suggestion = ""

        active_df = st.session_state.cleaned_df

        profiler = DatasetProfiler(active_df)
        profile = profiler.generate_profile()
        
        rows, cols = active_df.shape
        total_cells = rows * cols
        total_missing = profile['quality_report']['Missing Values'].sum()
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        st.markdown(f"**Active Dataset:** `{dataset_name}`")

        # --- METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROWS", f"{rows:,}")
        col2.metric("COLUMNS", f"{cols}")
        col3.metric("MISSING", f"{missing_pct:.1f}%")
        col4.metric("DUPLICATES", f"{profile['duplicates']}")

        # --- DASHBOARD OVERVIEW ---
        st.markdown("---")
        st.markdown("### 📊 DASHBOARD OVERVIEW")

        dynamic_kpis = get_dynamic_kpis(active_df)
        if dynamic_kpis:
            kpi_columns = st.columns(len(dynamic_kpis))
            for i, kpi in enumerate(dynamic_kpis):
                with kpi_columns[i]:
                    st.metric(kpi["label"], kpi["value"])

        st.markdown("---")

        # --- ORGANIZED TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Dataset Preview", 
            "🏷️ Column Roles",
            "🧹 Data Cleaning", 
            "📊 Statistics", 
            "📈 Visualizations", 
            "💡 Key Insights",
            "🤖 AI Data Analyst",
            "⚙️ AutoML Engine"
        ])

        with tab1:
            st.subheader("Dataset Preview (First 5 Rows)")
            st.dataframe(active_df.head(5), use_container_width=True)
            st.subheader("Data Quality Report")
            st.dataframe(profile['quality_report'], use_container_width=True)

        with tab2:
            st.subheader("🏷️ Semantic Column Role Inspector")
            roles_map = get_column_roles_summary(active_df)
            roles_df = pd.DataFrame(list(roles_map.items()), columns=["Column Name", "Detected Semantic Role"])
            st.dataframe(roles_df, use_container_width=True)
            st.info("InsightIQ automatically inspects every column to assign roles (Identifier, Numeric measurement, Binary categorical, Date/year, Target variable, etc.) ensuring robust dataset-agnostic intelligence.")

        with tab3:
            st.subheader("🧹 Dataset Cleaning Suite")
            cleaner = DataCleaner(active_df)
            missing_quality = profile['quality_report'][profile['quality_report']['Missing Values'] > 0]
            
            if not missing_quality.empty:
                selected_col = st.selectbox("Select column to fix missing values", missing_quality.index.tolist())
                strategy = st.selectbox("Choose cleaning strategy", ["-- Select Strategy --", "drop", "mean", "median", "mode"])
                if st.button("Apply Missing Value Strategy"):
                    if strategy != "-- Select Strategy --":
                        st.session_state.cleaned_df = cleaner.handle_missing_values(selected_col, strategy)
                        st.success(f"Successfully applied '{strategy}' to column '{selected_col}'!")
                        st.rerun()
            else:
                st.success("No missing values found in this dataset!")

            if profile['duplicates'] > 0:
                if st.button("Remove Duplicates"):
                    st.session_state.cleaned_df = cleaner.remove_duplicates()
                    st.success("Duplicates successfully removed!")
                    st.rerun()

            csv_data = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Cleaned CSV", data=csv_data, file_name=f"cleaned_{dataset_name}", mime="text/csv")

        with tab4:
            st.subheader("Automated Statistical Analysis")
            analyzer = StatisticalAnalyzer(active_df)
            num_stats = analyzer.compute_numerical_stats()
            if num_stats:
                roles = get_column_roles_summary(active_df)
                filtered_num_stats = {k: v for k, v in num_stats.items() if roles.get(k) in ["Numeric measurement", "Ordinal coded variable"]}
                if filtered_num_stats:
                    st.dataframe(pd.DataFrame(filtered_num_stats).T, use_container_width=True)

        with tab5:
            st.subheader("📉 Automated Visualization Suite")
            num_cols = get_filtered_numeric_cols(active_df)
            if num_cols:
                col_choice = st.selectbox("Select Numerical Column for Distribution", num_cols)
                fig = px.histogram(active_df, x=col_choice, nbins=30, title=f"Distribution of {col_choice}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No continuous numeric columns available.")

        with tab6:
            st.subheader("💡 Automated Insight Generator")
            st.markdown("Fact-backed patterns and data intelligence synthesized by Groq.")
            
            with st.spinner("Computing facts and synthesizing insights..."):
                try:
                    insight_engine = InsightEngine(active_df)
                    facts = insight_engine.compute_deterministic_facts()
                    insights = insight_engine.generate_insight_narratives(facts)
                    
                    if insights:
                        for insight in insights:
                            st.info(insight)
                    else:
                        st.warning("No insights were returned. Displaying raw computed facts:")
                        st.json(facts)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    logger.exception("Insight generation failed")

        with tab7:
            st.markdown("### 🤖 AI DATA ANALYST")
            st.markdown("Ask questions about your dataset in natural language.")
            st.markdown("")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            ai_analyst = AIAnalyst()
            query_engine = QueryEngine(active_df)
            dataset_summary = query_engine.get_dataset_summary()

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("raw_result") is not None and not isinstance(message["raw_result"], str):
                        with st.expander("View verified calculation"):
                            st.json(message["raw_result"], expanded=True)
                    if message.get("chart_data") is not None:
                        st.pyplot(message["chart_data"])

            user_question = st.chat_input("Ask InsightIQ anything about your dataset...", key="chat_input_val")

            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your dataset..."):
                        try:
                            instruction = ai_analyst.interpret_question(user_question, dataset_summary)

                            if "error" in instruction:
                                final_answer = f"⚠️ **Validation Error:** {instruction['error']}"
                                st.error(final_answer)
                                raw_result = None
                                chart_fig = None
                            else:
                                raw_result = query_engine.execute_query(instruction)
                                final_answer = ai_analyst.generate_natural_answer(
                                    user_question, raw_result, dataset_summary
                                )
                                st.markdown(final_answer)

                                if raw_result is not None and not isinstance(raw_result, str):
                                    with st.expander("View verified calculation"):
                                        st.json(raw_result, expanded=True)

                                # Chart Rendering Pipeline
                                chart_fig = None
                                if instruction.get("chart_type") in ["bar", "line"] and isinstance(raw_result, (dict, list)):
                                    try:
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        
                                        if isinstance(raw_result, dict):
                                            categories = list(raw_result.keys())
                                            values = list(raw_result.values())
                                        elif isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                                            keys = list(raw_result[0].keys())
                                            categories = [str(item.get(keys[0], '')) for item in raw_result]
                                            values = [item.get(keys[1], 0) for item in raw_result]
                                        else:
                                            categories, values = [], []

                                        if categories and values:
                                            if instruction.get("chart_type") == "bar":
                                                ax.bar(categories, values, color="#4c72b0")
                                            else:
                                                ax.plot(categories, values, marker="o", color="#4c72b0")
                                                
                                            ax.set_title(instruction.get("chart_title", "Data Visualization"))
                                            plt.xticks(rotation=45, ha="right")
                                            st.pyplot(fig)
                                            chart_fig = fig
                                            plt.close(fig)
                                    except Exception as e:
                                        logger.error(f"Failed to render chart: {e}")

                        except Exception as e:
                            logger.exception("AI Analyst request failed")
                            final_answer = f"⚠️ **Execution Error:** {str(e)}"
                            st.error(final_answer)
                            raw_result = None
                            instruction = None
                            chart_fig = None

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "instruction": instruction if "instruction" in locals() else None,
                    "raw_result": raw_result if "raw_result" in locals() else None,
                    "chart_data": chart_fig if "chart_fig" in locals() else None,
                })
                st.rerun()

        with tab8:
            st.subheader("⚙️ AutoML Modeling Engine")
            st.markdown("Automatically train machine learning models (Classification, Regression, or Clustering) on your active dataset.")

            ml_task = st.selectbox("Select Machine Learning Task", ["Classification", "Regression", "Clustering"])
            roles = get_column_roles_summary(active_df)

            if ml_task in ["Classification", "Regression"]:
                valid_target_cols = [c for c, r in roles.items() if r != "Identifier" and r != "Text"]
                target_col = st.selectbox("Select Target Column (Label)", valid_target_cols)
                feature_cols = st.multiselect("Select Feature Columns", [c for c in valid_target_cols if c != target_col], default=[c for c in valid_target_cols if c != target_col][:4])

                if st.button("🚀 Train Model"):
                    if not feature_cols:
                        st.warning("Please select at least one feature column.")
                    else:
                        with st.spinner("Training model with Random Forest..."):
                            try:
                                work_df = active_df[feature_cols + [target_col]].dropna()
                                X = work_df[feature_cols]
                                y = work_df[target_col]
                                X = pd.get_dummies(X, drop_first=True)

                                if ml_task == "Classification":
                                    if y.dtype == 'object' or y.dtype.name == 'category':
                                        le = LabelEncoder()
                                        y = le.fit_transform(y)
                                    
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                    model = RandomForestClassifier(random_state=42)
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    acc = accuracy_score(y_test, y_pred)
                                    st.success(f"Model Trained Successfully! Accuracy: **{acc * 100:.2f}%**")
                                    
                                    st.markdown("#### Feature Importances")
                                    fi_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
                                    fig_fi = px.bar(fi_df.head(10), x="Importance", y="Feature", orientation='h', title="Top Feature Importances")
                                    st.plotly_chart(fig_fi, use_container_width=True)

                                elif ml_task == "Regression":
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                    model = RandomForestRegressor(random_state=42)
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)

                                    mse = mean_squared_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    st.success("Model Trained Successfully!")
                                    col_m1, col_m2 = st.columns(2)
                                    col_m1.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
                                    col_m2.metric("R² Score", f"{r2:,.2f}")

                                    st.markdown("#### Feature Importances")
                                    fi_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
                                    fig_fi = px.bar(fi_df.head(10), x="Importance", y="Feature", orientation='h', title="Top Feature Importances")
                                    st.plotly_chart(fig_fi, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error during training: {e}")

            elif ml_task == "Clustering":
                num_cols_cluster = get_filtered_numeric_cols(active_df)
                cluster_features = st.multiselect("Select Numeric Features for Clustering", num_cols_cluster, default=num_cols_cluster[:2] if len(num_cols_cluster) >= 2 else num_cols_cluster)
                n_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

                if st.button("🚀 Run Clustering"):
                    if len(cluster_features) < 2:
                        st.warning("Please select at least 2 numerical features for clustering.")
                    else:
                        with st.spinner("Running KMeans Clustering..."):
                            try:
                                cluster_df = active_df[cluster_features].dropna()
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(cluster_df)

                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_df["Cluster"] = kmeans.fit_predict(scaled_data)

                                st.success("Clustering Completed Successfully!")
                                
                                fig_cluster = px.scatter(
                                    cluster_df, 
                                    x=cluster_features[0], 
                                    y=cluster_features[1], 
                                    color=cluster_df["Cluster"].astype(str),
                                    title=f"KMeans Clustering ({cluster_features[0]} vs {cluster_features[1]})"
                                )
                                st.plotly_chart(fig_cluster, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error during clustering: {e}")

    else:
        st.info("📂 **No dataset loaded**\n\nUpload a CSV file via the sidebar to start analyzing your data and running AutoML models with InsightIQ.")

if __name__ == "__main__":
    main()