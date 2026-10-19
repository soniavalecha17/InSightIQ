import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from groq import Groq
import os

logger = logging.getLogger(__name__)

class InsightEngine:
    """
    Computes statistical facts, data quality flags, and correlations using Pandas,
    then uses Groq to synthesize them into clear, professional insight cards.
    """
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        self.df = df.copy()
        # Clean column names
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def compute_deterministic_facts(self) -> Dict[str, Any]:
        """Extracts 5 distinct raw analytical facts using Pandas."""
        facts = {}
        
        # 1. Dataset Dimensions & Quality Flag
        rows, cols = self.df.shape
        missing_cells = int(self.df.isna().sum().sum())
        duplicates = int(self.df.duplicated().sum())
        facts["quality"] = {
            "rows": rows,
            "columns": cols,
            "missing_cells": missing_cells,
            "duplicates": duplicates
        }

        # 2. Top Categorical Distribution / Extreme Group
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

        # 3. Numeric Summary & Outliers
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

            # Outlier detection using IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
            facts["outliers"] = {
                "column": primary_num,
                "count": int(len(outliers))
            }

        # 4. Strongest Correlation (if at least 2 numeric columns exist)
        if len(num_cols) >= 2:
            corr_matrix = self.df[num_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = corr_matrix.max().max()
            if not np.isnan(max_corr) and max_corr > 0.3:
                col1, col2 = corr_matrix.unstack().idxmax()
                facts["strongest_correlation"] = {
                    "col1": col1,
                    "col2": col2,
                    "value": float(self.df[col1].corr(self.df[col2]))
                }

        return facts

    def generate_insight_narratives(self, facts: Dict[str, Any]) -> List[str]:
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
                return [line.strip() for line in text.split("\n") if line.strip() and ("•" in line or line[0].isdigit())]
        except Exception as e:
            logger.exception("Failed to generate insight narrative via Groq")

        # Fallback if API fails
        return [
            f"📊 Dataset contains {facts.get('quality', {}).get('rows', 0):,} rows and {facts.get('quality', {}).get('columns', 0)} columns.",
            f"🧹 Data quality check found {facts.get('quality', {}).get('missing_cells', 0)} missing values and {facts.get('quality', {}).get('duplicates', 0)} duplicate rows."
        ]