import pandas as pd

class InsightGenerator:
    def __init__(self, df: pd.DataFrame, profile: dict):
        self.df = df
        self.profile = profile
        self.insights = []

    def generate_insights(self) -> list:
        """Analyze dataset properties and compile automated textual insights."""
        
        # 1. Missing Values insight
        total_cells = self.profile['total_rows'] * self.profile['total_columns']
        if total_cells > 0:
            total_missing = self.profile['quality_report']['Missing Values'].sum()
            missing_pct = (total_missing / total_cells) * 100
            if missing_pct > 0:
                self.insights.append(f"The dataset contains {missing_pct:.1f}% overall missing values.")

        # 2. Duplicate rows insight
        if self.profile['duplicates'] > 0:
            self.insights.append(f"Identified and flagged {self.profile['duplicates']} duplicate rows.")

        # 3. Categorical Insights (Top frequent values)
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        for col in cat_cols:
            series = self.df[col].dropna()
            top_val = series.mode()
            if not top_val.empty:
                freq = series.value_counts().iloc[0]
                pct = (freq / len(series)) * 100
                self.insights.append(f"'{col}' is dominated by '{top_val[0]}', accounting for {pct:.1f}% of the records.")

        # 4. Numerical Insights (Correlations)
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) >= 2:
            corr_matrix = self.df[num_cols].corr().abs()
            # Find strongest correlation pair (excluding diagonal 1.0)
            unstacked = corr_matrix.unstack()
            filtered = unstacked[unstacked < 1.0].sort_values(ascending=False)
            if not filtered.empty:
                strongest_pair = filtered.index[0]
                strongest_val = filtered.iloc[0]
                if strongest_val > 0.7:
                    self.insights.append(f"Strong correlation detected between '{strongest_pair[0]}' and '{strongest_pair[1]}' (r = {strongest_val:.2f}).")

        return self.insights