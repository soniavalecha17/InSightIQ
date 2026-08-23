import pandas as pd

class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_numerical_stats(self) -> dict:
        """Calculate comprehensive statistics for numerical columns."""
        num_cols = self.df.select_dtypes(include=['number']).columns
        stats = {}
        
        for col in num_cols:
            series = self.df[col].dropna()
            stats[col] = {
                "mean": series.mean(),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "std": series.std(),
                "q25": series.quantile(0.25),
                "q75": series.quantile(0.75)
            }
        return stats

    def compute_categorical_stats(self) -> dict:
        """Calculate frequency and distribution metrics for categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        stats = {}
        
        for col in cat_cols:
            series = self.df[col].dropna()
            unique_count = series.nunique()
            mode_val = series.mode()[0] if not series.mode().empty else None
            
            # Calculate frequency distribution percentages
            value_counts_pct = (series.value_counts(normalize=True) * 100).round(2).to_dict()
            
            stats[col] = {
                "unique_values": unique_count,
                "most_frequent": mode_val,
                "distribution_percentage": value_counts_pct
            }
        return stats