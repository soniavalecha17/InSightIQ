import pandas as pd

class DatasetProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_profile(self) -> dict:
        """Generate a complete structural and quality profile of the dataset."""
        total_rows = len(self.df)
        
        # Missing values per column and percentages
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / total_rows) * 100
        
        quality_df = pd.DataFrame({
            "Missing Values": missing_counts,
            "Missing Percentage (%)": missing_percentages.round(2),
            "Data Type": self.df.dtypes
        })

        profile = {
            "total_rows": total_rows,
            "total_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "duplicates": self.df.duplicated().sum(),
            "quality_report": quality_df,
            "numerical_stats": self.df.describe().to_dict(),
            "unique_values": self.df.nunique().to_dict()
        }
        
        return profile