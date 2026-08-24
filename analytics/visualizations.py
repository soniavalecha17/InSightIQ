import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs"):
        self.df = df
        self.output_dir = output_dir
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # Set visual style
        sns.set_theme(style="whitegrid")

    def generate_all_visualizations(self):
        """Automatically detect column types and generate appropriate charts."""
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # 1. Numerical Distributions (Histograms)
        for col in num_cols[:3]:  # Limit to first 3 to prevent clutter
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col].dropna(), kde=True, color='blue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"hist_{col}.png")
            plt.savefig(path)
            plt.close()
            print(f"Generated histogram: {path}")

        # 2. Categorical Distributions (Bar charts / Count plots)
        for col in cat_cols[:2]:  # Limit to first 2
            plt.figure(figsize=(8, 5))
            top_cats = self.df[col].value_counts().head(10).index
            sns.countplot(data=self.df[self.df[col].isin(top_cats)], x=col, order=top_cats, palette="viridis")
            plt.title(f"Frequency Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"bar_{col}.png")
            plt.savefig(path)
            plt.close()
            print(f"Generated bar chart: {path}")

        # 3. Correlation Heatmap (If at least 2 numerical columns exist)
        if len(num_cols) >= 2:
            plt.figure(figsize=(8, 6))
            corr = self.df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.savefig(path)
            plt.close()
            print(f"Generated correlation heatmap: {path}")

        # 4. Numerical vs Numerical (Scatter plot for first 2 numerical cols)
        if len(num_cols) >= 2:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], alpha=0.7)
            plt.title(f"Scatter Plot: {num_cols[0]} vs {num_cols[1]}")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"scatter_{num_cols[0]}_vs_{num_cols[1]}.png")
            plt.savefig(path)
            plt.close()
            print(f"Generated scatter plot: {path}")