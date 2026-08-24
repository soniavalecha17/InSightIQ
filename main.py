import os
import pandas as pd
from analytics.loader import DatasetLoader
from analytics.profiler import DatasetProfiler
from analytics.cleaner import DataCleaner
from analytics.statistics import StatisticalAnalyzer
from analytics.visualizations import DataVisualizer
from analytics.insights import InsightGenerator

def run_insightiq():
    # Path to sample dataset
    sample_path = "data/sample.csv"
    
    if not os.path.exists(sample_path):
        print(f"Please place a sample CSV file at '{sample_path}' to run InsightIQ.")
        return

    print("========================================")
    print("      INSIGHTIQ ANALYTICS ENGINE        ")
    print("========================================")

    # 1. Load Data
    loader = DatasetLoader(sample_path)
    df = loader.load_data()
    
    if df is None:
        return

    info = loader.get_basic_info()
    print(f"\n[1] DATASET OVERVIEW")
    print(f"Dataset: {info['dataset_name']}")
    print(f"Rows: {info['rows']} | Columns: {info['columns']}")

    # 2. Profile Data
    profiler = DatasetProfiler(df)
    profile = profiler.generate_profile()
    
    print(f"\n[2] DATA QUALITY REPORT")
    print(f"Duplicate Rows Detected: {profile['duplicates']}")
    print("\nColumn Quality Metrics:")
    print(profile['quality_report'])

    # 3. Clean Data Pipeline
    print(f"\n[3] DATA CLEANING PIPELINE")
    cleaner = DataCleaner(df)
    
    if profile['duplicates'] > 0:
        df = cleaner.remove_duplicates()
    
    missing_cols = profile['quality_report'][profile['quality_report']['Missing Values'] > 0].index
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df = cleaner.handle_missing_values(col, strategy='mean')
        else:
            df = cleaner.handle_missing_values(col, strategy='mode')

    # 4. Statistical Analysis
    print(f"\n[4] AUTOMATED STATISTICAL ANALYSIS")
    analyzer = StatisticalAnalyzer(df)
    
    num_stats = analyzer.compute_numerical_stats()
    if num_stats:
        print("\n--- Numerical Columns Summary ---")
        for col, metrics in num_stats.items():
            print(f"Column: {col}")
            print(f"  Mean: {metrics['mean']:.2f} | Median: {metrics['median']:.2f} | Std: {metrics['std']:.2f}")

    cat_stats = analyzer.compute_categorical_stats()
    if cat_stats:
        print("\n--- Categorical Columns Summary ---")
        for col, metrics in cat_stats.items():
            print(f"Column: {col}")
            print(f"  Unique Values: {metrics['unique_values']} | Most Frequent: {metrics['most_frequent']}")

    # 5. Visualizations
    print(f"\n[5] AUTOMATED VISUALIZATION SUITE")
    visualizer = DataVisualizer(df)
    visualizer.generate_all_visualizations()

    # 6. Basic Insight Generator
    print(f"\n[6] INSIGHT GENERATION")
    insight_engine = InsightGenerator(df, profile)
    key_insights = insight_engine.generate_insights()

    # 7. Final Report Summary
    print("\n" + "━━━━━━━━━━━━━━━━━━━━━━━━")
    print("      INSIGHTIQ REPORT")
    print("━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Dataset: {info['dataset_name']}")
    print(f"Rows: {info['rows']:,} | Columns: {info['columns']}")
    
    total_missing = profile['quality_report']['Missing Values'].sum()
    total_cells = info['rows'] * info['columns']
    missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    print(f"Missing Values: {missing_pct:.1f}%")
    print(f"Duplicate Rows: {profile['duplicates']}")
    
    print("\nKey Insights:")
    if key_insights:
        for insight in key_insights:
            print(f"• {insight}")
    else:
        print("• No notable automated insights generated.")
        
    print("━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ Phase 1 Completed Successfully! Charts saved in /outputs folder.")

if __name__ == "__main__":
    run_insightiq()