import os
import pandas as pd  # <-- Add this line
from analytics.loader import DatasetLoader
from analytics.profiler import DatasetProfiler
from analytics.cleaner import DataCleaner
from analytics.statistics import StatisticalAnalyzer

def run_phase_1_preview():
    # Path to sample dataset
    sample_path = "data/sample.csv"
    
    if not os.path.exists(sample_path):
        print(f"Please place a sample CSV file at '{sample_path}' to run the test.")
        return

    print("========================================")
    print("      INSIGHTIQ ANALYTICS ENGINE        ")
    print("========================================")

    # 1. Load Data
    loader = DatasetLoader(sample_path)
    df = loader.load_data()
    
    if df is not None:
        info = loader.get_basic_info()
        print(f"\n[1] DATASET OVERVIEW")
        print(f"Dataset: {info['dataset_name']}")
        print(f"Rows: {info['rows']} | Columns: {info['columns']}")
        print(f"\nFirst 5 rows:\n{info['preview']}")

        # 2. Profile Data
        profiler = DatasetProfiler(df)
        profile = profiler.generate_profile()
        
        print(f"\n[2] DATA QUALITY REPORT")
        print(f"Duplicate Rows Detected: {profile['duplicates']}")
        print("\nColumn Quality Metrics:")
        print(profile['quality_report'])

        # 3. Clean Data (Demonstrating cleaning operations)
        print(f"\n[3] DATA CLEANING PIPELINE")
        cleaner = DataCleaner(df)
        
        # Example 1: Remove duplicates if any exist
        if profile['duplicates'] > 0:
            df = cleaner.remove_duplicates()
        
        # Example 2: Handle missing values interactively or automatically for demo
        # (Checking if any columns have missing values)
        missing_cols = profile['quality_report'][profile['quality_report']['Missing Values'] > 0].index
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Automatically fill numerical missing values with mean for this pipeline run
                df = cleaner.handle_missing_values(col, strategy='mean')
            else:
                # Fill categorical with mode
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
                print(f"  Min: {metrics['min']} | Max: {metrics['max']}")

        cat_stats = analyzer.compute_categorical_stats()
        if cat_stats:
            print("\n--- Categorical Columns Summary ---")
            for col, metrics in cat_stats.items():
                print(f"Column: {col}")
                print(f"  Unique Values: {metrics['unique_values']} | Most Frequent: {metrics['most_frequent']}")
                print(f"  Top Distribution: {metrics['distribution_percentage']}")

        print("\n✅ Phase 1 pipeline executed up to Statistics successfully!")

if __name__ == "__main__":
    run_phase_1_preview()