import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self, column: str, strategy: str) -> pd.DataFrame:
        """
        Handle missing values for a specific column based on strategy:
        - 'drop': Drop rows with missing values in this column
        - 'mean': Fill with column mean (numerical only)
        - 'median': Fill with column median (numerical only)
        - 'mode': Fill with column mode (most frequent value)
        """
        if column not in self.df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            return self.df

        if strategy == 'drop':
            self.df = self.df.dropna(subset=[column])
            print(f"Dropped rows with missing values in '{column}'.")
            
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                val = self.df[column].mean()
                self.df[column] = self.df[column].fillna(val)
                print(f"Filled missing values in '{column}' with mean: {val:.2f}")
            else:
                print(f"Cannot use 'mean' on non-numeric column '{column}'.")
                
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                val = self.df[column].median()
                self.df[column] = self.df[column].fillna(val)
                print(f"Filled missing values in '{column}' with median: {val}")
            else:
                print(f"Cannot use 'median' on non-numeric column '{column}'.")
                
        elif strategy == 'mode':
            if not self.df[column].mode().empty:
                val = self.df[column].mode()[0]
                self.df[column] = self.df[column].fillna(val)
                print(f"Filled missing values in '{column}' with mode: {val}")
            else:
                print(f"No mode found for column '{column}'.")
                
        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows.")
        return self.df

    def convert_data_types(self, column: str, target_type: str) -> pd.DataFrame:
        """Convert a column to a target data type ('datetime', 'numeric', 'category')."""
        if column not in self.df.columns:
            return self.df
            
        try:
            if target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                print(f"Converted column '{column}' to datetime.")
            elif target_type == 'numeric':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                print(f"Converted column '{column}' to numeric.")
            elif target_type == 'category':
                self.df[column] = self.df[column].astype('category')
                print(f"Converted column '{column}' to category.")
        except Exception as e:
            print(f"Failed to convert column '{column}' to {target_type}: {e}")
            
        return self.df