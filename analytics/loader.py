import pandas as pd
import os

class DatasetLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.dataset_name = os.path.basename(file_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the CSV file into a Pandas DataFrame with error handling."""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} was not found.")
            
            self.df = pd.read_csv(self.file_path)
            print(f"Successfully loaded: {self.dataset_name}")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def get_basic_info(self) -> dict:
        """Return basic overview metrics of the dataset."""
        if self.df is None:
            return {}
        
        rows, cols = self.df.shape
        return {
            "dataset_name": self.dataset_name,
            "rows": rows,
            "columns": cols,
            "preview": self.df.head(5)
        }