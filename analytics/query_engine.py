import pandas as pd
from typing import Any


class QueryEngine:
    """
    Takes structured intent instructions from the AIAnalyst and executes
    safe, precise calculations on the Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dataset_summary(self) -> dict:
        """Returns metadata to give the AI context about columns and data types."""

        return {
            "rows": self.df.shape[0],
            "columns": list(self.df.columns),
            "dtypes": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },
            "sample_data": self.df.head(2).to_dict(orient="records")
        }

    def execute_query(self, query_instruction: dict) -> Any:
        """
        Interprets the AI's instruction dictionary and executes
        the corresponding Pandas operation.
        """

        op_type = query_instruction.get("operation_type")
        target_col = query_instruction.get("target_col")
        group_col = query_instruction.get("group_col")

        df = self.df

        try:

            if op_type == "mean" and target_col in df.columns:
                return float(df[target_col].mean())

            elif op_type == "sum" and target_col in df.columns:
                return float(df[target_col].sum())

            elif op_type == "max" and target_col in df.columns:
                return float(df[target_col].max())

            elif op_type == "min" and target_col in df.columns:
                return float(df[target_col].min())

            elif op_type == "count":
                return int(len(df))

            elif (
                op_type == "highest_group"
                and group_col in df.columns
                and target_col in df.columns
            ):

                grouped = df.groupby(group_col)[target_col].sum()

                top_item = grouped.idxmax()
                top_value = float(grouped.max())

                return {
                    "item": top_item,
                    "value": top_value,
                    "group_col": group_col,
                    "target_col": target_col
                }

            elif op_type == "top_n" and target_col in df.columns:

                n = 5

                top_records = df.nlargest(n, target_col)

                return top_records.to_dict(orient="records")

            elif (
                op_type == "group_mean"
                and group_col in df.columns
                and target_col in df.columns
            ):

                grouped = df.groupby(group_col)[target_col].mean().to_dict()

                return {
                    str(k): float(v)
                    for k, v in grouped.items()
                }

            else:
                return (
                    f"Could not match operation '{op_type}' "
                    f"with columns "
                    f"{{Target: {target_col}, Group: {group_col}}}."
                )

        except Exception as e:
            return f"Error executing query on DataFrame: {str(e)}"