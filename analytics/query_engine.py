import pandas as pd

class QueryEngine:
    """
    Takes structured intent instructions and filters from the AIAnalyst 
    and executes safe, precise operations on the Pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dataset_summary(self) -> dict:
        return {
            "rows": self.df.shape[0],
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "sample_data": self.df.head(2).to_dict(orient="records")
        }

    def _apply_filters(self, df: pd.DataFrame, filters: list) -> pd.DataFrame:
        """Applies a sequence of condition filters to the DataFrame with case-insensitivity and type safety."""
        if not filters:
            return df
        
        filtered_df = df.copy()
        for f in filters:
            col = str(f.get("column", "")).strip()
            op = str(f.get("operator", "")).strip()
            val = f.get("value")

            # Match column name case-insensitively
            matching_cols = [c for c in filtered_df.columns if c.strip().lower() == col.lower()]
            if not matching_cols:
                continue
            target_col = matching_cols[0]

            try:
                # Coerce column and value to numeric if performing numerical comparisons
                if op in [">", "<", ">=", "<=", "==", "!="]:
                    numeric_series = pd.to_numeric(filtered_df[target_col], errors='coerce')
                    if not numeric_series.isna().all():
                        filtered_df[target_col] = numeric_series
                        try:
                            val = float(val) if '.' in str(val) else int(val)
                        except ValueError:
                            pass

                if op == "==":
                    filtered_df = filtered_df[filtered_df[target_col] == val]
                elif op == "!=":
                    filtered_df = filtered_df[filtered_df[target_col] != val]
                elif op == ">":
                    filtered_df = filtered_df[filtered_df[target_col] > val]
                elif op == "<":
                    filtered_df = filtered_df[filtered_df[target_col] < val]
                elif op == ">=":
                    filtered_df = filtered_df[filtered_df[target_col] >= val]
                elif op == "<=":
                    filtered_df = filtered_df[filtered_df[target_col] <= val]
                elif op == "contains":
                    filtered_df = filtered_df[filtered_df[target_col].astype(str).str.contains(str(val), case=False, na=False)]
            except Exception as e:
                print(f"Filter execution error on column {target_col}: {e}")
                continue
        return filtered_df

    def execute_query(self, query_instruction: dict) -> any:
        op_type = query_instruction.get("operation_type")
        target_col = query_instruction.get("target_col")
        group_col = query_instruction.get("group_col")
        filters = query_instruction.get("filters", [])

        # Match target_col and group_col case-insensitively against actual dataframe columns
        if target_col:
            matching_targets = [c for c in self.df.columns if c.strip().lower() == target_col.strip().lower()]
            if matching_targets:
                target_col = matching_targets[0]

        if group_col:
            matching_groups = [c for c in self.df.columns if c.strip().lower() == group_col.strip().lower()]
            if matching_groups:
                group_col = matching_groups[0]

        df = self._apply_filters(self.df, filters)

        if df.empty or len(df) == 0:
            return "No matching records found after applying filters."

        try:
            if op_type == "mean" and target_col in df.columns:
                return float(df[target_col].mean())

            elif op_type == "sum" and target_col in df.columns:
                return float(df[target_col].sum())

            elif op_type == "max" and target_col in df.columns:
                return float(df[target_col].max())

            elif op_type == "min" and target_col in df.columns:
                return float(df[target_col].min())

            elif op_type == "median" and target_col in df.columns:
                return float(df[target_col].median())

            elif op_type == "count":
                return int(len(df))

            elif op_type == "unique_count" and target_col in df.columns:
                return int(df[target_col].nunique())

            elif op_type == "highest_group" and group_col in df.columns and target_col in df.columns:
                grouped = df.groupby(group_col)[target_col].sum()
                top_item = grouped.idxmax()
                top_value = float(grouped.max())
                return {"item": top_item, "value": top_value, "group_col": group_col, "target_col": target_col}

            elif op_type == "lowest_group" and group_col in df.columns and target_col in df.columns:
                grouped = df.groupby(group_col)[target_col].sum()
                low_item = grouped.idxmin()
                low_value = float(grouped.min())
                return {"item": low_item, "value": low_value, "group_col": group_col, "target_col": target_col}

            elif op_type == "group_mean" and group_col in df.columns and target_col in df.columns:
                grouped = df.groupby(group_col)[target_col].mean().to_dict()
                return {str(k): float(v) for k, v in grouped.items()}

            elif op_type == "group_sum" and group_col in df.columns and target_col in df.columns:
                grouped = df.groupby(group_col)[target_col].sum().to_dict()
                return {str(k): float(v) for k, v in grouped.items()}

            elif op_type == "top_n" and target_col in df.columns:
                n = 5
                top_records = df.nlargest(n, target_col)
                return top_records.to_dict(orient="records")

            else:
                return f"Could not match operation '{op_type}' with columns {{Target: {target_col}, Group: {group_col}}}."

        except Exception as e:
            return f"Error executing filtered query on DataFrame: {str(e)}"