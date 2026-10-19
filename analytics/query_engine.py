import pandas as pd
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Deterministic Pandas-based query execution engine ensuring strict adherence 
    to structured instructions without AI calculation hallucinations.
    """
    def __init__(self, df: pd.DataFrame):
        # Strip trailing/leading spaces from column names automatically
        self.df = df.copy()
        self.df.columns = [str(c).strip() for c in self.df.columns]

    def get_dataset_summary(self) -> dict:
        categories_sample = {}
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            categories_sample[col] = self.df[col].dropna().unique().tolist()[:15]

        return {
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "categorical_sample": categories_sample
        }

    def apply_filters(self, df: pd.DataFrame, filters: list) -> pd.DataFrame:
        """Applies list of filters robustly with type conversions and safe handling."""
        filtered_df = df.copy()
        for f in filters:
            if not isinstance(f, dict):
                continue
            col = str(f.get("column", "")).strip()
            op = f.get("operator")
            val = f.get("value")

            if col not in filtered_df.columns:
                continue

            series = filtered_df[col]

            if op == "==":
                filtered_df = filtered_df[series.astype(str).str.casefold() == str(val).casefold()]
            elif op == "!=":
                filtered_df = filtered_df[series.astype(str).str.casefold() != str(val).casefold()]
            elif op in [">", "<", ">=", "<="]:
                num_series = pd.to_numeric(series, errors="coerce")
                num_val = pd.to_numeric(val, errors="coerce")
                if pd.notna(num_val):
                    if op == ">": filtered_df = filtered_df[num_series > num_val]
                    elif op == "<": filtered_df = filtered_df[num_series < num_val]
                    elif op == ">=": filtered_df = filtered_df[num_series >= num_val]
                    elif op == "<=": filtered_df = filtered_df[num_series <= num_val]
            elif op == "contains":
                filtered_df = filtered_df[series.astype(str).str.contains(str(val), case=False, na=False)]

        return filtered_df

    def apply_comparison(self, df: pd.DataFrame, comparison: dict) -> pd.DataFrame:
        """Executes column-to-column or safe fall-through comparisons."""
        left_col = str(comparison.get("left_col", "")).strip()
        operator = comparison.get("operator")
        right_col = str(comparison.get("right_col", "")).strip()

        if left_col not in df.columns:
            return df  # Return unmodified if left column isn't found

        left = pd.to_numeric(df[left_col], errors="coerce")

        # If right_col is an actual column in the dataframe:
        if right_col in df.columns:
            right = pd.to_numeric(df[right_col], errors="coerce")
            valid = left.notna() & right.notna()
            operators = {
                "<": left < right,
                "<=": left <= right,
                ">": left > right,
                ">=": left >= right,
                "==": left == right,
                "!=": left != right,
            }
            if operator in operators:
                return df.loc[valid & operators[operator]]
        else:
            # Fallback: right_col is actually a literal numeric value (e.g., "50")
            num_val = pd.to_numeric(right_col, errors="coerce")
            if pd.notna(num_val):
                operators = {
                    "<": left < num_val,
                    "<=": left <= num_val,
                    ">": left > num_val,
                    ">=": left >= num_val,
                    "==": left == num_val,
                    "!=": left != num_val,
                }
                if operator in operators:
                    return df.loc[left.notna() & operators[operator]]

        return df

    def get_extreme_row(self, df: pd.DataFrame, target_col: str, highest: bool = True) -> dict:
        """Returns the full matching row dictionary for highest/lowest value queries."""
        if target_col not in df.columns:
            return None
        
        values = pd.to_numeric(df[target_col], errors="coerce")
        if values.notna().sum() == 0:
            return None
            
        index = values.idxmax() if highest else values.idxmin()
        return df.loc[index].to_dict()

    def execute_query(self, instruction: dict) -> Any:
        """Executes the validated analytical instruction against the Pandas DataFrame."""
        op = instruction.get("operation_type")
        target = str(instruction.get("target_col", "")).strip()
        group = str(instruction.get("group_col", "")).strip()
        filters = instruction.get("filters", [])
        comparison = instruction.get("comparison")

        # 1. Apply literal filters
        work_df = self.apply_filters(self.df, filters)

        # 2. Apply comparison if populated
        if isinstance(comparison, dict) and comparison.get("left_col"):
            work_df = self.apply_comparison(work_df, comparison)

        # 3. Dispatch Operation
        if op in ["count", "count_where"]:
            return int(len(work_df))

        elif op == "mean":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            return float(pd.to_numeric(work_df[target], errors="coerce").mean())

        elif op == "sum":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            return float(pd.to_numeric(work_df[target], errors="coerce").sum())

        elif op == "max":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            extreme_row = self.get_extreme_row(work_df, target, highest=True)
            return extreme_row if extreme_row else float(pd.to_numeric(work_df[target], errors="coerce").max())

        elif op == "min":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            extreme_row = self.get_extreme_row(work_df, target, highest=False)
            return extreme_row if extreme_row else float(pd.to_numeric(work_df[target], errors="coerce").min())

        elif op == "median":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            return float(pd.to_numeric(work_df[target], errors="coerce").median())

        elif op == "unique_count":
            if target not in work_df.columns:
                return f"Error: Target column '{target}' not found."
            return int(work_df[target].nunique())

        elif op == "group_mean":
            if not group or target not in work_df.columns:
                return "Error: group_mean requires valid group and target columns."
            grouped = work_df.groupby(group)[target].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            return grouped.to_dict()

        elif op == "group_sum":
            if not group or target not in work_df.columns:
                return "Error: group_sum requires valid group and target columns."
            grouped = work_df.groupby(group)[target].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
            return grouped.to_dict()

        elif op == "group_count":
            if not group:
                return "Error: group_count requires a grouping column."
            grouped = work_df.groupby(group).size()
            return grouped.to_dict()

        elif op in ["highest_group", "lowest_group"]:
            if not group or target not in work_df.columns:
                return f"Error: {op} requires group and target columns."
            agg = work_df.groupby(group)[target].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
            if agg.empty:
                return "No data available."
            best_group = agg.idxmax() if op == "highest_group" else agg.idxmin()
            return {str(best_group): float(agg[best_group])}

        return f"Unsupported operation: {op}"