
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def resolve_column(self, requested):
        """Resolve column names without case or surrounding-space sensitivity."""
        if requested is None or not str(requested).strip():
            return None

        lookup = {
            str(col).strip().casefold(): col
            for col in self.df.columns
        }
        return lookup.get(str(requested).strip().casefold())

    def _numeric_series(self, df, column):
        """Convert a column to numeric values, excluding invalid entries."""
        if column is None or column not in df.columns:
            return None

        return pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

    def get_dataset_summary(self):
        return {
            "columns": list(self.df.columns),
            "dtypes": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },
            "shape": self.df.shape,
            "rows": len(self.df),
        }

    def _apply_filters(self, df, filters):
        """Apply validated filters and return the filtered DataFrame."""
        df = df.copy()

        allowed_operators = {
            "==", "!=", ">", "<", ">=", "<=", "contains"
        }

        for condition in filters:
            column = self.resolve_column(condition.get("column"))
            operator = condition.get("operator")
            value = condition.get("value")

            if column is None:
                raise ValueError(
                    f"Filter column '{condition.get('column')}' not found."
                )

            if operator not in allowed_operators:
                raise ValueError(
                    f"Unsupported filter operator: {operator}"
                )

            series = df[column]

            if operator == "contains":
                mask = series.astype("string").str.contains(
                    str(value), case=False, na=False, regex=False
                )

            elif operator in {"==", "!="}:
                if pd.api.types.is_numeric_dtype(series):
                    number = pd.to_numeric(value, errors="coerce")

                    if pd.isna(number):
                        raise ValueError(
                            f"'{value}' is not a valid numeric filter value."
                        )

                    mask = (
                        pd.to_numeric(series, errors="coerce") == number
                    )
                else:
                    left = series.astype("string").str.strip().str.casefold()
                    right = str(value).strip().casefold()
                    mask = left == right

                if operator == "!=":
                    # Missing values should not count as unequal matches.
                    mask = mask & series.notna() if operator == "!=" else mask

            else:
                number = pd.to_numeric(value, errors="coerce")
                numeric_series = pd.to_numeric(series, errors="coerce")

                if pd.isna(number):
                    raise ValueError(
                        f"'{value}' is not a valid numeric filter value."
                    )

                if operator == ">":
                    mask = numeric_series > number
                elif operator == "<":
                    mask = numeric_series < number
                elif operator == ">=":
                    mask = numeric_series >= number
                else:
                    mask = numeric_series <= number

            df = df.loc[mask.fillna(False)]

        return df

    def execute_query(self, query_instruction: dict):
        """Execute a structured analytical instruction on the dataset."""
        if not isinstance(query_instruction, dict):
            return "Invalid query instruction."

        if "error" in query_instruction:
            return query_instruction["error"]

        op = query_instruction.get("operation_type")

        if op == "unknown":
            return query_instruction.get(
                "explanation",
                "I couldn't map that question to a supported calculation."
            )

        target_requested = query_instruction.get("target_col", "")
        group_requested = query_instruction.get("group_col", "")

        target = self.resolve_column(target_requested)
        group = self.resolve_column(group_requested)

        target_required = {
            "mean", "sum", "max", "min", "median",
            "unique_count", "highest_group", "lowest_group",
            "highest_group_mean", "lowest_group_mean",
            "top_n", "group_mean", "group_sum",
        }

        group_required = {
            "highest_group", "lowest_group",
            "highest_group_mean", "lowest_group_mean",
            "group_mean", "group_sum", "group_count",
        }

        if op not in {
            "mean", "sum", "max", "min", "median",
            "count", "unique_count", "highest_group",
            "lowest_group", "highest_group_mean",
            "lowest_group_mean", "top_n", "group_mean",
            "group_sum", "group_count",
        }:
            return f"Unsupported operation: {op}"

        if op in target_required and target is None:
            return f"A valid target column is required for '{op}'."

        if op in group_required and group is None:
            return f"A valid grouping column is required for '{op}'."

        try:
            df = self._apply_filters(
                self.df, query_instruction.get("filters", [])
            )
        except (ValueError, TypeError) as exc:
            return f"Filter error: {exc}"

        if df.empty:
            return "No matching records found after applying filters."

        # Single-column numeric aggregations
        if op in {"mean", "sum", "max", "min", "median"}:
            values = self._numeric_series(df, target)

            if values is None or values.empty:
                return f"No valid numeric values found in '{target}'."

            if op == "mean":
                result = values.mean()
            elif op == "sum":
                result = values.sum()
            elif op == "max":
                result = values.max()
            elif op == "min":
                result = values.min()
            else:
                result = values.median()

            return float(result)

        # Count all rows after filtering
        if op == "count":
            return int(len(df))

        if op == "unique_count":
            return int(df[target].nunique(dropna=True))

        # Include missing group labels rather than silently dropping rows
        grouped_df = df.copy()
        grouped_df[group] = grouped_df[group].astype("object")
        grouped_df[group] = grouped_df[group].where(
            grouped_df[group].notna(), "(Missing)"
        )

        if op == "group_count":
            return grouped_df.groupby(
                group, dropna=False, sort=False
            ).size().to_dict()

        # Grouped numerical calculations
        if op in {
            "group_sum", "group_mean",
            "highest_group", "lowest_group",
            "highest_group_mean", "lowest_group_mean",
        }:
            numeric_values = pd.to_numeric(
                grouped_df[target], errors="coerce"
            )

            valid = grouped_df.loc[
                numeric_values.notna(), [group]
            ].copy()

            valid[target] = numeric_values.loc[numeric_values.notna()]

            if valid.empty:
                return f"No valid numeric values found in '{target}'."

            aggregation = (
                "mean"
                if op in {"group_mean", "highest_group_mean",
                          "lowest_group_mean"}
                else "sum"
            )

            grouped = valid.groupby(
                group, dropna=False, sort=False
            )[target]

            if aggregation == "mean":
                result = grouped.mean()
            else:
                result = grouped.sum()

            if op in {"group_mean", "group_sum"}:
                return {
                    str(key): float(value)
                    for key, value in result.items()
                }

            item = (
                result.idxmax()
                if op in {"highest_group", "highest_group_mean"}
                else result.idxmin()
            )

            return {
                "item": str(item),
                "value": float(result.loc[item]),
                "group_col": group,
                "target_col": target,
                "aggregation": aggregation,
            }

        if op == "top_n":
            n = query_instruction.get("n", 5)

            try:
                n = int(n)
            except (TypeError, ValueError):
                return "The number of top records must be an integer."

            if n < 1:
                return "The number of top records must be at least 1."

            numeric_values = pd.to_numeric(
                df[target], errors="coerce"
            )

            ranked = df.loc[numeric_values.notna()].copy()
            ranked[target] = numeric_values.loc[numeric_values.notna()]
            ranked = ranked.sort_values(
                by=target, ascending=False
            ).head(n)

            if ranked.empty:
                return f"No valid numeric values found in '{target}'."

            return ranked.to_dict(orient="records")

        return f"Unsupported operation: {op}"
