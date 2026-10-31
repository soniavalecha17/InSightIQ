import pandas as pd
import numpy as np
import logging
import os
from groq import Groq

logger = logging.getLogger(__name__)

class InsightEngine:
    """
    Dataset-independent InsightEngine implementing a strict 3-layer architecture:
    1. Deterministic Pandas calculations, advanced column role classification, and data profiling.
    2. Structured insight categories (Overview, Quality, Categorical Patterns, Numerical Analysis, Category Comparisons).
    3. Groq narrative synthesis with strict ordered heading validation and verified facts.
    """
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        self.df = df.copy()
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def normalize_binary(self, series: pd.Series) -> tuple[pd.Series, str]:
        """Robustly normalize binary columns and document the explicit mapping used."""
        non_null = series.dropna()
        if non_null.empty:
            return pd.Series(index=series.index, dtype="Int64"), "Empty column"

        mapping = {
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "cancelled": 1, "canceled": 1,
            "not cancelled": 0, "not canceled": 0,
            "active": 1, "inactive": 0,
            "pass": 1, "passed": 1,
            "fail": 0, "failed": 0
        }
        
        if pd.api.types.is_bool_dtype(series):
            return series.astype("Int64"), "Boolean (True=1, False=0)"
        if pd.api.types.is_numeric_dtype(series):
            if non_null.isin([0, 1, 0.0, 1.0]).all():
                return series.astype("Int64"), "Numeric (1 / 0)"
        
        mapped = (
            series.astype("string")
            .str.strip()
            .str.lower()
            .map(mapping)
        )
        return mapped.astype("Int64"), "Mapped text labels (1 / 0)"

    def detect_column_roles(self) -> dict:
        """Refined column role classifier using column name context and safe thresholds."""
        column_types = {}
        total_count = len(self.df)
        
        for col in self.df.columns:
            series = self.df[col]
            non_null_series = series.dropna()
            unique_count = non_null_series.nunique()
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            col_lower = col.lower()
            
            # 1. Identifier detection
            id_keywords = ["_id", "id_", "identifier", "uuid", "guid", "customer_no", "user_no", "serial"]
            is_id_name = any(kw in col_lower for kw in id_keywords)
            if is_id_name and total_count > 20:
                column_types[col] = "identifier"
                continue
            elif unique_ratio >= 0.95 and not pd.api.types.is_numeric_dtype(series) and total_count > 20:
                column_types[col] = "identifier"
                continue

            # 2. Date detection
            is_date_type = pd.api.types.is_datetime64_any_dtype(series)
            if not is_date_type and ("date" in col_lower or "time" in col_lower) and "lead_time" not in col_lower:
                try:
                    parsed = pd.to_datetime(non_null_series.head(50), errors='coerce')
                    if parsed.notna().mean() >= 0.8:
                        is_date_type = True
                except Exception:
                    is_date_type = False
            
            if is_date_type:
                column_types[col] = "date"
                continue

            # 3. Strict Binary detection via complete mapping verification
            if unique_count == 2:
                normalized, _ = self.normalize_binary(non_null_series)
                if (
                    not non_null_series.empty
                    and normalized.notna().all()
                    and normalized.nunique() == 2
                ):
                    column_types[col] = "binary"
                    continue

            # 4. Numerical vs Categorical (Heuristic keyword check for numeric categorization)
            if pd.api.types.is_numeric_dtype(series):
                categorical_keywords = [
                    "rating", "rank", "level", "code", "grade",
                    "priority", "class", "category"
                ]
                is_categorical_numeric = any(
                    keyword in col_lower for keyword in categorical_keywords
                )
                if is_categorical_numeric:
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "numerical"
            else:
                column_types[col] = "categorical"
                
        return column_types

    def select_best_categorical_column(self, column_roles: dict) -> str:
        """Select a useful categorical column for group comparisons."""
        categorical_cols = [
            col for col, role in column_roles.items()
            if role == "categorical"
        ]
        if not categorical_cols:
            return None
        
        preferred_keywords = [
            "category", "segment", "type", "status",
            "region", "department", "product", "group",
            "class", "channel"
        ]
        discouraged_keywords = [
            "customer", "user", "email", "phone",
            "address", "name", "uuid", "identifier"
        ]
        best_col = None
        best_score = float("-inf")
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique(dropna=True)
            col_lower = col.lower()
            score = 0
            
            if 2 <= unique_count <= 20:
                score += 10
            elif 21 <= unique_count <= 50:
                score += 5
            elif unique_count > 50:
                score -= 10
                
            if any(pk in col_lower for pk in preferred_keywords):
                score += 15
            if any(dk in col_lower for dk in discouraged_keywords):
                score -= 20
                
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def select_best_continuous_numerical_column(self, column_roles: dict) -> str:
        """Selects a meaningful continuous numerical column for outlier/distribution analysis, avoiding binary flags."""
        numerical_cols = [
            col for col, role in column_roles.items()
            if role == "numerical"
        ]
        if not numerical_cols:
            return None
            
        preferred_keywords = [
            "lead_time", "adr", "price", "amount", "cost", "sales", 
            "revenue", "quantity", "score", "days", "waiting"
        ]
        
        best_col = numerical_cols[0]
        best_score = float("-inf")
        
        for col in numerical_cols:
            col_lower = col.lower()
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if series.nunique() <= 2:
                continue # Skip binary-like numerical columns
                
            score = 0
            if any(pk in col_lower for pk in preferred_keywords):
                score += 20
            score += min(series.nunique(), 10) # prefer higher cardinality measurements
            
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def analyze_category_comparisons(self, column_roles: dict) -> dict:
        """
        Evaluates binary and numerical columns independently across a grouping column,
        calculating group record counts, filtered valid groups, and range differences.
        """
        primary_cat = self.select_best_categorical_column(column_roles)
        binary_cols = [col for col, ctype in column_roles.items() if ctype == "binary"]
        numerical_cols = [col for col, ctype in column_roles.items() if ctype == "numerical"]

        comparisons = {}
        if not primary_cat:
            return comparisons

        # 1. Binary Group Comparisons (e.g. cancellation rates across hotel types / market segments)
        binary_comparisons = {}
        for target_bin in binary_cols[:2]:
            norm_series, mapping_desc = self.normalize_binary(self.df[target_bin])
            temp_df = self.df.copy()
            temp_df["__norm_bin"] = norm_series
            
            grouped = temp_df.groupby(primary_cat)["__norm_bin"]
            group_rates = grouped.mean().mul(100).round(2)
            group_counts = grouped.count()
            
            combined_results = {}
            for grp, rate in group_rates.items():
                valid_cnt = int(group_counts.loc[grp])
                if valid_cnt > 0 and not pd.isna(rate):
                    combined_results[str(grp)] = {
                        "value_1_rate": f"{rate:.2f}%",
                        "valid_observation_count": valid_cnt
                    }
            
            if combined_results:
                all_rates = [
                    float(v["value_1_rate"].replace("%", ""))
                    for v in combined_results.values()
                ]
                max_diff = round(max(all_rates) - min(all_rates), 2) if all_rates else 0.0
                
                sorted_res = dict(
                    sorted(
                        combined_results.items(),
                        key=lambda item: float(
                            item[1]["value_1_rate"].replace("%", "")
                        ),
                        reverse=True
                    )[:5]
                )
                
                binary_comparisons[target_bin] = {
                    "group_column": primary_cat,
                    "target_column": target_bin,
                    "mapping_description": mapping_desc,
                    "max_rate_difference": f"{max_diff:.2f}%",
                    "results": sorted_res
                }
        
        if binary_comparisons:
            comparisons["binary_group_comparison"] = binary_comparisons

        # 2. Numerical Group Comparisons
        num_comparisons = {}
        for target_num in numerical_cols[:2]:
            grouped = self.df.groupby(primary_cat)[target_num]
            group_means = grouped.mean().round(2)
            group_counts = grouped.count()
            
            combined_nums = {}
            for grp, mean_val in group_means.items():
                valid_cnt = int(group_counts.loc[grp])
                if valid_cnt > 0 and not pd.isna(mean_val):
                    combined_nums[str(grp)] = {
                        "mean_average": f"{mean_val:.2f}",
                        "valid_observation_count": valid_cnt
                    }
            
            if combined_nums:
                all_means = [
                    float(v["mean_average"])
                    for v in combined_nums.values()
                ]
                max_diff = round(max(all_means) - min(all_means), 2) if all_means else 0.0
                
                sorted_nums = dict(
                    sorted(
                        combined_nums.items(),
                        key=lambda item: float(
                            item[1]["mean_average"]
                        ),
                        reverse=True
                    )[:5]
                )
                
                num_comparisons[target_num] = {
                    "group_column": primary_cat,
                    "max_mean_difference": f"{max_diff:.2f}",
                    "results": sorted_nums
                }
        
        if num_comparisons:
            comparisons["numerical_group_comparisons"] = num_comparisons

        return comparisons

    def compute_deterministic_facts(self) -> dict:
        """Layer 1: Deterministic calculations mapped to insight categories."""
        rows, cols = self.df.shape
        total_cells = rows * cols
        column_roles = self.detect_column_roles()

        overview = {
            "rows": rows,
            "columns": cols,
            "total_cells": total_cells
        }

        total_missing = int(self.df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
        
        column_completeness = []
        for col in self.df.columns:
            missing_cnt = int(self.df[col].isna().sum())
            missing_p = round((missing_cnt / rows) * 100, 2) if rows > 0 else 0.0
            complete_p = round(100.0 - missing_p, 2)
            column_completeness.append({
                "column": col,
                "missing_count": missing_cnt,
                "missing_percentage": f"{missing_p:.2f}%",
                "complete_percentage": f"{complete_p:.2f}%"
            })
        
        sorted_completeness = sorted(column_completeness, key=lambda x: x["missing_count"], reverse=True)
        top_missing_details = [item for item in sorted_completeness if item["missing_count"] > 0][:5]

        constant_columns = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        exact_dupes = int(self.df.duplicated().sum())

        data_quality = {
            "total_missing_cells": total_missing,
            "total_missing_percentage": f"{missing_pct:.2f}%",
            "top_missing_columns": top_missing_details,
            "constant_columns": constant_columns,
            "exact_duplicate_rows": exact_dupes
        }

        primary_cat = self.select_best_categorical_column(column_roles)
        categorical_patterns = {}
        if primary_cat:
            vc = self.df[primary_cat].value_counts(dropna=True)
            if not vc.empty:
                top_name = str(vc.index[0])
                top_count = int(vc.iloc[0])
                top_pct = round((top_count / rows) * 100, 2) if rows > 0 else 0.0
                categorical_patterns = {
                    "column": primary_cat,
                    "top_category": top_name,
                    "count": top_count,
                    "percentage": f"{top_pct:.2f}%",
                    "unique_categories": int(vc.count())
                }

        # Select a proper continuous numerical column for numerical/outlier analysis (Priority 1)
        primary_num = self.select_best_continuous_numerical_column(column_roles)
        numerical_summaries = {}
        outlier_summaries = {}
        
        if primary_num:
            series = pd.to_numeric(self.df[primary_num], errors="coerce").dropna()
            if len(series) > 0:
                mean_val = round(float(series.mean()), 2)
                median_val = round(float(series.median()), 2)
                std_value = series.std()
                std_val = (
                    round(float(std_value), 2)
                    if pd.notna(std_value)
                    else None
                )
                
                numerical_summaries[primary_num] = {
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val
                }

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    outlier_count = int(outliers.count())
                    outlier_pct = round((outlier_count / len(series)) * 100, 2)
                    outlier_summaries[primary_num] = {
                        "method": "Interquartile Range (IQR)",
                        "count": outlier_count,
                        "percentage": f"{outlier_pct:.2f}%"
                    }
                else:
                    outlier_summaries[primary_num] = {"method": "IQR", "count": 0, "percentage": "0.00%", "note": "IQR is zero"}

        # Business Insights with dynamic positive outcome definition
        binary_cols = [
            col for col, role in column_roles.items()
            if role == "binary"
        ]
        business_insights = {}
        for b_col in binary_cols:
            original_series = self.df[b_col]
            norm_series, mapping_desc = self.normalize_binary(original_series)
            norm_clean = norm_series.dropna()
            if len(norm_clean) == 0:
                continue
            pos_count = int((norm_clean == 1).sum())
            neg_count = int((norm_clean == 0).sum())
            total_valid = len(norm_clean)
            rate = round((pos_count / total_valid) * 100, 2)
            
            normalized_values = (
                original_series.dropna()
                .astype(str)
                .str.strip()
                .str.lower()
            )
            positive_definition = "1 / True / Yes"
            for val in normalized_values.unique():
                if val in [
                    "yes", "true", "1", "cancelled", "canceled",
                    "pass", "active"
                ]:
                    positive_definition = val.capitalize()
                    break

            business_insights[b_col] = {
                "mapping_description": mapping_desc,
                "positive_outcome_definition": positive_definition,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "total_valid_records": total_valid,
                "value_1_rate_percentage": f"{rate:.2f}%",
                "median": float(norm_clean.median())
            }

        category_comparisons = self.analyze_category_comparisons(column_roles)

        return {
            "dataset_overview": overview,
            "data_quality": data_quality,
            "categorical_patterns": categorical_patterns,
            "numerical_summaries": numerical_summaries,
            "outlier_summaries": outlier_summaries,
            "business_insights": business_insights,
            "category_comparisons": category_comparisons
        }

    def generate_insight_narratives(self, facts: dict) -> list:
        """Layer 3: Send verified facts to Groq with strict sequential ordered heading validation and formatting rules."""
        prompt = f"""
        You are a senior data analyst. Based strictly on the following verified computational facts, generate exactly 5 distinct professional insight cards corresponding to these exact category headers in sequential order:
        1. Dataset Overview
        2. Data Quality
        3. Categorical Patterns
        4. Numerical Analysis & Outliers
        5. Category Comparisons & Business Insights
        
        Verified Facts:
        {facts}

        Rules:
        - Strict Accuracy & Formatting: Use exact numbers provided. Format all percentages and numerical values consistently to exactly two decimal places (e.g., 37.04% instead of raw unformatted decimals). Never perform outlier assessment on binary columns; outlier assessment must target continuous numerical columns.
        - Required Ordered Prefixes: Each card must start strictly with these prefixes in this exact order:
          1) '📊 Dataset Overview — '
          2) '📊 Data Quality — '
          3) '📊 Categorical Patterns — '
          4) '📊 Numerical Analysis & Outliers — '
          5) '📊 Category Comparisons & Business Insights — '
        - Grounded & Cautious: Distinguish measured results from recommendations. Avoid claiming causation without supporting evidence. Do not invent explanations.
        """

        required_prefixes = [
            "📊 Dataset Overview —",
            "📊 Data Quality —",
            "📊 Categorical Patterns —",
            "📊 Numerical Analysis & Outliers —",
            "📊 Category Comparisons & Business Insights —"
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            text = response.choices[0].message.content
            
            if text:
                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                validated_cards = []
                current_card = None

                for line in lines:
                    line = line.lstrip("•-*0123456789. ").strip()
                    line = line.replace("**", "").replace("__", "").strip()
                    line = line.replace("`", "").strip()

                    if line.startswith("💡 "):
                        line = line[2:].strip()

                    matched_prefix = next(
                        (
                            prefix
                            for prefix in required_prefixes
                            if line.startswith(prefix)
                        ),
                        None
                    )

                    if matched_prefix:
                        if current_card:
                            validated_cards.append(current_card.strip())

                        current_card = line
                    elif current_card:
                        cleaned_line = line.replace("**", "").strip()
                        if cleaned_line:
                            current_card += " " + cleaned_line

                if current_card:
                    validated_cards.append(current_card.strip())

                if (
                    len(validated_cards) == 5
                    and all(
                        card.startswith(prefix)
                        and card[len(prefix):].strip()
                        for card, prefix in zip(
                            validated_cards,
                            required_prefixes
                        )
                    )
                ):
                    return [f"💡 {card}" for card in validated_cards]

        except Exception as e:
            logger.exception("Groq API call failed")

        # Comprehensive deterministic fallback matching the 5 verified categories in exact order
        overview = facts.get("dataset_overview", {})
        quality = facts.get("data_quality", {})
        cat = facts.get("categorical_patterns", {})
        num = facts.get("numerical_summaries", {})
        outliers = facts.get("outlier_summaries", {})
        comp = facts.get("category_comparisons", {})
        
        top_miss = quality.get('top_missing_columns', [])
        missing_cols_str = ", ".join([f"{item['column']} ({item['missing_percentage']})" for item in top_miss]) if top_miss else "None"
        
        primary_num_key = list(num.keys())[0] if num else None
        if primary_num_key:
            num_stat = num[primary_num_key]
            outlier_stat = outliers.get(primary_num_key, {})
            num_text = (
                f"Continuous variable '{primary_num_key}' has a mean of {num_stat['mean']:.2f}, "
                f"median of {num_stat['median']:.2f}, and logged {outlier_stat.get('count', 0):,} IQR outliers "
                f"({outlier_stat.get('percentage', '0.00%')})."
            )
        else:
            num_text = "No suitable continuous numerical columns available for statistical analysis."

        comp_text = "No suitable group comparisons available."
        comparison_parts = []
        
        # Binary outcome comparisons (e.g., cancellation rates across groups)
        binary_comparisons = comp.get("binary_group_comparison", {})
        for target_column, bg in binary_comparisons.items():
            res_str = ", ".join(
                f"{group}: {values['value_1_rate']}"
                for group, values in list(bg["results"].items())[:3]
            )
            comparison_parts.append(
                f"Outcome rates for '{target_column}' "
                f"across '{bg['group_column']}': {res_str}."
            )
            
        # Numerical group comparisons
        numerical_comparisons = comp.get("numerical_group_comparisons", {})
        for target_column, comparison in numerical_comparisons.items():
            res_str = ", ".join(
                f"{group}: {value['mean_average']}"
                for group, value in list(
                    comparison["results"].items()
                )[:3]
            )
            comparison_parts.append(
                f"Average '{target_column}' by "
                f"'{comparison['group_column']}': {res_str}."
            )
            
        if comparison_parts:
            comp_text = " ".join(comparison_parts)
            
        if cat:
            cat_text = (
                f"Leading category '{cat['top_category']}' in '{cat['column']}' "
                f"appears {cat['count']:,} times ({cat['percentage']} of records)."
            )
        else:
            cat_text = "No suitable categorical column was available for category analysis."  

        return [
            f"💡 📊 Dataset Overview — The dataset contains {overview.get('rows', 0):,} rows and {overview.get('columns', 0)} columns ({overview.get('total_cells', 0):,} total cells).",
            f"💡 📊 Data Quality — Found {quality.get('total_missing_cells', 0):,} missing cells ({quality.get('total_missing_percentage', '0.00%')}), {quality.get('exact_duplicate_rows', 0):,} exact duplicate rows, and constant columns: {', '.join(quality.get('constant_columns', [])) or 'None'}. Columns with missing values: {missing_cols_str}.",
            f"💡 📊 Categorical Patterns — {cat_text}",
            f"💡 📊 Numerical Analysis & Outliers — {num_text}",
            f"💡 📊 Category Comparisons & Business Insights — {comp_text} Compare sample sizes and group distributions before drawing conclusions."
        ]