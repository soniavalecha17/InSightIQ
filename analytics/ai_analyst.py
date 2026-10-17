import os
import json
import logging
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

class AIAnalyst:
    """
    Interfaces with Gemini using structured output schemas to interpret 
    user natural language questions and map them to structured analytical tasks, filters, and chart visualizations.
    """
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        # Configurable model name with fallback default
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def interpret_question(self, question: str, dataset_summary: dict) -> dict:
        prompt = f"""
        You are an AI data analyst backend. 
        Here is the dataset summary:
        - Columns and Types: {json.dumps(dataset_summary.get('dtypes', {}))}
        - Total Rows: {dataset_summary.get('rows', dataset_summary.get('shape', [0,0])[0])}
        - Columns List: {dataset_summary.get('columns', list(dataset_summary.get('dtypes', {}).keys()))}

        The user asked this question in natural language: "{question}"

        Map this question into the required JSON structure. 
        CRITICAL RULES:
        1. If the user asks for a category breakdown, distribution, or count per group (e.g., "in each city", "by product", "per category", "number of customers in each city"), you MUST set:
           - "operation_type": "group_count" (or "group_sum", "group_mean" if summing/averaging values)
           - "group_col": the exact categorical column name to group by (e.g., "City", "Product")
           - "chart_type": "bar" or "pie"
        2. If the user specifies a condition (e.g., "rating greater than 4", "where city is Mumbai", "price less than 500"), you MUST populate the "filters" array with an object containing:
           - "column": the exact column name matching the condition
           - "operator": one of ["==", "!=", ">", "<", ">=", "<=", "contains"]
           - "value": the numeric or string threshold value.
        3. For "highest average", "highest mean", or "which group has the greatest average", use "highest_group_mean".
        4. For "lowest average", "lowest mean", or "which group has the smallest average", use "lowest_group_mean".
        5. For "average X by category" or "average X for each group", use "group_mean".
        6. Use "highest_group" and "lowest_group" ONLY when the user asks for the highest or lowest total or sum.
        7. For "how many records in each category", use "group_count".
        8. Never invent column names. Select columns only from the supplied dataset schema.
        9. If the requested calculation cannot be mapped reliably, use "unknown" rather than guessing.
        10. Use an empty string for target_col or group_col when that column is not needed.

        Also recommend an appropriate visualization chart type ("bar", "line", "pie", "scatter", or "none") and a concise chart title if the query lends itself to visual representation.
        """

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "operation_type": {
                    "type": "STRING",
                    "enum": [
                        "mean", "sum", "max", "min", "median", "count", "unique_count",
                        "highest_group", "lowest_group", "highest_group_mean", "lowest_group_mean",
                        "top_n", "group_mean", "group_sum", "group_count", "unknown"
                    ]
                },
                "target_col": {
                    "type": "STRING",
                    "description": "Exact dataset column name, or empty string when the operation does not require one."
                },
                "group_col": {
                    "type": "STRING",
                    "description": "Exact dataset grouping column, or empty string when grouping is not required."
                },
                "filters": {
                    "type": "ARRAY",
                    "description": "List of filter conditions to apply before the operation.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "column": {"type": "STRING", "description": "Exact column name to filter on"},
                            "operator": {"type": "STRING", "enum": ["==", "!=", ">", "<", ">=", "<=", "contains"]},
                            "value": {"type": "STRING", "description": "Value or threshold to filter by as string or number"}
                        },
                        "required": ["column", "operator", "value"]
                    }
                },
                "chart_type": {
                    "type": "STRING",
                    "enum": ["bar", "line", "pie", "scatter", "none"],
                    "description": "Recommended chart type to visualize the results if applicable."
                },
                "chart_title": {
                    "type": "STRING",
                    "description": "Descriptive title for the recommended chart."
                },
                "explanation": {
                    "type": "STRING",
                    "description": "Friendly explanation or fallback message."
                }
            },
            "required": [
                "operation_type",
                "target_col",
                "group_col",
                "filters",
                "chart_type",
                "chart_title",
                "explanation"
            ]
        }

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )
            )

            if not response.text or not response.text.strip():
                logger.error("Gemini returned an empty response text during interpret_question.")
                return {"error": "The AI model returned an empty response. Please check API quota or model configuration."}

            instruction = json.loads(response.text.strip())
            return self.validate_instruction(instruction, dataset_summary)

        except Exception as e:
            logger.exception("Gemini request failed in interpret_question with exception")
            return {"error": f"AI service error: {str(e)}"}

    def validate_instruction(self, instruction: dict, dataset_summary: dict) -> dict:
        """Validates interpreted instructions against actual dataset columns and operation requirements."""
        valid_columns = dataset_summary.get("columns", list(dataset_summary.get("dtypes", {}).keys()))
        
        for key in ("target_col", "group_col"):
            col = instruction.get(key, "")
            if not col:
                instruction[key] = ""
            else:
                matches = [
                    c for c in valid_columns
                    if str(c).strip().casefold() == str(col).strip().casefold()
                ]
                if matches:
                    instruction[key] = matches[0]
                else:
                    return {"error": f"Invalid column: {col}"}

        allowed_operations = {
            "mean", "sum", "max", "min", "median",
            "count", "unique_count", "highest_group",
            "lowest_group", "highest_group_mean",
            "lowest_group_mean", "top_n", "group_mean",
            "group_sum", "group_count", "unknown"
        }
        op = instruction.get("operation_type")
        if op not in allowed_operations:
            return {"error": f"Unsupported analytical operation: {op}"}

        target = instruction.get("target_col", "")
        group = instruction.get("group_col", "")
        
        target_required = {
            "mean", "sum", "max", "min", "median",
            "unique_count", "highest_group", "lowest_group",
            "highest_group_mean", "lowest_group_mean",
            "top_n", "group_mean", "group_sum"
        }
        group_required = {
            "highest_group", "lowest_group",
            "highest_group_mean", "lowest_group_mean",
            "group_mean", "group_sum", "group_count"
        }

        if op in target_required and not target:
            return {"error": f"Operation '{op}' requires a target column."}
        if op in group_required and not group:
            return {"error": f"Operation '{op}' requires a grouping column."}

        if op in {"group_mean", "group_sum", "highest_group_mean", "lowest_group_mean"}:
            numeric_cols = dataset_summary.get("dtypes", {})
            if target and target in numeric_cols:
                dtype = numeric_cols[target]
                if not any(t in dtype.lower() for t in ("int", "float", "uint", "numeric", "decimal")):
                    return {"error": f"Column '{target}' must be numeric for '{op}'."}

        allowed_operators = {"==", "!=", ">", "<", ">=", "<=", "contains"}
        for f in instruction.get("filters", []):
            col = f.get("column", "")
            if f.get("operator") not in allowed_operators:
                return {"error": f"Unsupported filter operator: {f.get('operator')}"}
            
            matches = [
                c for c in valid_columns
                if str(c).strip().casefold() == str(col).strip().casefold()
            ]
            if not matches:
                return {"error": f"Invalid filter column: {col}"}
            f["column"] = matches[0]

        return instruction

    def generate_natural_answer(self, question: str, raw_result: Any, explanation_context: dict) -> str:
        """Generates a detailed, natural language explanation for verified analytical results without modifying values."""
        prompt = f"""
        You are a data analyst explaining a verified calculation.
        User question: {question}
        Verified result from the analytical engine: {json.dumps(raw_result, default=str) if isinstance(raw_result, (dict, list)) else raw_result}
        Dataset context: {json.dumps(explanation_context, default=str)}

        Rules:
        - The supplied result is the source of truth.
        - Never recalculate, alter, or invent numerical values.
        - Do not invent units, causal relationships, or business facts.
        - If the result is an error, explain that the calculation failed.
        - If the result is a dictionary, explain the returned items, categories, and values clearly.
        - If the result is a single number or count, explain what this metric represents in the context of the dataset columns and overall structure.
        - Provide a professional, informative, and detailed breakdown while directly answering the question.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if not response.text or not response.text.strip():
                return f"The computed result is: {raw_result}"
            return response.text.strip()
        except Exception as e:
            logger.exception("Gemini request failed in generate_natural_answer")
            return f"The computed result is: {raw_result}"