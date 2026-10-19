import os
import json
import logging
from typing import Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

class AIAnalyst:
    """
    Interfaces with Groq using JSON mode to interpret user natural language 
    questions and map them to structured analytical tasks, filters, and chart visualizations.
    """
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def interpret_question(self, question: str, dataset_summary: dict) -> dict:
        system_prompt = """You are an AI data analyst backend. Map user natural language questions into the required JSON structure.
        You MUST respond ONLY with a valid JSON object matching this exact schema:
        {
            "operation_type": "mean | sum | max | min | median | count | unique_count | highest_group | lowest_group | highest_group_mean | lowest_group_mean | top_n | group_mean | group_sum | group_count | count_where | unknown",
            "target_col": "Exact dataset column name or empty string",
            "group_col": "Exact dataset grouping column or empty string",
            "filters": [
                {
                    "column": "Exact column name",
                    "operator": "== | != | > | < | >= | <= | contains",
                    "value": "Literal threshold value as string or number"
                }
            ],
            "comparison": {
                "left_col": "Exact left column name",
                "operator": "< | <= | > | >= | == | !=",
                "right_col": "Exact right column name to compare against"
            },
            "chart_type": "bar | line | pie | scatter | none",
            "chart_title": "Descriptive chart title",
            "explanation": "Friendly explanation or fallback message"
        }
        CRITICAL RULES FOR FILTERS VS COMPARISONS:
        1. **Column-to-Literal Filters (`filters`)**: Use when filtering by a fixed number, string, or threshold (e.g., "price greater than 50", "rating above 4", "where status is active"). Populate the `filters` array and set `comparison` to null or empty strings.
           - Example: "How many products have a price greater than 50?" -> operation_type: "count", filters: [{"column": "Price", "operator": ">", "value": 50}], comparison: null.
        2. **Column-to-Column Comparisons (`comparison`)**: Use ONLY when comparing two different columns against each other (e.g., "Stock below Reorder_Level", "sales greater than cost"). Populate the `comparison` object and leave `filters` empty (`[]`).
        3. Do NOT treat general descriptive text ("across all products") as filters.
        4. Never invent column names. Select columns only from the supplied dataset schema.
        """

        user_prompt = f"""
        Dataset Summary:
        - Columns and Types: {json.dumps(dataset_summary.get('dtypes', {}))}
        - Total Rows: {dataset_summary.get('rows', dataset_summary.get('shape', [0,0])[0])}
        - Columns List: {dataset_summary.get('columns', list(dataset_summary.get('dtypes', {}).keys()))}
        - Categorical Samples: {json.dumps(dataset_summary.get('categorical_sample', {}))}

        User Question: "{question}"
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                logger.error("Groq returned an empty response text during interpret_question.")
                return {"error": "The AI model returned an empty response. Please check API configuration."}

            instruction = json.loads(response_text.strip())
            return self.validate_instruction(instruction, dataset_summary)

        except Exception as e:
            logger.exception("Groq request failed in interpret_question with exception")
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
            "group_sum", "group_count", "count_where", "unknown"
        }
        op = instruction.get("operation_type")
        if op not in allowed_operations:
            return {"error": f"Unsupported analytical operation: {op}"}

        # Validate comparison object if operation is count_where
        comp = instruction.get("comparison")
        if op == "count_where":
            if not isinstance(comp, dict):
                return {"error": "Operation 'count_where' requires a valid comparison object."}
            for k in ("left_col", "right_col"):
                c_col = comp.get(k, "")
                if c_col:
                    matches = [c for c in valid_columns if str(c).strip().casefold() == str(c_col).strip().casefold()]
                    if matches:
                        comp[k] = matches[0]
                    else:
                        return {"error": f"Invalid comparison column: {c_col}"}

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
        """Generates a concise, structured analytics response for verified results using Groq."""
        prompt = f"""
        You are a professional data analyst. Provide a concise, structured answer.
        User question: {question}
        Verified result from analytical engine: {json.dumps(raw_result, default=str) if isinstance(raw_result, (dict, list)) else raw_result}
        Dataset context: {json.dumps(explanation_context, default=str)}

        Formatting Requirements:
        - Keep the answer clean and concise (aim for a few bullet points or short lines).
        - Format clearly as:
          - **Answer**: [The core metric or result with proper units if applicable]
          - **Method**: [Brief description of the calculation or operation used]
          - **Dataset Scope**: [Mention total rows or relevant context]
        - Never invent or modify numerical values.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                return f"**Answer**: {raw_result}"
            return response_text.strip()
        except Exception as e:
            logger.exception("Groq request failed in generate_natural_answer")
            return f"**Answer**: {raw_result}"