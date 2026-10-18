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
            "operation_type": "mean | sum | max | min | median | count | unique_count | highest_group | lowest_group | highest_group_mean | lowest_group_mean | top_n | group_mean | group_sum | group_count | unknown",
            "target_col": "Exact dataset column name or empty string",
            "group_col": "Exact dataset grouping column or empty string",
            "filters": [
                {
                    "column": "Exact column name",
                    "operator": "== | != | > | < | >= | <= | contains",
                    "value": "Threshold value as string or number"
                }
            ],
            "chart_type": "bar | line | pie | scatter | none",
            "chart_title": "Descriptive chart title",
            "explanation": "Friendly explanation or fallback message"
        }
        CRITICAL RULES:
        1. Do NOT treat descriptive nouns or domain context phrases (like "across all grocery products", "total items", "overall") as filters unless the user explicitly specifies a condition using words like "where", "filter by", or "equal to".
        2. If no explicit condition or threshold is given, leave the "filters" array completely empty (`[]`) and compute the requested metric globally across the target column.
        3. Never invent column names. Select columns only from the supplied dataset schema.
        4. Use empty string for target_col or group_col when not needed.
        """
        
        user_prompt = f"""
        Dataset Summary:
        - Columns and Types: {json.dumps(dataset_summary.get('dtypes', {}))}
        - Total Rows: {dataset_summary.get('rows', dataset_summary.get('shape', [0,0])[0])}
        - Columns List: {dataset_summary.get('columns', list(dataset_summary.get('dtypes', {}).keys()))}

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