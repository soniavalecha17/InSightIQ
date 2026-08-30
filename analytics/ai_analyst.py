import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class AIAnalyst:
    """
    Interfaces with Gemini using structured output schemas to interpret 
    user natural language questions and map them to structured analytical tasks, filters, and chart visualizations.
    """
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def interpret_question(self, question: str, dataset_summary: dict) -> dict:
        prompt = f"""
        You are an AI data analyst backend. 
        Here is the dataset summary:
        - Columns and Types: {json.dumps(dataset_summary['dtypes'])}
        - Total Rows: {dataset_summary['rows']}

        The user asked this question in natural language: "{question}"

        Map this question into the required JSON structure. 
        CRITICAL: If the user specifies a condition (e.g., "rating greater than 4", "where city is Mumbai", "price less than 500"), you MUST populate the "filters" array with an object containing:
        - "column": the exact column name matching the condition
        - "operator": one of ["==", "!=", ">", "<", ">=", "<=", "contains"]
        - "value": the numeric or string threshold value.

        Also recommend an appropriate visualization chart type ("bar", "line", "pie", "scatter", or "none") and a concise chart title if the query lends itself to visual representation.

        Example for "How many products have a rating greater than 4?":
        - operation_type: "count"
        - target_col: "Product" (or null)
        - filters: [{{"column": "Rating", "operator": ">", "value": "4"}}]
        - chart_type: "none"
        """

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "operation_type": {
                    "type": "STRING",
                    "enum": [
                        "mean", "sum", "max", "min", "median", "count", "unique_count",
                        "highest_group", "lowest_group", "top_n", "group_mean", "group_sum", "unknown"
                    ]
                },
                "target_col": {
                    "type": "STRING",
                    "description": "Exact column name from dataset to apply operation on"
                },
                "group_col": {
                    "type": "STRING",
                    "description": "Exact column name to group by"
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
            "required": ["operation_type", "filters", "chart_type", "explanation"]
        }

        try:
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1
                )
            )
            
            return json.loads(response.text.strip())
        
        except Exception as e:
            return {
                "operation_type": "unknown",
                "target_col": None,
                "group_col": None,
                "filters": [],
                "chart_type": "none",
                "chart_title": "",
                "explanation": f"Sorry, I encountered an error communicating with the AI model: {str(e)}"
            }

    def generate_natural_answer(self, question: str, raw_result: any, explanation_context: dict) -> str:
        prompt = f"""
        The user asked: "{question}"
        The analytical tool calculated this raw result: {json.dumps(raw_result) if isinstance(raw_result, (dict, list)) else raw_result}
        Context info: {json.dumps(explanation_context)}

        Write a concise, professional, and friendly data analyst response that presents this answer clearly and explains what it means for the business.
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"The computed result is: {raw_result}"