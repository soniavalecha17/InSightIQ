import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

class AIAnalyst:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def interpret_question(self, question: str, dataset_summary: dict) -> dict:
        prompt = f"""
        You are an AI data analyst backend. 
        Here is the dataset summary:
        - Columns and Types: {json.dumps(dataset_summary['dtypes'])}
        - Total Rows: {dataset_summary['rows']}

        The user asked this question in natural language: "{question}"

        Based on the columns available, map this question into a JSON response with these exact keys:
        1. "operation_type": choose from ["mean", "sum", "max", "min", "count", "highest_group", "top_n", "group_mean", "unknown"]
        2. "target_col": exact column name from the dataset to apply the operation on (or null if not applicable)
        3. "group_col": exact column name to group by (or null if not applicable)
        4. "explanation": a friendly, professional explanation or fallback message if the question cannot be answered.

        Return ONLY valid JSON. No extra markdown blocks or text.
        """

        try:
            # Pass empty config to avoid automatic function calling warnings
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            return json.loads(raw_text.strip())
        
        except Exception as e:
            return {
                "operation_type": "unknown",
                "target_col": None,
                "group_col": None,
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