import pandas as pd
from analytics.ai_analyst import AIAnalyst
from analytics.query_engine import QueryEngine

def test_pipeline():
    # 1. Create a mock DataFrame matching your sample dataset
    data = {
        "Customer_ID": [101, 102, 103, 104, 105],
        "City": ["Mumbai", "Pune", "Mumbai", "Delhi", "Pune"],
        "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone"],
        "Purchase_Amount": [45000, 25000, 15000, 50000, 30000],
        "Rating": [4.5, 4.0, 3.8, 4.9, 4.2]
    }
    df = pd.DataFrame(data)

    print("--- 1. Testing QueryEngine & AIAnalyst Initialization ---")
    query_engine = QueryEngine(df)
    analyst = AIAnalyst()
    
    summary = query_engine.get_dataset_summary()
    print("Dataset Summary Generated Successfully:\n", summary)

    # 2. Test Questions
    test_questions = [
        "What is the average purchase amount?",
        "Which city has the highest purchase amount?"
    ]

    for q in test_questions:
        print(f"\n================================")
        print(f"User Question: {q}")
        
        # Step A: AI interprets question
        instruction = analyst.interpret_question(q, summary)
        print(f"AI Instruction JSON: {instruction}")

        # Step B: Query Engine executes operation
        raw_result = query_engine.execute_query(instruction)
        print(f"Raw Pandas Result: {raw_result}")

        # Step C: AI generates final answer
        final_answer = analyst.generate_natural_answer(q, raw_result, instruction)
        print(f"AI Final Explanation:\n{final_answer}")

if __name__ == "__main__":
    test_pipeline()