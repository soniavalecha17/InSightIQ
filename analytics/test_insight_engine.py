import pandas as pd
from InsightEngine import InsightEngine

# Step 1: Create a sample dataset
data = {
    "Customer": ["A", "B", "C", "D", "E", "F"],
    "Category": ["Electronics", "Clothing", "Electronics",
                 "Clothing", "Food", "Food"],
    "Purchase_Amount": [1000, 500, 1500, 700, 300, 400],
    "Cancelled": ["No", "Yes", "No", "No", "Yes", "No"],
    "Rating": [5, 3, 4, 4, 2, 5]
}

df = pd.DataFrame(data)

print("\n========== SAMPLE DATASET ==========")
print(df)

# Step 2: Create the InsightEngine object
engine = InsightEngine(df)

# Step 3: Test column role detection
print("\n========== COLUMN ROLES ==========")
roles = engine.detect_column_roles()

for column, role in roles.items():
    print(f"{column}: {role}")

# Step 4: Test deterministic calculations
print("\n========== DETERMINISTIC FACTS ==========")
facts = engine.compute_deterministic_facts()

for section, values in facts.items():
    print(f"\n{section}:")
    print(values)

# Step 5: Test AI-generated insights
print("\n========== GENERATED INSIGHTS ==========")
insights = engine.generate_insight_narratives(facts)

for insight in insights:
    print(insight)