"""
categorizer.py
--------------
Uses Claude API to categorize bank transactions in bulk.
Returns structured category labels for each transaction.

Categories:
    income       | salary, wages, freelance, government payments
    groceries    | supermarkets, food stores
    dining       | restaurants, cafes, takeaway, delivery
    transport    | fuel, public transport, rideshare, parking, tolls
    utilities    | electricity, gas, water, internet, phone
    rent         | rent, mortgage payments
    entertainment| streaming, subscriptions, events, hobbies
    health       | pharmacy, medical, gym, dental
    shopping     | clothing, electronics, general retail
    savings      | transfers to savings accounts
    investment   | brokerage, ETFs, crypto, managed funds
    travel       | flights, hotels, holiday expenses
    education    | courses, books, tuition
    other        | anything that doesn't fit above
"""

import os
import json
import pandas as pd
from openai import OpenAI

# Initialize client pointing to OpenRouter's free endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

CATEGORIES = [
    "income", "groceries", "dining", "transport", "rent", 
    "utilities", "entertainment", "health", "shopping", 
    "savings", "investment", "travel", "education", "other"
]

def categorize_transactions(descriptions: list[str]) -> list[dict]:
    """
    Categorizes transaction descriptions using OpenRouter's free open-source models.
    """
    if not descriptions:
        return []

    prompt = f"""
    Categorize the following bank transaction descriptions.
    Choose exactly ONE category for each from this allowed list: {CATEGORIES}.
    
    Transactions:
    {json.dumps(descriptions, indent=2)}
    
    Respond strictly with a JSON array of objects:
    [
      {{"description": "transaction_text", "category": "chosen_category"}}
    ]
    Do not include markdown ticks or any explanation outside the JSON array.
    """

    try:
        response = client.chat.completions.create(
            # Dynamic router that routes to free models supporting structured JSON output
            model="openrouter/free",
            messages=[
                {"role": "system", "content": "You are a financial parsing assistant that outputs strict JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "FinSight AI Clone",
            }
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"Error during OpenRouter transaction categorization: {e}")
        # Fallback to 'other' if the API call fails
        return [{"description": desc, "category": "other"} for desc in descriptions]


def get_uncategorized_recurring(df: pd.DataFrame, min_occurrences: int = 2) -> list[str]:
    """
    Identifies recurring transaction descriptions that are categorized as 'other'.
    Required by app.py to pass into clarification_agent.
    """
    if df.empty or 'category' not in df.columns or 'description' not in df.columns:
        return []

    # Filter transactions assigned to 'other'
    uncategorized = df[df['category'] == 'other']
    
    # Calculate occurrence frequency
    counts = uncategorized['description'].value_counts()
    
    # Return descriptions meeting the recurrence threshold
    recurring = counts[counts >= min_occurrences].index.tolist()
    return recurring
