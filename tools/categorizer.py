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

def categorize_transactions(data, *args, **kwargs):
    """
    Categorizes transaction descriptions using OpenRouter's free open-source models.
    Safely handles both Pandas DataFrames and standard Python lists.
    """
    # 1. Safely check if input is empty based on its data type
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        if data.empty:
            return data
        # Extract the 'description' column into a standard list for the LLM
        descriptions = data['description'].astype(str).tolist()
    else:
        if not data:
            return []
        descriptions = data

    # 2. Construct the strict JSON prompt
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
        
        result = json.loads(response.choices[0].message.content)
        
        # 3. Format the return value correctly (DataFrame vs List)
        if is_dataframe:
            # Create a dictionary mapping the description to the LLM's chosen category
            category_map = {item['description']: item['category'] for item in result}
            # Map the categories back to the original DataFrame
            data['category'] = data['description'].map(category_map).fillna("other")
            return data
        else:
            return result

    except Exception as e:
        print(f"Error during OpenRouter transaction categorization: {e}")
        # Fallback to 'other' if the API call fails
        if is_dataframe:
            data['category'] = "other"
            return data
        return [{"description": desc, "category": "other"} for desc in descriptions]


# --- Helper functions required by app.py ---

def get_uncategorized_recurring(df: pd.DataFrame, min_occurrences: int = 2) -> list[str]:
    """
    Identifies recurring transaction descriptions that are categorized as 'other'.
    """
    if df.empty or 'category' not in df.columns or 'description' not in df.columns:
        return []

    uncategorized = df[df['category'] == 'other']
    counts = uncategorized['description'].value_counts()
    return counts[counts >= min_occurrences].index.tolist()


def get_uncategorized_oneoffs(df: pd.DataFrame, max_occurrences: int = 1) -> list[str]:
    """
    Identifies one-off transaction descriptions that are categorized as 'other'.
    """
    if df.empty or 'category' not in df.columns or 'description' not in df.columns:
        return []

    uncategorized = df[df['category'] == 'other']
    counts = uncategorized['description'].value_counts()
    return counts[counts <= max_occurrences].index.tolist()
