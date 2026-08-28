"""
clarification_agent.py
-----------------------
Manages the multi-turn clarification conversation with the user.

Flow:
1. Receive list of recurring 'other' transactions → ask user one by one
2. Receive list of one-off 'other' transactions → offer 'skip to other' or clarify
3. Save confirmed mappings to memory store
4. Return fully categorized DataFrame

This module is designed to work with Streamlit's session state
for managing conversation flow across rerenders.
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1" if False else "https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

def clarify_merchant(merchant_name: str, past_transactions: list) -> str:
    """
    Asks an open-weight model on OpenRouter to explain an ambiguous merchant.
    """
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful financial assistant clarifying unknown merchants."
                },
                {
                    "role": "user",
                    "content": f"I see an unknown merchant named '{merchant_name}' with context: {past_transactions}. What business type is this?"
                }
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "FinSight AI Clone",
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Unable to clarify merchant right now: {e}"
