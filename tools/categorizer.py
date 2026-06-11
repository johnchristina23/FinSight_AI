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

import json
import re
import anthropic
import pandas as pd
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "income", "groceries", "dining", "transport", "utilities",
    "rent", "entertainment", "health", "shopping", "savings",
    "investment", "travel", "education", "other"
]

BATCH_SIZE = 50   # transactions per API call — keeps prompts manageable

# ── Claude client ─────────────────────────────────────────────────────────────

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment


# ── Core categorization ───────────────────────────────────────────────────────

def categorize_batch(descriptions: list[str]) -> list[str]:
    """
    Send a batch of transaction descriptions to Claude and return
    a category label for each one.

    Returns a list of category strings in the same order as input.
    Falls back to 'other' for any that can't be parsed.
    """
    if not descriptions:
        return []

    # Build numbered list for the prompt
    numbered = "\n".join(
        f"{i+1}. {desc}" for i, desc in enumerate(descriptions)
    )

    prompt = f"""You are a personal finance categorization engine.

Categorize each transaction description below into exactly one of these categories:
{', '.join(CATEGORIES)}

Rules:
- Reply ONLY with a JSON array of category strings, one per transaction, in the same order
- No explanation, no markdown, no extra text — just the JSON array
- If uncertain, use "other"
- Salary, wages, freelance payments → "income"
- Transfers to savings accounts → "savings"  
- Brokerage, ETF, share purchases → "investment"
- Supermarkets (Woolworths, Coles, Aldi, etc.) → "groceries"
- Restaurants, cafes, UberEats, DoorDash → "dining"
- Uber, Lyft, public transport, fuel → "transport"
- Netflix, Spotify, Disney+, subscriptions → "entertainment"
- Rent, mortgage → "rent"
- Electricity, gas, water, internet, phone → "utilities"

Transactions:
{numbered}

Reply with ONLY a JSON array like: ["income", "groceries", "transport", ...]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        categories = json.loads(raw)
        # Validate — ensure same length and all valid categories
        if len(categories) != len(descriptions):
            raise ValueError(f"Got {len(categories)} categories for {len(descriptions)} transactions")
        return [c if c in CATEGORIES else "other" for c in categories]

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Categorization parse error: {e}. Falling back to 'other' for batch.")
        return ["other"] * len(descriptions)


def categorize_transactions(df: pd.DataFrame,
                             known_mappings: Optional[dict] = None) -> pd.DataFrame:
    """
    Categorize all transactions in a DataFrame.

    Args:
        df:             Standard transactions DataFrame from parser.py
        known_mappings: Dict of {description: category} from user's saved mappings.
                        These are applied first before calling the LLM.

    Returns:
        DataFrame with an added 'category' column.
    """
    df = df.copy()
    df["category"] = None

    known_mappings = known_mappings or {}

    # ── Step 1: Apply known mappings first (no API call needed) ──
    if known_mappings:
        for desc, cat in known_mappings.items():
            mask = df["description"].str.upper() == desc.upper()
            df.loc[mask, "category"] = cat

    # ── Step 2: Batch-categorize remaining unclassified transactions ──
    uncategorized_mask = df["category"].isna()
    uncategorized_descs = df.loc[uncategorized_mask, "description"].tolist()

    if not uncategorized_descs:
        return df

    print(f"🤖 Categorizing {len(uncategorized_descs)} transactions with Claude...")

    all_categories = []
    for i in range(0, len(uncategorized_descs), BATCH_SIZE):
        batch = uncategorized_descs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(uncategorized_descs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} transactions)...")
        categories = categorize_batch(batch)
        all_categories.extend(categories)

    # Write results back to the correct rows
    df.loc[uncategorized_mask, "category"] = all_categories

    print(f"✅ Categorization complete.")
    return df


def get_uncategorized_recurring(df: pd.DataFrame,
                                 min_occurrences: int = 2) -> list[dict]:
    """
    Find transaction descriptions categorized as 'other' that appear
    multiple times — these are the ones to ask the user about.

    Returns list of dicts: [{description, count, total_amount, sample_dates}]
    """
    other_df = df[df["category"] == "other"].copy()
    if other_df.empty:
        return []

    grouped = other_df.groupby("description").agg(
        count=("amount", "count"),
        total_amount=("amount", "sum"),
        sample_dates=("date", lambda x: sorted(x.dt.strftime("%Y-%m-%d").tolist())[:3])
    ).reset_index()

    recurring = grouped[grouped["count"] >= min_occurrences].sort_values(
        "count", ascending=False
    )

    return recurring.to_dict("records")


def get_uncategorized_oneoffs(df: pd.DataFrame,
                               min_occurrences: int = 2) -> list[dict]:
    """
    Find transactions categorized as 'other' that appear only once —
    these get offered 'other' as default with an option to clarify.

    Returns list of dicts: [{description, amount, date}]
    """
    other_df = df[df["category"] == "other"].copy()
    if other_df.empty:
        return []

    counts = other_df["description"].value_counts()
    oneoff_descs = counts[counts < min_occurrences].index

    oneoffs = other_df[other_df["description"].isin(oneoff_descs)].copy()
    oneoffs["date"] = oneoffs["date"].dt.strftime("%Y-%m-%d")

    return oneoffs[["date", "description", "amount"]].to_dict("records")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_descriptions = [
        "SALARY DEPOSIT ACME CORP",
        "WOOLWORTHS SUPERMARKET",
        "NETFLIX SUBSCRIPTION",
        "UBER TRIP",
        "RENT PAYMENT AUTO DEBIT",
        "BPAY ELECTRICITY ORIGIN",
        "COMMSEC ETF PURCHASE",
        "TRANSFER TO SAVINGS",
        "COFFEE SHOP SINGLE ORIGIN",
        "FREELANCE PAYMENT RECEIVED",
        "AMAZON PURCHASE",
        "SPOTIFY PREMIUM",
    ]

    print("Testing categorizer with sample transactions...\n")
    categories = categorize_batch(test_descriptions)

    for desc, cat in zip(test_descriptions, categories):
        print(f"  {desc:<40} → {cat}")
