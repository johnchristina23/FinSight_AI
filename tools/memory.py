"""
memory.py
---------
Persists user-confirmed merchant → category mappings to a local JSON file.
On next upload, known mappings are applied before calling the LLM —
saving API calls and making categorization faster and more accurate over time.

Storage format (mappings.json):
{
    "WOOLWORTHS SUPERMARKET": "groceries",
    "NETFLIX SUBSCRIPTION": "entertainment",
    "UBER TRIP": "transport",
    ...
}
"""

import json
from pathlib import Path

DEFAULT_STORE_PATH = Path("data/mappings.json")


# ── Load / Save ───────────────────────────────────────────────────────────────

def load_mappings(store_path: str | Path = DEFAULT_STORE_PATH) -> dict:
    """Load saved merchant → category mappings from disk."""
    path = Path(store_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_mappings(mappings: dict,
                  store_path: str | Path = DEFAULT_STORE_PATH) -> None:
    """Persist merchant → category mappings to disk."""
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2, sort_keys=True)


def add_mapping(description: str,
                category: str,
                store_path: str | Path = DEFAULT_STORE_PATH) -> dict:
    """
    Add or update a single merchant → category mapping and save.
    Returns the full updated mappings dict.
    """
    mappings = load_mappings(store_path)
    mappings[description.strip().upper()] = category.lower()
    save_mappings(mappings, store_path)
    return mappings


def add_mappings_bulk(new_mappings: dict,
                      store_path: str | Path = DEFAULT_STORE_PATH) -> dict:
    """
    Merge a dict of {description: category} into the store and save.
    Returns the full updated mappings dict.
    """
    mappings = load_mappings(store_path)
    for desc, cat in new_mappings.items():
        mappings[desc.strip().upper()] = cat.lower()
    save_mappings(mappings, store_path)
    return mappings


def apply_mappings(df, store_path: str | Path = DEFAULT_STORE_PATH):
    """
    Apply saved mappings to a transactions DataFrame.
    Returns DataFrame with 'category' column pre-filled where known.
    """
    import pandas as pd
    mappings = load_mappings(store_path)
    if not mappings:
        return df

    df = df.copy()
    if "category" not in df.columns:
        df["category"] = None

    # Normalise descriptions to uppercase for matching
    for desc, cat in mappings.items():
        mask = df["description"].str.upper().str.strip() == desc
        df.loc[mask, "category"] = cat

    return df


def get_mapping_stats(store_path: str | Path = DEFAULT_STORE_PATH) -> dict:
    """Return summary stats about the current mapping store."""
    mappings = load_mappings(store_path)
    if not mappings:
        return {"total": 0, "by_category": {}}

    from collections import Counter
    by_cat = Counter(mappings.values())
    return {
        "total": len(mappings),
        "by_category": dict(by_cat.most_common())
    }


def clear_mappings(store_path: str | Path = DEFAULT_STORE_PATH) -> None:
    """Clear all saved mappings — use with caution."""
    path = Path(store_path)
    if path.exists():
        path.unlink()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        test_path = f.name

    try:
        # Test add and load
        add_mapping("WOOLWORTHS SUPERMARKET", "groceries", test_path)
        add_mapping("NETFLIX SUBSCRIPTION", "entertainment", test_path)
        add_mapping("UBER TRIP", "transport", test_path)
        add_mappings_bulk({
            "RENT PAYMENT": "rent",
            "COMMSEC ETF PURCHASE": "investment"
        }, test_path)

        mappings = load_mappings(test_path)
        print(f"✅ Saved {len(mappings)} mappings:")
        for k, v in mappings.items():
            print(f"   {k:<40} → {v}")

        stats = get_mapping_stats(test_path)
        print(f"\n✅ Stats: {stats}")

        # Test apply_mappings
        import pandas as pd
        df = pd.DataFrame({
            "description": ["WOOLWORTHS SUPERMARKET", "UNKNOWN MERCHANT XYZ", "UBER TRIP"],
            "amount": [-50.0, -25.0, -12.0],
            "type": ["debit", "debit", "debit"],
        })
        df_mapped = apply_mappings(df, test_path)
        print(f"\n✅ Apply mappings result:")
        print(df_mapped[["description", "category"]].to_string())

    finally:
        os.unlink(test_path)
