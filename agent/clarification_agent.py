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

from tools.memory import add_mapping, load_mappings, CATEGORIES_LIST
from tools.categorizer import CATEGORIES

# ── Session state keys ────────────────────────────────────────────────────────

STATE_RECURRING     = "clarification_recurring"      # list of recurring unknowns
STATE_ONEOFFS       = "clarification_oneoffs"         # list of one-off unknowns
STATE_RECURRING_IDX = "clarification_recurring_idx"  # current index
STATE_ONEOFFS_IDX   = "clarification_oneoffs_idx"
STATE_CONFIRMED     = "clarification_confirmed"       # {desc: category} confirmed this session
STATE_PHASE         = "clarification_phase"           # 'recurring' | 'oneoffs' | 'done'
STATE_STORE_PATH    = "clarification_store_path"


# ── Initialise ────────────────────────────────────────────────────────────────

def init_clarification(recurring: list[dict],
                       oneoffs: list[dict],
                       store_path: str = "data/mappings.json") -> dict:
    """
    Initialise session state for a clarification session.
    Call this once when new statements are uploaded.

    Returns the initial state dict (caller stores in st.session_state).
    """
    return {
        STATE_RECURRING:     recurring,
        STATE_ONEOFFS:       oneoffs,
        STATE_RECURRING_IDX: 0,
        STATE_ONEOFFS_IDX:   0,
        STATE_CONFIRMED:     {},
        STATE_PHASE:         "recurring" if recurring else ("oneoffs" if oneoffs else "done"),
        STATE_STORE_PATH:    store_path,
    }


# ── Current question ──────────────────────────────────────────────────────────

def get_current_question(state: dict) -> dict | None:
    """
    Get the current clarification question based on session state.

    Returns dict with:
        phase:       'recurring' | 'oneoffs' | 'done'
        description: merchant name
        count:       how many times it appears (recurring only)
        total:       total amount (recurring only)
        amount:      single transaction amount (oneoffs only)
        date:        transaction date (oneoffs only)
        progress:    (current, total) tuple
    """
    phase = state.get(STATE_PHASE, "done")

    if phase == "recurring":
        items  = state[STATE_RECURRING]
        idx    = state[STATE_RECURRING_IDX]
        total  = len(items)

        if idx >= total:
            # Move to oneoffs phase
            state[STATE_PHASE] = "oneoffs" if state[STATE_ONEOFFS] else "done"
            return get_current_question(state)

        item = items[idx]
        return {
            "phase":       "recurring",
            "description": item["description"],
            "count":       item["count"],
            "total":       item["total_amount"],
            "dates":       item.get("sample_dates", []),
            "progress":    (idx + 1, total),
        }

    elif phase == "oneoffs":
        items = state[STATE_ONEOFFS]
        idx   = state[STATE_ONEOFFS_IDX]
        total = len(items)

        if idx >= total:
            state[STATE_PHASE] = "done"
            return get_current_question(state)

        item = items[idx]
        return {
            "phase":       "oneoffs",
            "description": item["description"],
            "amount":      item["amount"],
            "date":        item["date"],
            "progress":    (idx + 1, total),
        }

    return None   # phase == 'done'


# ── Answer handling ───────────────────────────────────────────────────────────

def submit_answer(state: dict,
                  description: str,
                  category: str) -> dict:
    """
    Record the user's category choice for a transaction.
    Advances to the next question automatically.

    Args:
        state:       Current session state dict (mutated in place)
        description: The merchant/transaction description
        category:    The chosen category string

    Returns:
        Updated state dict
    """
    phase = state[STATE_PHASE]
    store_path = state[STATE_STORE_PATH]

    # Save to confirmed dict and persist to disk
    state[STATE_CONFIRMED][description] = category

    if category != "other":
        # Only persist non-other answers — 'other' stays dynamic
        add_mapping(description, category, store_path)

    # Advance index
    if phase == "recurring":
        state[STATE_RECURRING_IDX] += 1
    elif phase == "oneoffs":
        state[STATE_ONEOFFS_IDX] += 1

    # Check if we should advance phase
    if phase == "recurring":
        if state[STATE_RECURRING_IDX] >= len(state[STATE_RECURRING]):
            state[STATE_PHASE] = "oneoffs" if state[STATE_ONEOFFS] else "done"
    elif phase == "oneoffs":
        if state[STATE_ONEOFFS_IDX] >= len(state[STATE_ONEOFFS]):
            state[STATE_PHASE] = "done"

    return state


def skip_answer(state: dict, description: str) -> dict:
    """
    Skip a one-off transaction — assign to 'other'.
    Same as submit_answer with category='other'.
    """
    return submit_answer(state, description, "other")


def skip_all_remaining(state: dict) -> dict:
    """
    Skip all remaining clarification questions — assign everything to 'other'.
    """
    phase = state[STATE_PHASE]

    if phase == "recurring":
        items = state[STATE_RECURRING]
        idx   = state[STATE_RECURRING_IDX]
        for item in items[idx:]:
            state[STATE_CONFIRMED][item["description"]] = "other"
        state[STATE_RECURRING_IDX] = len(items)
        state[STATE_PHASE] = "oneoffs" if state[STATE_ONEOFFS] else "done"

    if state[STATE_PHASE] == "oneoffs":
        items = state[STATE_ONEOFFS]
        idx   = state[STATE_ONEOFFS_IDX]
        for item in items[idx:]:
            state[STATE_CONFIRMED][item["description"]] = "other"
        state[STATE_ONEOFFS_IDX] = len(items)
        state[STATE_PHASE] = "done"

    return state


# ── Apply confirmed answers to DataFrame ──────────────────────────────────────

def apply_confirmed(df, state: dict):
    """
    Apply all confirmed answers from this session to the DataFrame.
    Call this when phase == 'done'.
    """
    import pandas as pd
    df = df.copy()
    confirmed = state.get(STATE_CONFIRMED, {})

    for desc, cat in confirmed.items():
        mask = df["description"].str.upper().str.strip() == desc.upper().strip()
        df.loc[mask, "category"] = cat

    # Any remaining None → 'other'
    df["category"] = df["category"].fillna("other")
    return df


# ── Progress helpers ──────────────────────────────────────────────────────────

def get_progress_summary(state: dict) -> dict:
    """
    Return a summary of clarification progress for display.
    """
    total_recurring = len(state.get(STATE_RECURRING, []))
    total_oneoffs   = len(state.get(STATE_ONEOFFS, []))
    confirmed       = len(state.get(STATE_CONFIRMED, {}))
    total           = total_recurring + total_oneoffs

    return {
        "total":           total,
        "confirmed":       confirmed,
        "remaining":       total - confirmed,
        "pct_complete":    int(confirmed / total * 100) if total > 0 else 100,
        "total_recurring": total_recurring,
        "total_oneoffs":   total_oneoffs,
        "phase":           state.get(STATE_PHASE, "done"),
    }


def is_done(state: dict) -> bool:
    return state.get(STATE_PHASE) == "done"
