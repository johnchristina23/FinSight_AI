"""
parser.py
---------
Parses bank statements from CSV or PDF into a standardised
pandas DataFrame with columns:
    date        | datetime
    description | str
    amount      | float  (negative = debit, positive = credit)
    type        | 'debit' | 'credit'
    source_file | str
"""

import re
import pdfplumber
import pandas as pd
from pathlib import Path
from datetime import datetime


# ── Column name aliases ──────────────────────────────────────────────────────
# Different banks use different column names — we normalise them all.

DATE_ALIASES = ["date", "transaction date", "posted date", "trans date", "value date"]
DESC_ALIASES = ["description", "details", "narrative", "memo",
                "transaction description", "merchant", "payee", "particulars"]
AMOUNT_ALIASES = ["amount", "transaction amount", "value"]
DEBIT_ALIASES  = ["debit", "debit amount", "withdrawal", "withdrawals", "dr"]
CREDIT_ALIASES = ["credit", "credit amount", "deposit", "deposits", "cr"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalise_col(col: str) -> str:
    return col.strip().lower().replace("_", " ")


def _find_col(df_cols: list[str], aliases: list[str]) -> str | None:
    normalised = {_normalise_col(c): c for c in df_cols}
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    return None


def _parse_amount(val) -> float:
    """Strip currency symbols, commas, brackets; return float."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    # Bracketed amounts are negative: (1,234.56) → -1234.56
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d.\-]", "", s.replace("(", "-").replace(")", ""))
    try:
        result = float(s)
        return -abs(result) if negative else result
    except ValueError:
        return 0.0


def _infer_type(amount: float) -> str:
    return "credit" if amount > 0 else "debit"


# ── CSV Parser ───────────────────────────────────────────────────────────────

def parse_csv(file_path: str) -> pd.DataFrame:
    """
    Parse a bank statement CSV into the standard schema.
    Handles both single-amount-column and separate debit/credit column layouts.
    """
    path = Path(file_path)
    df_raw = pd.read_csv(path, thousands=",")
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    cols = list(df_raw.columns)

    # ── Date ──
    date_col = _find_col(cols, DATE_ALIASES)
    if not date_col:
        raise ValueError(f"Could not find a date column in {path.name}. "
                         f"Columns found: {cols}")
    # Try both date formats — use whichever parses more rows successfully
    dates_a = pd.to_datetime(df_raw[date_col], dayfirst=False, errors="coerce")
    dates_b = pd.to_datetime(df_raw[date_col], dayfirst=True,  errors="coerce")
    df_raw[date_col] = dates_a if dates_a.notna().sum() >= dates_b.notna().sum() else dates_b

    # ── Description ──
    desc_col = _find_col(cols, DESC_ALIASES)
    if not desc_col:
        raise ValueError(f"Could not find a description column in {path.name}.")

    # ── Amount — two layouts ──
    amount_col  = _find_col(cols, AMOUNT_ALIASES)
    debit_col   = _find_col(cols, DEBIT_ALIASES)
    credit_col  = _find_col(cols, CREDIT_ALIASES)

    if amount_col:
        # Single column — negative = debit, positive = credit
        amounts = df_raw[amount_col].apply(_parse_amount)

    elif debit_col and credit_col:
        # Separate columns — combine; debits become negative
        debits  = df_raw[debit_col].apply(_parse_amount).abs() * -1
        credits = df_raw[credit_col].apply(_parse_amount).abs()
        # Rows have only one populated; sum them
        amounts = debits.fillna(0) + credits.fillna(0)

    else:
        raise ValueError(f"Could not find amount column(s) in {path.name}.")

    # ── Build standard DataFrame ──
    result = pd.DataFrame({
        "date":        df_raw[date_col],
        "description": df_raw[desc_col].astype(str).str.strip(),
        "amount":      amounts,
        "type":        amounts.apply(_infer_type),
        "source_file": path.name,
    })

    return result.dropna(subset=["date"]).reset_index(drop=True)


# ── PDF Parser ───────────────────────────────────────────────────────────────

# Common date patterns found in bank statement PDFs
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE
)
_AMOUNT_PATTERN = re.compile(r"-?\$?[\d,]+\.\d{2}")


def _extract_transactions_from_text(text: str, source_file: str) -> pd.DataFrame:
    """
    Heuristic extraction from raw PDF text.
    Looks for lines containing a date and at least one dollar amount.
    """
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        date_match   = _DATE_PATTERN.search(line)
        amount_match = _AMOUNT_PATTERN.findall(line)

        if not date_match or not amount_match:
            continue

        # Use the last amount on the line (usually the transaction amount)
        raw_amount = amount_match[-1]
        amount = _parse_amount(raw_amount)

        # Description = text between date and amount
        desc_start = date_match.end()
        desc_end   = line.rfind(raw_amount)
        description = line[desc_start:desc_end].strip(" -|,")
        if not description:
            description = line[date_match.end():].strip()

        try:
            date = pd.to_datetime(date_match.group(), 
                                   dayfirst=True)
        except Exception:
            continue

        rows.append({
            "date":        date,
            "description": description,
            "amount":      amount,
            "type":        _infer_type(amount),
            "source_file": source_file,
        })

    return pd.DataFrame(rows)


def parse_pdf(file_path: str) -> pd.DataFrame:
    """
    Parse a bank statement PDF into the standard schema.
    Uses pdfplumber for table extraction first, falls back to text heuristics.
    """
    path = Path(file_path)
    all_frames = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            # ── Try table extraction first ──
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                df_table = pd.DataFrame(table[1:], columns=table[0])
                df_table.columns = [str(c).strip() if c else f"col_{i}"
                                    for i, c in enumerate(df_table.columns)]
                try:
                    parsed = parse_csv.__wrapped__(df_table) if hasattr(parse_csv, "__wrapped__") \
                             else _parse_df_table(df_table, path.name)
                    if not parsed.empty:
                        all_frames.append(parsed)
                        continue
                except Exception:
                    pass

            # ── Fallback: raw text heuristics ──
            text = page.extract_text() or ""
            text_frame = _extract_transactions_from_text(text, path.name)
            if not text_frame.empty:
                all_frames.append(text_frame)

    if not all_frames:
        raise ValueError(f"No transactions could be extracted from {path.name}. "
                         "The PDF may be scanned/image-based — try exporting as CSV from your bank.")

    return pd.concat(all_frames, ignore_index=True).drop_duplicates(
        subset=["date", "description", "amount"]
    ).reset_index(drop=True)


def _parse_df_table(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Parse a DataFrame extracted from a PDF table using the same logic as CSV."""
    cols = list(df.columns)

    date_col = _find_col(cols, DATE_ALIASES)
    desc_col = _find_col(cols, DESC_ALIASES)
    if not date_col or not desc_col:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], 
                                   dayfirst=True, errors="coerce")

    amount_col = _find_col(cols, AMOUNT_ALIASES)
    debit_col  = _find_col(cols, DEBIT_ALIASES)
    credit_col = _find_col(cols, CREDIT_ALIASES)

    if amount_col:
        amounts = df[amount_col].apply(_parse_amount)
    elif debit_col and credit_col:
        debits  = df[debit_col].apply(_parse_amount).abs() * -1
        credits = df[credit_col].apply(_parse_amount).abs()
        amounts = debits.fillna(0) + credits.fillna(0)
    else:
        return pd.DataFrame()

    result = pd.DataFrame({
        "date":        df[date_col],
        "description": df[desc_col].astype(str).str.strip(),
        "amount":      amounts,
        "type":        amounts.apply(_infer_type),
        "source_file": source_file,
    })
    return result.dropna(subset=["date"]).reset_index(drop=True)


# ── Main entry point ─────────────────────────────────────────────────────────

def parse_statement(file_path: str) -> pd.DataFrame:
    """
    Parse any bank statement file (CSV or PDF) into the standard schema.
    This is the function called by the LangChain agent tool.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return parse_csv(file_path)
    elif suffix == ".pdf":
        return parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Please upload a CSV or PDF.")


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = parse_statement(sys.argv[1])
        print(f"\nParsed {len(df)} transactions from {sys.argv[1]}")
        print(df.head(10).to_string())
    else:
        print("Usage: python parser.py <path_to_statement.csv_or_pdf>")
