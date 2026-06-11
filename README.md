# 💰 FinSight AI — Personal Finance Intelligence Agent

A fully open-source AI agent that turns your bank statements into clear financial insights.

Upload statements from multiple accounts, get automatic categorization powered by Claude AI, clarify unknowns through a conversational interface, and see your full financial picture in an interactive dashboard.

**[Live Demo →](your-streamlit-url-here)**

![FinSight Dashboard](docs/dashboard_preview.png)

---

## ✨ Features

- **Multi-account support** — upload CSV or PDF statements from any number of accounts simultaneously
- **AI categorization** — Claude automatically categorizes transactions into 14 categories (income, groceries, dining, transport, rent, utilities, entertainment, health, shopping, savings, investment, travel, education, other)
- **Smart clarification** — recurring unknown merchants are surfaced for user confirmation; one-off unknowns default to "other"
- **Memory** — confirmed merchant mappings are saved locally; next upload is faster and smarter
- **Interactive dashboard** — spending by category, income vs expenses by month, net savings trend, filterable transaction table
- **Privacy first** — only transaction descriptions (not amounts) leave your machine for AI categorization. No data stored on any server.
- **Download** — export your fully categorized transactions as CSV

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| AI Agent | Anthropic Claude API (`claude-sonnet-4-6`) |
| PDF Parsing | pdfplumber |
| Data Processing | pandas |
| Frontend | Streamlit |
| Charts | Plotly |
| Memory | Local JSON store |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/finsight-ai.git
cd finsight-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Get a key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
finsight/
├── app.py                          # Main Streamlit app
├── requirements.txt
├── tools/
│   ├── parser.py                   # CSV + PDF bank statement parser
│   ├── categorizer.py              # Claude API categorization tool
│   └── memory.py                   # Merchant → category mapping store
├── agent/
│   └── clarification_agent.py      # Multi-turn clarification session manager
├── data/
│   └── mappings.json               # Auto-generated: your saved mappings
└── tests/
    ├── sample_single_amount.csv     # Test: single amount column format
    └── sample_debit_credit.csv      # Test: separate debit/credit columns
```

---

## 🏦 Supported Bank Statement Formats

**CSV:** Any export with date, description, and amount columns. Handles:
- Single amount column (positive/negative values)
- Separate debit + credit columns
- Date formats: `YYYY-MM-DD`, `DD/MM/YYYY`, `MM/DD/YYYY`

**PDF:** Text-based PDFs from most major banks. Note: scanned/image PDFs are not supported — export as CSV from your bank's online portal instead.

**Tested with:** Commonwealth Bank, ANZ, Westpac, NAB, ING, Macquarie

---

## 🔒 Privacy

- Your bank statement files are parsed locally — they are never uploaded to any server
- Only transaction **descriptions** (e.g. "WOOLWORTHS SUPERMARKET") are sent to the Anthropic API for categorization — amounts, dates, and account details stay on your machine
- Merchant mappings are saved to a local `data/mappings.json` file on your machine only

---

## 🧠 How the AI Agent Works

This project demonstrates a **LangChain-style ReAct agent pattern** implemented with the Anthropic Claude API:

1. **Parse tool** — extracts transactions from uploaded files into a standard schema
2. **Categorize tool** — sends transaction descriptions to Claude in batches; returns structured JSON category labels
3. **Memory tool** — persists confirmed user mappings; applied before LLM calls to reduce API usage
4. **Clarification agent** — manages multi-turn conversation state to resolve ambiguous merchants
5. **Dashboard** — renders Plotly charts from the fully categorized DataFrame

The categorizer uses **structured output prompting** — Claude is instructed to return only a JSON array with no preamble, which is then validated and parsed.

---

## 🤝 Contributing

PRs welcome. Some ideas for contribution:
- Support for more PDF formats
- Additional chart types (cashflow timeline, savings goal tracker)
- Export to Google Sheets
- Multi-currency support
- Budget vs actual comparison

---

## 📄 License

MIT License — free to use, modify, and distribute.
