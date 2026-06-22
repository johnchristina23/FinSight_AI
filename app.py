"""
app.py
------
FinSight AI — Personal Finance Intelligence Agent
Streamlit app: upload bank statements → categorize → clarify → dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from pathlib import Path

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tools.parser import parse_statement
from tools.categorizer import categorize_transactions, get_uncategorized_recurring, get_uncategorized_oneoffs, CATEGORIES
from tools.memory import load_mappings, apply_mappings, get_mapping_stats
from agent.clarification_agent import (
    init_clarification, get_current_question, submit_answer,
    skip_answer, skip_all_remaining, apply_confirmed,
    get_progress_summary, is_done
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #2E75B6;
        margin-bottom: 1rem;
    }
    .category-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .clarify-box {
        background: #EBF5FB;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2E75B6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────

def init_session():
    defaults = {
        "stage":           "upload",    # upload | categorizing | clarifying | dashboard
        "transactions":    None,        # combined DataFrame
        "clarif_state":    None,        # clarification agent state
        "store_path":      "data/mappings.json",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💰 FinSight AI")
    st.markdown("*Personal Finance Intelligence*")
    st.divider()

    # Stage indicator
    stages = ["upload", "categorizing", "clarifying", "dashboard"]
    stage_labels = {
        "upload":       "1. Upload Statements 📁",
        "categorizing": "2. AI Categorization 🤖",
        "clarifying":   "3. Clarify Unknowns💬",
        "dashboard":    "4. Dashboard 📊",
    }
    for s, label in stage_labels.items():
        if st.session_state.stage == s:
            st.markdown(f"**→ {label}**")
        else:
            st.markdown(f"  {label}")

    st.divider()

    # Memory stats
    stats = get_mapping_stats(st.session_state.store_path)
    if stats["total"] > 0:
        st.markdown(f"**🧠 Learned mappings:** {stats['total']}")
        for cat, count in list(stats["by_category"].items())[:5]:
            st.markdown(f"  · {cat}: {count}")

    st.divider()

    if st.session_state.stage == "dashboard":
        if st.button("🔄 Upload New Statements"):
            for key in ["stage","transactions","clarif_state"]:
                st.session_state[key] = "upload" if key == "stage" else None
            st.rerun()


# ── Stage 1: Upload ───────────────────────────────────────────────────────────

if st.session_state.stage == "upload":
    st.markdown('<div class="main-header">💰 FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your bank statements to get a clear picture of your finances</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📁 Upload Bank Statements")
        st.markdown("Upload one or more statements from any account — checking, savings, credit card, or investment.")

        uploaded_files = st.file_uploader(
            "Drop your CSV or PDF bank statements here",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            help="Your files are processed locally and never stored on any server."
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) ready to process:**")
            for f in uploaded_files:
                size_kb = f.size / 1024
                st.markdown(f"  · {f.name} ({size_kb:.1f} KB)")

            if st.button("🚀 Analyze My Finances", type="primary", use_container_width=True):
                all_dfs = []
                errors  = []

                with st.spinner("Parsing your statements..."):
                    for uploaded_file in uploaded_files:
                        suffix = Path(uploaded_file.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        try:
                            df = parse_statement(tmp_path)
                            df["source_file"] = uploaded_file.name
                            all_dfs.append(df)
                            st.success(f"✅ {uploaded_file.name}: {len(df)} transactions parsed")
                        except Exception as e:
                            errors.append(f"❌ {uploaded_file.name}: {str(e)}")
                            st.error(errors[-1])
                        finally:
                            os.unlink(tmp_path)

                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    combined = combined.sort_values("date").reset_index(drop=True)

                    # Apply known mappings from memory
                    combined = apply_mappings(combined, st.session_state.store_path)

                    st.session_state.transactions = combined
                    st.session_state.stage = "categorizing"
                    st.rerun()

    with col2:
        st.markdown("### 🔒 Privacy First")
        st.info(
            "Your data stays on your machine.\n\n"
            "Only transaction descriptions (not amounts) are sent to the AI for categorization.\n\n"
            "No data is stored on any server."
        )
        st.markdown("### 📋 Supported Formats")
        st.markdown(
            "**CSV:** Any bank export with date, description, and amount columns\n\n"
            "**PDF:** Most bank statement PDFs (text-based, not scanned images)"
        )
        st.markdown("### 🏦 Works With")
        st.markdown("Commonwealth Bank · ANZ · Westpac · NAB · ING · Macquarie · Most other banks")


# ── Stage 2: Categorizing ─────────────────────────────────────────────────────

elif st.session_state.stage == "categorizing":
    st.markdown("### 🤖 AI Categorization in Progress")

    df = st.session_state.transactions
    total = len(df)
    already_mapped = df["category"].notna().sum() if "category" in df.columns else 0

    st.markdown(f"**{total} transactions found** · {already_mapped} already recognized from memory")

    with st.spinner(f"Asking Claude to categorize {total - already_mapped} new transactions..."):
        try:
            df = categorize_transactions(df, load_mappings(st.session_state.store_path))

            # Find unknowns for clarification
            recurring = get_uncategorized_recurring(df, min_occurrences=2)
            oneoffs   = get_uncategorized_oneoffs(df, min_occurrences=2)

            st.session_state.transactions    = df
            st.session_state.clarif_state    = init_clarification(
                recurring, oneoffs, st.session_state.store_path
            )
            st.session_state.stage = "clarifying" if (recurring or oneoffs) else "dashboard"
            st.rerun()

        except Exception as e:
            st.error(f"Categorization failed: {str(e)}")
            st.info("Make sure your ANTHROPIC_API_KEY is set in your environment.")
            if st.button("⬅️ Go Back"):
                st.session_state.stage = "upload"
                st.rerun()


# ── Stage 3: Clarification ────────────────────────────────────────────────────

elif st.session_state.stage == "clarifying":
    state = st.session_state.clarif_state

    if is_done(state):
        df = apply_confirmed(st.session_state.transactions, state)
        st.session_state.transactions = df
        st.session_state.stage = "dashboard"
        st.rerun()

    question = get_current_question(state)
    progress = get_progress_summary(state)

    st.markdown("### 💬 Help Me Categorize These Transactions")
    st.markdown(f"I found **{progress['total']} merchants** I'm not sure about. Your answers help me get smarter over time.")

    # Progress bar
    st.progress(progress["pct_complete"] / 100,
                text=f"{progress['confirmed']}/{progress['total']} clarified ({progress['pct_complete']}%)")

    if question:
        st.markdown('<div class="clarify-box">', unsafe_allow_html=True)

        if question["phase"] == "recurring":
            amt_str = f"${abs(question['total']):.2f}" if question['total'] < 0 else f"+${question['total']:.2f}"
            st.markdown(f"**I see this {question['count']} times in your statements:**")
            st.markdown(f"### `{question['description']}`")
            st.markdown(f"Total: **{amt_str}** across {question['count']} transactions")
            if question.get("dates"):
                st.caption(f"Dates: {', '.join(question['dates'])}")
        else:
            amt = question["amount"]
            amt_str = f"-${abs(amt):.2f}" if amt < 0 else f"+${amt:.2f}"
            st.markdown(f"**I saw this transaction once:**")
            st.markdown(f"### `{question['description']}`")
            st.markdown(f"{question['date']} · **{amt_str}**")

        st.markdown('</div>', unsafe_allow_html=True)

        # Category buttons — grid layout
        st.markdown("**What is this?**")

        # Split categories into rows of 4
        cat_cols = st.columns(4)
        cat_icons = {
            "income": "💵", "groceries": "🛒", "dining": "🍽️",
            "transport": "🚗", "utilities": "💡", "rent": "🏠",
            "entertainment": "🎬", "health": "💊", "shopping": "🛍️",
            "savings": "🏦", "investment": "📈", "travel": "✈️",
            "education": "📚", "other": "📦"
        }

        for i, cat in enumerate(CATEGORIES):
            with cat_cols[i % 4]:
                icon = cat_icons.get(cat, "•")
                if st.button(f"{icon} {cat.title()}", key=f"cat_{cat}",
                             use_container_width=True):
                    submit_answer(state, question["description"], cat)
                    st.session_state.clarif_state = state
                    st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏭️ Skip (mark as 'Other')", use_container_width=True):
                skip_answer(state, question["description"])
                st.session_state.clarif_state = state
                st.rerun()
        with col2:
            if st.button("⏩ Skip All Remaining", use_container_width=True):
                skip_all_remaining(state)
                st.session_state.clarif_state = state
                st.rerun()


# ── Stage 4: Dashboard ────────────────────────────────────────────────────────

elif st.session_state.stage == "dashboard":
    df = st.session_state.transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    income_df  = df[df["type"] == "credit"]
    expense_df = df[df["type"] == "debit"]

    total_income   = income_df["amount"].sum()
    total_expenses = abs(expense_df["amount"].sum())
    net_savings    = total_income - total_expenses
    savings_rate   = (net_savings / total_income * 100) if total_income > 0 else 0

    # ── Header ──
    st.markdown('<div class="main-header">📊 Your Financial Dashboard</div>', unsafe_allow_html=True)

    date_range = f"{df['date'].min().strftime('%b %d, %Y')} → {df['date'].max().strftime('%b %d, %Y')}"
    sources    = ", ".join(df["source_file"].unique())
    st.markdown(f'<div class="sub-header">{date_range} · {sources}</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💵 Total Income",   f"${total_income:,.2f}")
    with col2:
        st.metric("💸 Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        delta_color = "normal" if net_savings >= 0 else "inverse"
        st.metric("🏦 Net Savings", f"${net_savings:,.2f}",
                  delta=f"{savings_rate:.1f}% savings rate")
    with col4:
        st.metric("📋 Transactions", f"{len(df):,}",
                  delta=f"{df['source_file'].nunique()} account(s)")

    st.divider()

    # ── Charts Row 1 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🍕 Spending by Category")
        cat_totals = expense_df.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
        fig_pie = px.pie(
            values=cat_totals.values,
            names=cat_totals.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Income vs Expenses by Month")
        monthly_income   = income_df.groupby("month")["amount"].sum()
        monthly_expenses = expense_df.groupby("month")["amount"].sum().abs()
        months = sorted(set(monthly_income.index) | set(monthly_expenses.index))

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=months,
            y=[monthly_income.get(m, 0) for m in months],
            name="Income", marker_color="#2E75B6"
        ))
        fig_bar.add_trace(go.Bar(
            x=months,
            y=[monthly_expenses.get(m, 0) for m in months],
            name="Expenses", marker_color="#E74C3C"
        ))
        fig_bar.update_layout(
            barmode="group",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_tickprefix="$"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Charts Row 2 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💸 Top Spending Categories")
        cat_totals_df = cat_totals.reset_index()
        cat_totals_df.columns = ["Category", "Amount"]
        fig_bar2 = px.bar(
            cat_totals_df, x="Amount", y="Category",
            orientation="h",
            color="Amount",
            color_continuous_scale="Blues",
            text_auto=".2s"
        )
        fig_bar2.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_tickprefix="$",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar2, use_container_width=True)

    with col2:
        st.markdown("#### 📅 Monthly Net Savings")
        monthly_net = pd.Series(
            {m: monthly_income.get(m, 0) - monthly_expenses.get(m, 0) for m in months}
        )
        colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in monthly_net.values]
        fig_net = go.Figure(go.Bar(
            x=months,
            y=monthly_net.values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in monthly_net.values],
            textposition="outside"
        ))
        fig_net.update_layout(
            margin=dict(t=30, b=10, l=10, r=10),
            yaxis_tickprefix="$"
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.divider()

    # ── Transaction Table ──
    st.markdown("#### 🔍 Transaction Details")

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_cats = st.multiselect(
            "Filter by category",
            options=sorted(df["category"].dropna().unique()),
            default=[]
        )
    with filter_col2:
        txn_type = st.selectbox("Transaction type", ["All", "Debits only", "Credits only"])
    with filter_col3:
        search = st.text_input("Search description", placeholder="e.g. Uber, Netflix...")

    filtered = df.copy()
    if selected_cats:
        filtered = filtered[filtered["category"].isin(selected_cats)]
    if txn_type == "Debits only":
        filtered = filtered[filtered["type"] == "debit"]
    elif txn_type == "Credits only":
        filtered = filtered[filtered["type"] == "credit"]
    if search:
        filtered = filtered[filtered["description"].str.contains(search, case=False, na=False)]

    filtered_display = filtered[["date", "description", "amount", "type", "category", "source_file"]].copy()
    filtered_display["date"] = filtered_display["date"].dt.strftime("%Y-%m-%d")
    filtered_display["amount"] = filtered_display["amount"].apply(
        lambda x: f"${abs(x):,.2f}" if x < 0 else f"+${x:,.2f}"
    )

    st.dataframe(
        filtered_display,
        use_container_width=True,
        height=400,
        column_config={
            "date":        st.column_config.TextColumn("Date"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "amount":      st.column_config.TextColumn("Amount"),
            "type":        st.column_config.TextColumn("Type"),
            "category":    st.column_config.TextColumn("Category"),
            "source_file": st.column_config.TextColumn("Source"),
        }
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download categorized transactions as CSV",
        csv,
        "finsight_transactions.csv",
        "text/csv",
        use_container_width=False
    )
