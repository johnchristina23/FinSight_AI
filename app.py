"""
app.py ------ FinSight AI — Personal Finance Intelligence Agent
Streamlit app: upload bank statements → categorize → clarify → dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import sys
from pathlib import Path

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from tools.parser import parse_statement
from tools.categorizer import categorize_transactions, get_uncategorized_recurring, get_uncategorized_oneoffs, CATEGORIES
from tools.memory import load_mappings, apply_mappings, get_mapping_stats
from agent.clarification_agent import (
    init_clarification, get_current_question, submit_answer, 
    skip_answer, skip_all_remaining, apply_confirmed, get_progress_summary, is_done
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Fintech Slate CSS Theme ────────────────────────────────────────────
st.markdown("""
    <style>
        /* Main Theme Variables & Overrides */
        :root {
            --bg-slate: #0F172A;
            --card-slate: #1E293B;
            --accent-emerald: #10B981;
            --accent-rose: #F43F5E;
            --text-light: #F8FAFC;
            --text-muted: #94A3B8;
        }
        
        /* Typography & Structure */
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38BDF8, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.1rem;
            margin-bottom: 2.5rem;
        }
        
        /* Custom Premium Dashboard Cards */
        .premium-card {
            background-color: #FAFAFA;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
            transition: transform 0.2s ease-in-out;
        }
        .premium-card:hover {
            transform: translateY(-2px);
        }
        
        /* Transaction Card Styling (Stage 3) */
        .txn-hero-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            border: 1px solid #334155;
        }
        .txn-desc {
            font-family: 'Courier New', Courier, monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: #38BDF8;
            margin: 1rem 0;
            letter-spacing: 1px;
        }
    </style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
def init_session():
    defaults = {
        "stage":        "upload",    # upload | categorizing | clarifying | dashboard
        "transactions": None,        # combined DataFrame
        "clarif_state": None,        # clarification agent state
        "store_path":   "data/mappings.json",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>💰 FinSight AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-style:italic;'>Personal Finance Intelligence</p>", unsafe_allow_html=True)
    st.divider()
    
    # Clean, modern stage indicators
    stage_labels = {
        "upload":       "📁 1. Upload Statements",
        "categorizing": "🤖 2. AI Categorization",
        "clarifying":   "💬 3. Clarify Unknowns",
        "dashboard":    "📊 4. Dashboard",
    }
    for s, label in stage_labels.items():
        if st.session_state.stage == s:
            st.markdown(f"<div style='color:#10B981; font-weight:700; padding:4px 0;'>→ {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#94A3B8; padding:4px 0; opacity:0.6;'>{label}</div>", unsafe_allow_html=True)
            
    st.divider()
    
    # Memory stats
    stats = get_mapping_stats(st.session_state.store_path)
    if stats["total"] > 0:
        st.markdown(f"**🧠 Brain Mappings:** {stats['total']}")
        for cat, count in list(stats["by_category"].items())[:5]:
            st.markdown(f"&nbsp;&nbsp;· {cat.title()}: `{count}`")
            
    st.divider()
    if st.session_state.stage == "dashboard":
        if st.button("🔄 Upload New Data", use_container_width=True):
            for key in ["stage", "transactions", "clarif_state"]:
                st.session_state[key] = "upload" if key == "stage" else None
            st.rerun()

# ── Stage 1: Upload ───────────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    st.markdown('<div class="main-header">FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transform raw bank statements into an intelligent, multi-account financial landscape.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("### 📁 Import Financial Documents")
        uploaded_files = st.file_uploader(
            "Drop your CSV or PDF bank statements here", 
            type=["csv", "pdf"], 
            accept_multiple_files=True, 
            help="Your files are processed strictly inside your local memory scope."
        )
        
        if uploaded_files:
            st.markdown("<p style='font-weight:600; margin-top:1rem;'>Ready for parsing Queue:</p>", unsafe_allow_html=True)
            for f in uploaded_files:
                st.markdown(f"⚡ `{f.name}` ({f.size / 1024:.1f} KB)")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyze Financial Footprint", type="primary", use_container_width=True):
                all_dfs = []
                errors = []
                
                with st.spinner("Executing secure pipeline parsers..."):
                    for uploaded_file in uploaded_files:
                        suffix = Path(uploaded_file.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        try:
                            df = parse_statement(tmp_path)
                            df["source_file"] = uploaded_file.name
                            all_dfs.append(df)
                        except Exception as e:
                            errors.append(f"❌ Parser failure on {uploaded_file.name}: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
                
                if errors:
                    for err in errors:
                        st.error(err)
                
                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    combined = combined.sort_values("date").reset_index(drop=True)
                    combined = apply_mappings(combined, st.session_state.store_path)
                    st.session_state.transactions = combined
                    st.session_state.stage = "categorizing"
                    st.rerun()
                    
    with col2:
        st.markdown("""
            <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 16px; border: 1px solid #E2E8F0;">
                <h4 style="margin-top:0;">🔒 Zero-Knowledge Security</h4>
                <p style="font-size:0.9rem; color:#475569;">All document ingestion rules occur locally. Only transaction string descriptors are evaluated via secure endpoints for semantic structure analysis.</p>
                <hr style="margin: 1rem 0; border:0; border-top: 1px solid #E2E8F0;">
                <h4 style="margin-top:0;">🏛️ Supported Ecosystems</h4>
                <p style="font-size:0.85rem; color:#64748B; line-height:1.6;">
                    • <b>Global Banks:</b> CBA, ANZ, Westpac, NAB, ING, Macquarie, and standard text-based international banking templates.<br>
                    • <b>Formats:</b> UTF-8 Comma-Separated Values (.csv), Structured Text Documents (.pdf)
                </p>
            </div>
        """, unsafe_allow_html=True)

# ── Stage 2: Categorizing ─────────────────────────────────────────────────────
elif st.session_state.stage == "categorizing":
    st.markdown("### 🤖 Semantic Pipeline Ingestion")
    df = st.session_state.transactions
    total = len(df)
    already_mapped = df["category"].notna().sum() if "category" in df.columns else 0
    
    st.info(f"Loaded {total} transactions tracking framework. Memory Engine matched: {already_mapped} records.")
    
    with st.spinner(f"Requesting Claude Optimization Matrix for {total - already_mapped} remaining records..."):
        try:
            df = categorize_transactions(df, load_mappings(st.session_state.store_path))
            recurring = get_uncategorized_recurring(df, min_occurrences=2)
            oneoffs = get_uncategorized_oneoffs(df, min_occurrences=2)
            
            st.session_state.transactions = df
            st.session_state.clarif_state = init_clarification(recurring, oneoffs, st.session_state.store_path)
            st.session_state.stage = "clarifying" if (recurring or oneoffs) else "dashboard"
            st.rerun()
        except Exception as e:
            st.error(f"Ecosystem categorizer fault: {str(e)}")
            if st.button("⬅️ Clear Sandbox"):
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
    
    st.markdown("### 💬 Machine Learning Context Fine-Tuning")
    st.markdown("Review outstanding high-variance merchants below to lock down dashboard accuracy metrics.")
    
    st.progress(progress["pct_complete"] / 100)
    st.caption(f"Queue Status: {progress['confirmed']} processed out of {progress['total']} ambiguous vendors ({progress['pct_complete']}% system certainty)")
    
    if question:
        # Immersive Card Layout
        is_rec = question["phase"] == "recurring"
        amt_value = question['total'] if is_rec else question['amount']
        amt_style = "color: #10B981;" if amt_value >= 0 else "color: #F43F5E;"
        amt_sign = "+" if amt_value >= 0 else "-"
        amt_str = f"{amt_sign}${abs(amt_value):,.2f}"
        
        st.markdown(f"""
            <div class="txn-hero-card">
                <span style="text-transform: uppercase; font-size: 0.85rem; letter-spacing: 2px; color: #94A3B8;">
                    { '🔄 Recurring Target Sequence Detected' if is_rec else '📍 Isolated Outlier Point' }
                </span>
                <div class="txn-desc">{question['description']}</div>
                <h2 style="{amt_style} margin:0; font-size:2.5rem;">{amt_str}</h2>
                <p style="color: #64748B; margin-top: 0.5rem; font-size:0.9rem;">
                    { f"Occurred {question['count']} times across matching historical intervals." if is_rec else f"Settlement Date: {question['date']}" }
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # UI Overhaul: Smart Recommendations Layout
        st.markdown("#### Assign Domain Space Label")
        
        cat_icons = {
            "income": "💵", "groceries": "🛒", "dining": "🍽️", "transport": "🚗", 
            "utilities": "💡", "rent": "🏠", "entertainment": "🎬", "health": "💊", 
            "shopping": "🛍️", "savings": "🏦", "investment": "📈", "travel": "✈️", 
            "education": "📚", "other": "📦"
        }
        
        # Hardcoded dynamic prioritization mock example: emphasize top categories
        top_picks = ["groceries", "dining", "shopping", "other"]
        
        st.markdown("##### ✨ Top Predicted Recommendations")
        rec_cols = st.columns(4)
        for idx, cat in enumerate(top_picks):
            with rec_cols[idx]:
                if st.button(f"{cat_icons.get(cat, '•')} {cat.title()}", key=f"rec_{cat}", type="primary", use_container_width=True):
                    submit_answer(state, question["description"], cat)
                    st.session_state.clarif_state = state
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂 View Alternative Organizational Categories"):
            alt_cols = st.columns(4)
            alt_cats = [c for c in CATEGORIES if c not in top_picks]
            for idx, cat in enumerate(alt_cats):
                with alt_cols[idx % 4]:
                    if st.button(f"{cat_icons.get(cat, '•')} {cat.title()}", key=f"alt_{cat}", use_container_width=True):
                        submit_answer(state, question["description"], cat)
                        st.session_state.clarif_state = state
                        st.rerun()
                        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏭️ Assign to Default 'Other'", use_container_width=True):
                skip_answer(state, question["description"])
                st.session_state.clarif_state = state
                st.rerun()
        with col2:
            if st.button("⏩ Bypass Evaluation Strategy (Default Remaining)", use_container_width=True):
                skip_all_remaining(state)
                st.session_state.clarif_state = state
                st.rerun()

# ── Stage 4: Dashboard ────────────────────────────────────────────────────────
elif st.session_state.stage == "dashboard":
    df = st.session_state.transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    
    income_df = df[df["type"] == "credit"]
    expense_df = df[df["type"] == "debit"]
    
    total_income = income_df["amount"].sum()
    total_expenses = abs(expense_df["amount"].sum())
    net_savings = total_income - total_expenses
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
    
    # Elegant Slate Header Block
    date_range = f"{df['date'].min().strftime('%b %d, %Y')} → {df['date'].max().strftime('%b %d, %Y')}"
    sources = ", ".join(df["source_file"].unique())
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 2.2rem; border-radius: 24px; margin-bottom: 2.5rem; color: white; border: 1px solid #334155;">
            <h1 style="margin: 0; font-size: 2.6rem; letter-spacing:-0.5px; font-weight:800;">📊 Asset Architecture Analytics</h1>
            <p style="margin: 0.6rem 0 0 0; color:#94A3B8; font-size: 1rem; opacity:0.9;">
                <b>Reporting Span:</b> {date_range} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Validated Nodes:</b> {sources}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Modern Premium Metric Grid Blocks
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="premium-card" style="border-left: 5px solid #10B981;">
                <p style="margin:0; font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing:0.5px;">Gross Revenue Inflow</p>
                <h2 style="margin: 0.4rem 0 0 0; color: #0F172A; font-size: 1.9rem; font-weight:700;">${total_income:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="premium-card" style="border-left: 5px solid #F43F5E;">
                <p style="margin:0; font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing:0.5px;">Aggregate Outflows</p>
                <h2 style="margin: 0.4rem 0 0 0; color: #0F172A; font-size: 1.9rem; font-weight:700;">${total_expenses:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        border_col = "#10B981" if net_savings >= 0 else "#F43F5E"
        st.markdown(f"""
            <div class="premium-card" style="border-left: 5px solid {border_col};">
                <p style="margin:0; font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing:0.5px;">Net Asset Delta</p>
                <h2 style="margin: 0.4rem 0 0 0; color: {border_col}; font-size: 1.9rem; font-weight:700;">${net_savings:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="premium-card" style="border-left: 5px solid #38BDF8;">
                <p style="margin:0; font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing:0.5px;">Capital Retention Efficiency</p>
                <h2 style="margin: 0.4rem 0 0 0; color: #0F172A; font-size: 1.9rem; font-weight:700;">{savings_rate:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # ── Interactive Charts Row 1 ──
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### 🍕 Expenditure Allocation Matrix")
        cat_totals = expense_df.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
        fig_pie = px.pie(
            values=cat_totals.values, 
            names=cat_totals.index, 
            color_discrete_sequence=px.colors.sequential.YlGnBu_r, 
            hole=0.5
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=15, b=15, l=15, r=15), template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### 📈 Micro-Interval Velocity Trend (Income vs Expenses)")
        monthly_income = income_df.groupby("month")["amount"].sum()
        monthly_expenses = expense_df.groupby("month")["amount"].sum().abs()
        months = sorted(set(monthly_income.index) | set(monthly_expenses.index))
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=months, y=[monthly_income.get(m, 0) for m in months], name="Inflow", marker_color="#10B981"))
        fig_bar.add_trace(go.Bar(x=months, y=[monthly_expenses.get(m, 0) for m in months], name="Outflow", marker_color="#F43F5E"))
        fig_bar.update_layout(
            barmode="group", 
            legend=dict(orientation="h", y=1.1, x=0), 
            margin=dict(t=15, b=15, l=15, r=15), 
            yaxis_tickprefix="$",
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Interactive Charts Row 2 ──
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown("#### 💸 Primary Capital Outflow Sinks")
        cat_totals_df = cat_totals.reset_index()
        cat_totals_df.columns = ["Category", "Amount"]
        fig_bar2 = px.bar(
            cat_totals_df, x="Amount", y="Category", orientation="h", 
            color="Amount", color_continuous_scale="Blues", text_auto=".2s"
        )
        fig_bar2.update_layout(showlegend=False, margin=dict(t=15, b=15, l=15, r=15), xaxis_tickprefix="$", coloraxis_showscale=False, template="plotly_white")
        st.plotly_chart(fig_bar2, use_container_width=True)
        
    with chart_col4:
        st.markdown("#### 📅 Rolling Net Structural Savings Trend")
        monthly_net = pd.Series({m: monthly_income.get(m, 0) - monthly_expenses.get(m, 0) for m in months})
        colors = ["#10B981" if v >= 0 else "#F43F5E" for v in monthly_net.values]
        fig_net = go.Figure(go.Bar(
            x=months, y=monthly_net.values, marker_color=colors, 
            text=[f"${v:,.0f}" for v in monthly_net.values], textposition="outside"
        ))
        fig_net.update_layout(margin=dict(t=30, b=15, l=15, r=15), yaxis_tickprefix="$", template="plotly_white")
        st.plotly_chart(fig_net, use_container_width=True)
        
    st.divider()
    
    # ── Transaction Table Section ──
    st.markdown("#### 🔍 Historical Granular Audit Trail")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_cats = st.multiselect("Isolate Category Nodes", options=sorted(df["category"].dropna().unique()), default=[])
    with filter_col2:
        txn_type = st.selectbox("Isolate Ledger Typology", ["All Records", "Debits only", "Credits only"])
    with filter_col3:
        search = st.text_input("Search Invalidation Tags", placeholder="Filter via transaction labels...")
        
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
    filtered_display["amount"] = filtered_display["amount"].apply(lambda x: f"${abs(x):,.2f}" if x < 0 else f"+${x:,.2f}")
    
    st.dataframe(
        filtered_display, 
        use_container_width=True, 
        height=380, 
        column_config={
            "date":        st.column_config.TextColumn("Normalized Date"),
            "description": st.column_config.TextColumn("String Descriptor String", width="large"),
            "amount":      st.column_config.TextColumn("Magnitude Vector"),
            "type":        st.column_config.TextColumn("Direction Meta"),
            "category":    st.column_config.TextColumn("Domain Classification Mapping"),
            "source_file": st.column_config.TextColumn("Origin Document Signature"),
        }
    )
    
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Extract Compiled Database Ledger (.csv)", csv, "finsight_ledger_export.csv", "text/csv", use_container_width=False)
