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

ICON_PATHS = {
    "wallet": '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h11A2.5 2.5 0 0 1 19 7.5v9a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 3 16.5v-9Z"/><path d="M3 8h13.5A2.5 2.5 0 0 1 19 10.5V14h-4a2.5 2.5 0 0 1 0-5h4"/><path d="M15 11.5h.01"/>',
    "folder": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h4l2 2H19a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 18H4.5A1.5 1.5 0 0 1 3 16.5v-10Z"/>',
    "bot": '<rect x="4" y="7" width="16" height="12" rx="3"/><path d="M12 3v4M8 12h.01M16 12h.01M8 16h8"/>',
    "message": '<path d="M5 5.5h14A2.5 2.5 0 0 1 21.5 8v7a2.5 2.5 0 0 1-2.5 2.5H11l-4.5 3v-3H5A2.5 2.5 0 0 1 2.5 15V8A2.5 2.5 0 0 1 5 5.5Z"/><path d="M7.5 11.5h.01M12 11.5h.01M16.5 11.5h.01"/>',
    "chart": '<path d="M4 19V5M4 19h16"/><path d="m7 15 3-4 3 2 5-7"/>',
    "brain": '<path d="M9 4.5A3.5 3.5 0 0 0 5.5 8c0 .35.05.69.15 1A3.5 3.5 0 0 0 4 15.5 3.5 3.5 0 0 0 8 19h1V4.5ZM15 4.5A3.5 3.5 0 0 1 18.5 8c0 .35-.05.69-.15 1A3.5 3.5 0 0 1 20 15.5 3.5 3.5 0 0 1 16 19h-1V4.5ZM9 8h6M9 12h6M9 16h6"/>',
    "refresh": '<path d="M19 8a7 7 0 0 0-12.5-2L5 8"/><path d="M5 4v4h4M5 16a7 7 0 0 0 12.5 2L19 16"/><path d="M19 20v-4h-4"/>',
    "analyze": '<path d="M4 12a8 8 0 1 0 8-8"/><path d="M4 4v8h8M12 8v8M8 12h8"/>',
    "lock": '<rect x="5" y="9" width="14" height="11" rx="2"/><path d="M8 9V6a4 4 0 0 1 8 0v3M12 13v3"/>',
    "bank": '<path d="m3 9 9-5 9 5M5 10v7M9 10v7M15 10v7M19 10v7M3 19h18"/>',
    "back": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "repeat": '<path d="M17 2l4 4-4 4M3 12V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4M21 12v3a3 3 0 0 1-3 3H3"/>',
    "pin": '<path d="M12 21s6-5.25 6-11a6 6 0 1 0-12 0c0 5.75 6 11 6 11Z"/><circle cx="12" cy="10" r="2"/>',
    "sparkles": '<path d="m12 3 1.2 4.8L18 9l-4.8 1.2L12 15l-1.2-4.8L6 9l4.8-1.2L12 3ZM19 15l.6 2.4L22 18l-2.4.6L19 21l-.6-2.4L16 18l2.4-.6L19 15Z"/>',
    "skip": '<path d="m5 5 7 7-7 7M12 5l7 7-7 7"/>',
    "file": '<path d="M6 3h8l4 4v14H6zM14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
    "error": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>',
    "generic": '<circle cx="12" cy="12" r="8"/><path d="M9 12h6M12 9v6"/>'
}

def svg_icon(name, size=18, color="#64748B"):
    path = ICON_PATHS.get(name, ICON_PATHS["generic"])
    return (
        f'<svg class="svg-icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )

def icon_label(icon_name, text, color="#64748B", size=18):
    return f'<span class="icon-label">{svg_icon(icon_name, size, color)}<span>{text}</span></span>'

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
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
        .svg-icon {
            display: inline-block;
            vertical-align: -0.18em;
            flex: 0 0 auto;
        }
        .icon-label {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
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
    st.markdown(f"<h2 style='margin-bottom:0;'>{icon_label('wallet', 'FinSight AI', '#10B981', 22)}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-style:italic;'>Personal Finance Intelligence</p>", unsafe_allow_html=True)
    st.divider()
    
    # Clean, modern stage indicators
    stage_labels = {
        "upload":       ("folder", "1. Upload Statements"),
        "categorizing": ("bot", "2. AI Categorization"),
        "clarifying":   ("message", "3. Clarify Unknowns"),
        "dashboard":    ("chart", "4. Dashboard"),
    }
    for s, (stage_icon, stage_text) in stage_labels.items():
        label = icon_label(stage_icon, stage_text, '#10B981' if st.session_state.stage == s else '#94A3B8')
        if st.session_state.stage == s:
            st.markdown(f"<div style='color:#10B981; font-weight:700; padding:4px 0;'>→ {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#94A3B8; padding:4px 0; opacity:0.6;'>{label}</div>", unsafe_allow_html=True)
            
    st.divider()
    
    # Memory stats
    stats = get_mapping_stats(st.session_state.store_path)
    if stats["total"] > 0:
        mapping_label = f"Brain Mappings: <b>{stats['total']}</b>"
        st.markdown(icon_label('brain', mapping_label), unsafe_allow_html=True)
        for cat, count in list(stats["by_category"].items())[:5]:
            st.markdown(f"&nbsp;&nbsp;· {cat.title()}: `{count}`")
            
    st.divider()
    if st.session_state.stage == "dashboard":
        st.markdown(icon_label('refresh', 'Upload New Data', '#64748B'), unsafe_allow_html=True)
        if st.button("Upload New Data", use_container_width=True):
            for key in ["stage", "transactions", "clarif_state"]:
                st.session_state[key] = "upload" if key == "stage" else None
            st.rerun()

# ── Stage 1: Upload ───────────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    st.markdown('<div class="main-header">FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transform raw bank statements into an intelligent, multi-account financial landscape.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown(f"### {icon_label('folder', 'Import Financial Documents')}", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop your CSV or PDF bank statements here", 
            type=["csv", "pdf"], 
            accept_multiple_files=True, 
            help="Your files are processed strictly inside your local memory scope."
        )
        
        if uploaded_files:
            st.markdown("<p style='font-weight:600; margin-top:1rem;'>Ready for parsing Queue:</p>", unsafe_allow_html=True)
            for f in uploaded_files:
                st.markdown(f"{icon_label('file', f'`{f.name}` ({f.size / 1024:.1f} KB)')}", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(icon_label('analyze', 'Analyze Financial Footprint', '#10B981'), unsafe_allow_html=True)
            if st.button("Analyze Financial Footprint", type="primary", use_container_width=True):
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
                            errors.append(f"Parser failure on {uploaded_file.name}: {str(e)}")
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
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 16px; border: 1px solid #E2E8F0;">
                <h4 style="margin-top:0;">{icon_label('lock', 'Zero-Knowledge Security')}</h4>
                <p style="font-size:0.9rem; color:#475569;">All document ingestion rules occur locally. Only transaction string descriptors are evaluated via secure endpoints for semantic structure analysis.</p>
                <hr style="margin: 1rem 0; border:0; border-top: 1px solid #E2E8F0;">
                <h4 style="margin-top:0;">{icon_label('bank', 'Supported Ecosystems')}</h4>
                <p style="font-size:0.85rem; color:#64748B; line-height:1.6;">
                    • <b>Global Banks:</b> CBA, ANZ, Westpac, NAB, ING, Macquarie, and standard text-based international banking templates.<br>
                    • <b>Formats:</b> UTF-8 Comma-Separated Values (.csv), Structured Text Documents (.pdf)
                </p>
            </div>
        """, unsafe_allow_html=True)

# ── Stage 2: Categorizing ─────────────────────────────────────────────────────
elif st.session_state.stage == "categorizing":
    st.markdown(f"### {icon_label('bot', 'Semantic Pipeline Ingestion')}", unsafe_allow_html=True)
    df = st.session_state.transactions
    total = len(df)
    already_mapped = df["category"].notna().sum() if "category" in df.columns else 0
    
    st.markdown(
        f"<div class='icon-label' style='padding:0.75rem 1rem; background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;'>"
        f"{svg_icon('chart', 18, '#2563EB')}<span>Loaded {total} transactions tracking framework. Memory Engine matched: {already_mapped} records.</span></div>",
        unsafe_allow_html=True,
    )
    
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
            st.markdown(icon_label('back', 'Clear Sandbox'), unsafe_allow_html=True)
            if st.button("Clear Sandbox"):
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
    
    st.markdown(f"### {icon_label('message', 'Machine Learning Context Fine-Tuning')}", unsafe_allow_html=True)
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
                <span class="icon-label" style="text-transform: uppercase; font-size: 0.85rem; letter-spacing: 2px; color: #94A3B8;">
                    {icon_label('repeat' if is_rec else 'pin', 'Recurring Target Sequence Detected' if is_rec else 'Isolated Outlier Point', '#94A3B8', 16)}
                </span>
                <div class="txn-desc">{question['description']}</div>
                <h2 style="{amt_style} margin:0; font-size:2.5rem;">{amt_str}</h2>
                <p style="color: #64748B; margin-top: 0.5rem; font-size:0.9rem;">
                    { f"Occurred {question['count']} times across matching historical intervals." if is_rec else f"Settlement Date: {question['date']}" }
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # UI Overhaul: Smart Recommendations Layout
        st.markdown(f"#### {icon_label('pin', 'Assign Domain Space Label')}", unsafe_allow_html=True)
        
        cat_icons = {
            "income": "wallet", "groceries": "file", "dining": "sparkles", "transport": "pin",
            "utilities": "chart", "rent": "bank", "entertainment": "sparkles", "health": "sparkles",
            "shopping": "folder", "savings": "bank", "investment": "chart", "travel": "pin",
            "education": "file", "other": "file"
        }
        
        # Hardcoded dynamic prioritization mock example: emphasize top categories
        top_picks = ["groceries", "dining", "shopping", "other"]
        
        st.markdown(f"##### {icon_label('sparkles', 'Top Predicted Recommendations')}", unsafe_allow_html=True)
        rec_cols = st.columns(4)
        for idx, cat in enumerate(top_picks):
            with rec_cols[idx]:
                st.markdown(icon_label(cat_icons.get(cat, 'generic'), cat.title()), unsafe_allow_html=True)
                if st.button(cat.title(), key=f"rec_{cat}", type="primary", use_container_width=True):
                    submit_answer(state, question["description"], cat)
                    st.session_state.clarif_state = state
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(icon_label('folder', 'View Alternative Organizational Categories'), unsafe_allow_html=True)
        with st.expander("View Alternative Organizational Categories"):
            alt_cols = st.columns(4)
            alt_cats = [c for c in CATEGORIES if c not in top_picks]
            for idx, cat in enumerate(alt_cats):
                with alt_cols[idx % 4]:
                    st.markdown(icon_label(cat_icons.get(cat, 'generic'), cat.title()), unsafe_allow_html=True)
                    if st.button(cat.title(), key=f"alt_{cat}", use_container_width=True):
                        submit_answer(state, question["description"], cat)
                        st.session_state.clarif_state = state
                        st.rerun()
                        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(icon_label('skip', "Assign to Default 'Other'"), unsafe_allow_html=True)
            if st.button("Assign to Default 'Other'", use_container_width=True):
                skip_answer(state, question["description"])
                st.session_state.clarif_state = state
                st.rerun()
        with col2:
            st.markdown(icon_label('skip', 'Bypass Evaluation Strategy (Default Remaining)'), unsafe_allow_html=True)
            if st.button("Bypass Evaluation Strategy (Default Remaining)", use_container_width=True):
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
            <h1 style="margin: 0; font-size: 2.6rem; letter-spacing:-0.5px; font-weight:800;">{icon_label('chart', 'Asset Architecture Analytics', '#38BDF8', 30)}</h1>
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
        st.markdown(f"#### {icon_label('chart', 'Expenditure Allocation Matrix')}", unsafe_allow_html=True)
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
        st.markdown(f"#### {icon_label('chart', 'Micro-Interval Velocity Trend (Income vs Expenses)')}", unsafe_allow_html=True)
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
        st.markdown(f"#### {icon_label('wallet', 'Primary Capital Outflow Sinks')}", unsafe_allow_html=True)
        cat_totals_df = cat_totals.reset_index()
        cat_totals_df.columns = ["Category", "Amount"]
        fig_bar2 = px.bar(
            cat_totals_df, x="Amount", y="Category", orientation="h", 
            color="Amount", color_continuous_scale="Blues", text_auto=".2s"
        )
        fig_bar2.update_layout(showlegend=False, margin=dict(t=15, b=15, l=15, r=15), xaxis_tickprefix="$", coloraxis_showscale=False, template="plotly_white")
        st.plotly_chart(fig_bar2, use_container_width=True)
        
    with chart_col4:
        st.markdown(f"#### {icon_label('chart', 'Rolling Net Structural Savings Trend')}", unsafe_allow_html=True)
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
    st.markdown(f"#### {icon_label('file', 'Historical Granular Audit Trail')}", unsafe_allow_html=True)
    
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
