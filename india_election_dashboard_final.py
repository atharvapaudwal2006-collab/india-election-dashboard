import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score



# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Election Analysis",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        background: linear-gradient(135deg, #FF6B00, #E63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label { color: #aaa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .metric-value { color: #fff; font-size: 2rem; font-weight: 700; line-height: 1.1; }
    .metric-sub { color: #FF6B00; font-size: 0.78rem; margin-top: 4px; }

    .winner-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f0f0;
        border-left: 3px solid #FF6B00;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-left: 3px solid #FF6B00;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #c9d1d9;
        line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        background: #1e1e2e !important;
        border: 1px solid #2d2d4e !important;
        color: #aaa !important;
        padding: 6px 18px !important;
        font-size: 0.88rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B00, #E63946) !important;
        color: white !important;
        border: none !important;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    .party-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

# ── Party colour map ─────────────────────────────────────────────────────────
PARTY_COLORS = {
    "BJP": "#FF6B00", "INC": "#00A651", "TMC": "#3A0CA3",
    "SP": "#E63946",  "BSP": "#00B4D8", "AAP": "#0070C0",
    "DMK": "#E9C46A", "AIADMK": "#EF476F", "JDU": "#F4A261",
    "RJD": "#FB5607", "NCP": "#6A0572", "SHS": "#E07A5F",
    "BJD": "#2D6A4F", "TDP": "#FFB703", "YSRCP": "#A8DADC",
    "BRS": "#9B2226", "JMM": "#3A86FF", "CPM": "#D62828",
    "SAD": "#90E0EF", "JD": "#A9DEF9", "OTHER": "#6C757D",
}

ALLIANCE_COLORS = {
    "NDA": "#FF6B00", "UPA": "#00A651", "INDIA": "#0070C0",
    "NF": "#E9C46A", "LF": "#D62828", "OTHER": "#6C757D",
    "INC+": "#00A651", "INC(S)": "#2DC653",
}

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    overview     = pd.read_csv(os.path.join(base, "elections_overview.csv"))
    party        = pd.read_csv(os.path.join(base, "party_results.csv"))
    state        = pd.read_csv(os.path.join(base, "state_results.csv"))
    constituency = pd.read_csv(os.path.join(base, "constituency_data.csv"))
    demographics = pd.read_csv(os.path.join(base, "demographics_data.csv"))
    alliance     = pd.read_csv(os.path.join(base, "alliance_data.csv"))
    return overview, party, state, constituency, demographics, alliance

overview_df, party_df, state_df, const_df, demo_df, alliance_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗳️ India Elections")
    st.markdown("---")
    all_years = sorted(overview_df["year"].unique(), reverse=True)
    selected_year = st.selectbox("📅 Election Year", all_years, index=0)
    st.markdown("---")
    st.markdown("### 📊 Quick Facts")
    yr = overview_df[overview_df["year"] == selected_year].iloc[0]
    st.markdown(f"**Winner:** {yr['winner_party']}")
    st.markdown(f"**Seats won:** {yr['winner_seats']} / {yr['total_seats']}")
    st.markdown(f"**Turnout:** {yr['turnout_pct']}%")
    st.markdown(f"**PM:** {yr['pm_name']}")
    st.markdown(f"**Coalition:** {'Yes' if yr['coalition_govt']=='Yes' else 'No'}")
    st.markdown("---")
    st.markdown("### 🎨 Legend")
    for party, color in list(PARTY_COLORS.items())[:8]:
        st.markdown(
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:10px;font-size:0.75rem;font-weight:600;">{party}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Data: Election Commission of India  \nVisualized with Streamlit + Plotly")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🗳️ India Election Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Lok Sabha Elections 1984 – 2024 · Interactive Analysis Platform</div>', unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
yr = overview_df[overview_df["year"] == selected_year].iloc[0]
yr_party = party_df[party_df["year"] == selected_year].copy()
yr_state  = state_df[state_df["year"]  == selected_year].copy()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("🏆 Winner", yr["winner_party"], f"{yr['winner_seats']} seats")
with kpi2:
    majority = 272
    gap = yr["winner_seats"] - majority
    st.metric("🎯 Majority Mark", "272 seats", f"{'+' if gap>=0 else ''}{gap} vs winner")
with kpi3:
    st.metric("📊 Voter Turnout", f"{yr['turnout_pct']}%")
with kpi4:
    st.metric("🗳️ Votes Cast", f"{yr['votes_cast_millions']}M")
with kpi5:
    st.metric("👥 Registered Voters", f"{yr['registered_voters_millions']}M")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🏛️ Party Performance",
    "🗺️ State Analysis",
    "📈 Trends & Turnout",
    "⚔️ Head-to-Head",
    "🤝 Alliance Battle",
    "🏟️ Constituencies",
    "👥 Demographics",
    "🔮 Insights",
    "🤖 Prediction",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Seat Distribution</div>', unsafe_allow_html=True)
        yr_party_sorted = yr_party.sort_values("seats_won", ascending=False)
        fig_seats = px.bar(
            yr_party_sorted, x="party", y="seats_won",
            color="party",
            color_discrete_map=PARTY_COLORS,
            text="seats_won",
            title=f"Seats Won by Party — {selected_year}",
            template="plotly_dark",
        )
        fig_seats.update_traces(textposition="outside", textfont_size=11)
        fig_seats.update_layout(
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="Seats Won",
            title_font_size=14,
            shapes=[dict(type="line", x0=-0.5, x1=len(yr_party_sorted)-0.5,
                         y0=272, y1=272, line=dict(color="#FF6B00", dash="dash", width=2))],
            annotations=[dict(x=len(yr_party_sorted)-1, y=272, text="Majority 272",
                              showarrow=False, font=dict(color="#FF6B00", size=10))],
        )
        st.plotly_chart(fig_seats, width="stretch")

    with col_r:
        st.markdown('<div class="section-header">Vote Share Breakdown</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            yr_party_sorted, values="vote_share_pct", names="party",
            color="party", color_discrete_map=PARTY_COLORS,
            hole=0.55,
            title=f"Vote Share % — {selected_year}",
            template="plotly_dark",
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        fig_pie.update_layout(
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", title_font_size=14,
        )
        st.plotly_chart(fig_pie, width="stretch")

    # Seats-vs-votes efficiency
    st.markdown('<div class="section-header">Seat Efficiency: Votes Cast vs Seats Won</div>', unsafe_allow_html=True)
    yr_party["efficiency"] = (yr_party["seats_won"] / yr_party["seats_contested"] * 100).round(1)
    fig_eff = px.scatter(
        yr_party[yr_party["seats_contested"] > 0],
        x="vote_share_pct", y="seats_won",
        size="seats_won", color="party",
        color_discrete_map=PARTY_COLORS,
        text="party", hover_data=["seats_contested", "efficiency"],
        title=f"Vote Share vs Seats Won — {selected_year}",
        template="plotly_dark",
        labels={"vote_share_pct": "Vote Share (%)", "seats_won": "Seats Won"},
    )
    fig_eff.update_traces(textposition="top center")
    fig_eff.update_layout(
        showlegend=False, plot_bgcolor="rgba(13,13,25,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_eff, width="stretch")

    # Summary table
    st.markdown('<div class="section-header">Complete Results Table</div>', unsafe_allow_html=True)
    disp = yr_party[["party","seats_won","vote_share_pct","seats_contested","alliance"]].copy()
    disp.columns = ["Party","Seats Won","Vote Share (%)","Seats Contested","Alliance"]
    disp = disp.sort_values("Seats Won", ascending=False).reset_index(drop=True)
    st.dataframe(disp, width="stretch", hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — PARTY PERFORMANCE
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    all_parties = sorted(party_df["party"].unique())
    sel_parties = st.multiselect(
        "Select parties to compare",
        all_parties,
        default=["BJP", "INC", "TMC", "SP"],
    )
    if not sel_parties:
        st.warning("Please select at least one party.")
    else:
        filtered = party_df[party_df["party"].isin(sel_parties)]

        col1, col2 = st.columns(2)
        with col1:
            fig_seats_trend = px.line(
                filtered, x="year", y="seats_won",
                color="party", color_discrete_map=PARTY_COLORS,
                markers=True,
                title="Seats Won Over Time",
                template="plotly_dark",
                labels={"seats_won": "Seats Won", "year": "Election Year"},
            )
            fig_seats_trend.update_layout(
                plot_bgcolor="rgba(13,13,25,0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title="Party",
            )
            fig_seats_trend.add_hline(y=272, line_dash="dash", line_color="#FF6B00",
                                      annotation_text="Majority 272")
            st.plotly_chart(fig_seats_trend, width="stretch")

        with col2:
            fig_vs_trend = px.line(
                filtered, x="year", y="vote_share_pct",
                color="party", color_discrete_map=PARTY_COLORS,
                markers=True,
                title="Vote Share (%) Over Time",
                template="plotly_dark",
                labels={"vote_share_pct": "Vote Share (%)", "year": "Election Year"},
            )
            fig_vs_trend.update_layout(
                plot_bgcolor="rgba(13,13,25,0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_vs_trend, width="stretch")

        # Stacked area — seats
        st.markdown('<div class="section-header">Proportional Seat Share Over Time (Stacked Area)</div>', unsafe_allow_html=True)
        pivot = filtered.pivot_table(index="year", columns="party", values="seats_won", fill_value=0).reset_index()
        fig_area = go.Figure()
        for p in sel_parties:
            if p in pivot.columns:
                fig_area.add_trace(go.Scatter(
                    x=pivot["year"], y=pivot[p],
                    name=p, mode="lines",
                    fill="tonexty",
                    line=dict(color=PARTY_COLORS.get(p, "#888"), width=1.5),
                    fillcolor=(
            f"rgba({int(PARTY_COLORS.get(p, "#888888")[1:3], 16)},"
            f"{int(PARTY_COLORS.get(p, "#888888")[3:5], 16)},"
            f"{int(PARTY_COLORS.get(p, "#888888")[5:7], 16)},0.33)"
        ),
                ))
        fig_area.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(13,13,25,0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Year", yaxis_title="Seats",
            legend_title="Party",
        )
        st.plotly_chart(fig_area, width="stretch")

        # Strike rate
        st.markdown('<div class="section-header">Win Strike Rate (Seats Won / Seats Contested)</div>', unsafe_allow_html=True)
        filtered2 = filtered[filtered["seats_contested"] > 0].copy()
        filtered2["strike_rate"] = (filtered2["seats_won"] / filtered2["seats_contested"] * 100).round(1)
        fig_strike = px.bar(
            filtered2, x="year", y="strike_rate",
            color="party", barmode="group",
            color_discrete_map=PARTY_COLORS,
            text="strike_rate",
            title="Win Strike Rate (%) by Party",
            template="plotly_dark",
            labels={"strike_rate": "Strike Rate (%)", "year": "Year"},
        )
        fig_strike.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_strike.update_layout(plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_strike, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — STATE ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tabs[2]:
    state_year = st.selectbox("Select Year", sorted(state_df["year"].unique(), reverse=True), key="state_yr")
    yr_state = state_df[state_df["year"] == state_year].copy()

    # Dominant party per state
    party_cols = [c for c in yr_state.columns if c not in
                  ["year","state","total_seats","turnout_pct","winning_party"]]
    yr_state["dominant_party"] = yr_state[party_cols].idxmax(axis=1)
    yr_state["dominant_seats"] = yr_state[party_cols].max(axis=1)
    yr_state["dom_share"] = (yr_state["dominant_seats"] / yr_state["total_seats"] * 100).round(1)

    col_map, col_list = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="section-header">Winning Party by State</div>', unsafe_allow_html=True)
        yr_state_sorted = yr_state.sort_values("dominant_seats", ascending=True)
        fig_state_bar = px.bar(
            yr_state_sorted,
            x="dominant_seats", y="state",
            color="dominant_party",
            color_discrete_map=PARTY_COLORS,
            text="dominant_seats",
            title=f"Dominant Party by State — {state_year}",
            template="plotly_dark",
            orientation="h",
            labels={"dominant_seats": "Seats Won (by dominant party)", "state": "State"},
        )
        fig_state_bar.update_traces(textposition="outside")
        fig_state_bar.update_layout(
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            height=600, showlegend=True,
        )
        st.plotly_chart(fig_state_bar, width="stretch")

    with col_list:
        st.markdown('<div class="section-header">State Turnout Ranking</div>', unsafe_allow_html=True)
        turnout_sorted = yr_state.sort_values("turnout_pct", ascending=False)
        fig_turnout = px.bar(
            turnout_sorted, x="turnout_pct", y="state",
            orientation="h",
            color="turnout_pct",
            color_continuous_scale=["#1a1a2e", "#FF6B00", "#FFD700"],
            title=f"Voter Turnout by State — {state_year}",
            template="plotly_dark",
            text="turnout_pct",
            labels={"turnout_pct": "Turnout (%)", "state": ""},
        )
        fig_turnout.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_turnout.update_layout(
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            height=600, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_turnout, width="stretch")

    # Drill-down
    st.markdown('<div class="section-header">State Deep Dive</div>', unsafe_allow_html=True)
    selected_state = st.selectbox("Choose a State", sorted(yr_state["state"].unique()))
    s_row = yr_state[yr_state["state"] == selected_state].iloc[0]

    s_party_data = {c: s_row[c] for c in party_cols if s_row[c] > 0}
    s_party_df = pd.DataFrame(list(s_party_data.items()), columns=["Party","Seats"]).sort_values("Seats", ascending=False)

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Total Seats", int(s_row["total_seats"]))
    dc2.metric("Turnout", f"{s_row['turnout_pct']}%")
    dc3.metric("Dominant Party", s_row["dominant_party"], f"{int(s_row['dominant_seats'])} seats")

    scol1, scol2 = st.columns(2)
    with scol1:
        fig_s = px.bar(
            s_party_df, x="Party", y="Seats",
            color="Party", color_discrete_map=PARTY_COLORS,
            text="Seats",
            title=f"{selected_state} — Seats by Party ({state_year})",
            template="plotly_dark",
        )
        fig_s.update_traces(textposition="outside")
        fig_s.update_layout(
            showlegend=False, plot_bgcolor="rgba(13,13,25,0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_s, width="stretch")

    with scol2:
        fig_s2 = px.pie(
            s_party_df, values="Seats", names="Party",
            color="Party", color_discrete_map=PARTY_COLORS,
            hole=0.5,
            title=f"{selected_state} — Seat Share ({state_year})",
            template="plotly_dark",
        )
        fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s2, width="stretch")

    # Compare state across years
    state_history = state_df[state_df["state"] == selected_state].copy()
    if len(state_history) > 1:
        st.markdown(f'<div class="section-header">{selected_state} — Historical Winning Party</div>', unsafe_allow_html=True)
        h_cols = [c for c in state_history.columns if c not in ["year","state","total_seats","turnout_pct","winning_party"]]
        hist_melt = state_history.melt(id_vars=["year"], value_vars=h_cols, var_name="Party", value_name="Seats")
        hist_melt = hist_melt[hist_melt["Seats"] > 0]
        fig_hist = px.bar(
            hist_melt, x="year", y="Seats", color="Party",
            color_discrete_map=PARTY_COLORS,
            barmode="stack",
            title=f"{selected_state} — Seat Distribution Over Years",
            template="plotly_dark",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS & TURNOUT
# ════════════════════════════════════════════════════════════════════
with tabs[3]:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown('<div class="section-header">National Voter Turnout Trend</div>', unsafe_allow_html=True)
        fig_turn = px.area(
            overview_df, x="year", y="turnout_pct",
            title="National Voter Turnout % (1984–2024)",
            template="plotly_dark",
            markers=True,
            color_discrete_sequence=["#FF6B00"],
            labels={"turnout_pct": "Turnout (%)", "year": "Election Year"},
        )
        fig_turn.update_traces(fill="tozeroy", fillcolor="rgba(255,107,0,0.2)")
        fig_turn.update_layout(
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_turn, width="stretch")

    with col_t2:
        st.markdown('<div class="section-header">Registered Voters vs Votes Cast</div>', unsafe_allow_html=True)
        fig_voters = go.Figure()
        fig_voters.add_trace(go.Bar(
            x=overview_df["year"], y=overview_df["registered_voters_millions"],
            name="Registered Voters", marker_color="#3A0CA3",
        ))
        fig_voters.add_trace(go.Bar(
            x=overview_df["year"], y=overview_df["votes_cast_millions"],
            name="Votes Cast", marker_color="#FF6B00",
        ))
        fig_voters.update_layout(
            barmode="group", template="plotly_dark",
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Election Year", yaxis_title="Millions",
            title="Electorate Size Growth",
        )
        st.plotly_chart(fig_voters, width="stretch")

    # Electorate growth
    st.markdown('<div class="section-header">Electorate Growth Rate</div>', unsafe_allow_html=True)
    ov = overview_df.sort_values("year").copy()
    ov["growth_pct"] = ov["registered_voters_millions"].pct_change() * 100
    fig_growth = px.bar(
        ov.dropna(), x="year", y="growth_pct",
        title="Registered Voter Growth Rate (%)",
        template="plotly_dark",
        color="growth_pct",
        color_continuous_scale=["#1a1a2e","#FF6B00","#FFD700"],
        text="growth_pct",
        labels={"growth_pct": "Growth (%)", "year": "Year"},
    )
    fig_growth.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_growth.update_layout(
        plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_growth, width="stretch")

    # PM timeline
    st.markdown('<div class="section-header">Prime Ministers & Election Results</div>', unsafe_allow_html=True)
    pm_df = overview_df[["year","pm_name","winner_party","winner_seats","turnout_pct","coalition_govt"]].copy()
    pm_df.columns = ["Year","Prime Minister","Party","Seats","Turnout (%)","Coalition?"]
    st.dataframe(pm_df.sort_values("Year", ascending=False), width="stretch", hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — HEAD TO HEAD
# ════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Head-to-Head Party Comparison</div>', unsafe_allow_html=True)
    hh_col1, hh_col2 = st.columns(2)
    with hh_col1:
        party_a = st.selectbox("Party A", sorted(party_df["party"].unique()), index=0)
    with hh_col2:
        party_b = st.selectbox("Party B", sorted(party_df["party"].unique()), index=1)

    if party_a == party_b:
        st.warning("Please select two different parties.")
    else:
        pa = party_df[party_df["party"] == party_a].set_index("year")
        pb = party_df[party_df["party"] == party_b].set_index("year")
        years_common = sorted(set(pa.index) & set(pb.index))

        # Grouped bar
        fig_hh = go.Figure()
        fig_hh.add_trace(go.Bar(
            x=years_common, y=[pa.loc[y,"seats_won"] for y in years_common],
            name=party_a, marker_color=PARTY_COLORS.get(party_a,"#888"),
        ))
        fig_hh.add_trace(go.Bar(
            x=years_common, y=[pb.loc[y,"seats_won"] for y in years_common],
            name=party_b, marker_color=PARTY_COLORS.get(party_b,"#888"),
        ))
        fig_hh.add_hline(y=272, line_dash="dash", line_color="#fff",
                         annotation_text="Majority 272")
        fig_hh.update_layout(
            barmode="group", template="plotly_dark",
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            title=f"{party_a} vs {party_b} — Seats Won",
            xaxis_title="Year", yaxis_title="Seats",
        )
        st.plotly_chart(fig_hh, width="stretch")

        # Vote share dual-line
        fig_hh2 = go.Figure()
        fig_hh2.add_trace(go.Scatter(
            x=years_common, y=[pa.loc[y,"vote_share_pct"] for y in years_common],
            name=party_a, mode="lines+markers",
            line=dict(color=PARTY_COLORS.get(party_a,"#888"), width=3),
            marker=dict(size=8),
        ))
        fig_hh2.add_trace(go.Scatter(
            x=years_common, y=[pb.loc[y,"vote_share_pct"] for y in years_common],
            name=party_b, mode="lines+markers",
            line=dict(color=PARTY_COLORS.get(party_b,"#888"), width=3, dash="dash"),
            marker=dict(size=8),
        ))
        fig_hh2.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            title=f"{party_a} vs {party_b} — Vote Share (%)",
            xaxis_title="Year", yaxis_title="Vote Share (%)",
        )
        st.plotly_chart(fig_hh2, width="stretch")

        # Score card
        st.markdown('<div class="section-header">All-Time Scorecard</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        a_best = pa["seats_won"].max()
        b_best = pb["seats_won"].max()
        a_wins = (pa.loc[years_common,"seats_won"] > pb.loc[years_common,"seats_won"]).sum()
        b_wins = (pb.loc[years_common,"seats_won"] > pa.loc[years_common,"seats_won"]).sum()
        draws  = len(years_common) - a_wins - b_wins
        sc1.metric(f"🏆 {party_a} Best", f"{a_best} seats",
                   delta=f"{pa['seats_won'].idxmax()}")
        sc2.metric(f"🏆 {party_b} Best", f"{b_best} seats",
                   delta=f"{pb['seats_won'].idxmax()}")
        sc3.metric("Elections compared", len(years_common),
                   delta=f"{party_a} ahead: {a_wins} | {party_b}: {b_wins} | Tied: {draws}")

# ════════════════════════════════════════════════════════════════════
# TAB 6 — ALLIANCE BATTLE
# ════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Alliance-wise Performance</div>', unsafe_allow_html=True)

    al_year = st.selectbox("Year", sorted(alliance_df["year"].unique(), reverse=True), key="al_yr")
    al_yr = alliance_df[alliance_df["year"] == al_year]

    ac1, ac2 = st.columns(2)
    with ac1:
        fig_al = px.bar(
            al_yr.sort_values("total_seats", ascending=False),
            x="alliance", y="total_seats",
            color="alliance", color_discrete_map=ALLIANCE_COLORS,
            text="total_seats",
            title=f"Alliance Seat Tally — {al_year}",
            template="plotly_dark",
        )
        fig_al.add_hline(y=272, line_dash="dash", line_color="#FF6B00",
                         annotation_text="Majority 272")
        fig_al.update_traces(textposition="outside")
        fig_al.update_layout(
            showlegend=False, plot_bgcolor="rgba(13,13,25,0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_al, width="stretch")

    with ac2:
        fig_al2 = px.pie(
            al_yr, values="vote_share_pct", names="alliance",
            color="alliance", color_discrete_map=ALLIANCE_COLORS,
            hole=0.5,
            title=f"Alliance Vote Share — {al_year}",
            template="plotly_dark",
        )
        fig_al2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_al2, width="stretch")

    st.markdown('<div class="section-header">Alliance Strength Over Time</div>', unsafe_allow_html=True)
    top_alliances = ["NDA","UPA","INDIA"]
    al_filtered = alliance_df[alliance_df["alliance"].isin(top_alliances)]
    fig_al_trend = px.line(
        al_filtered, x="year", y="total_seats",
        color="alliance", color_discrete_map=ALLIANCE_COLORS,
        markers=True,
        title="NDA vs UPA/INDIA — Seats Over Time",
        template="plotly_dark",
        labels={"total_seats":"Seats","year":"Election Year"},
    )
    fig_al_trend.add_hline(y=272, line_dash="dash", line_color="#fff",
                           annotation_text="Majority 272")
    fig_al_trend.update_layout(
        plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_al_trend, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 7 — CONSTITUENCIES
# ════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Constituency-level Results</div>', unsafe_allow_html=True)

    c_year = st.selectbox("Year", sorted(const_df["year"].unique(), reverse=True), key="c_yr")
    c_state_filter = st.selectbox(
        "Filter by State (optional)", ["All"] + sorted(const_df["state"].unique()))

    c_df = const_df[const_df["year"] == c_year]
    if c_state_filter != "All":
        c_df = c_df[c_df["state"] == c_state_filter]

    if c_df.empty:
        st.info("No constituency data available for this selection.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Constituencies shown", len(c_df))
        cc2.metric("Avg Turnout", f"{c_df['turnout_pct'].mean():.1f}%")
        cc3.metric("Avg Margin", f"{c_df['margin'].mean():,.0f} votes")

        # Margin chart
        fig_margin = px.bar(
            c_df.sort_values("margin", ascending=False),
            x="constituency", y="margin",
            color="winner_party", color_discrete_map=PARTY_COLORS,
            title=f"Winning Margins — {c_year}",
            template="plotly_dark",
            text="margin",
            labels={"margin":"Margin (votes)","constituency":"Constituency"},
        )
        fig_margin.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_margin.update_layout(
            plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_margin, width="stretch")

        # Table
        disp_c = c_df[["constituency","state","winner_name","winner_party",
                        "winner_votes","runner_up_party","runner_up_votes","margin","turnout_pct"]].copy()
        disp_c.columns = ["Constituency","State","Winner","Party",
                          "Winner Votes","Runner-up Party","Runner-up Votes","Margin","Turnout (%)"]
        st.dataframe(disp_c.reset_index(drop=True), width="stretch", hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 8 — DEMOGRAPHICS
# ════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">Demographic Profile of Parliament</div>', unsafe_allow_html=True)

    d_year = st.selectbox("Year", sorted(demo_df["year"].unique(), reverse=True), key="d_yr")
    d_df = demo_df[demo_df["year"] == d_year]

    dm1, dm2, dm3, dm4 = st.columns(4)
    for col, row in zip([dm1,dm2,dm3,dm4], d_df.itertuples()):
        col.metric(row.category, f"{row.seats_won}",
                   delta=f"{row.vote_share_pct}%")

    # Women representation trend
    st.markdown('<div class="section-header">Women in Parliament — Historical Trend</div>', unsafe_allow_html=True)
    women_df = demo_df[demo_df["category"] == "Women candidates elected"]
    fig_women = px.bar(
        women_df, x="year", y="seats_won",
        text="seats_won",
        title="Women MPs Elected (Lok Sabha)",
        template="plotly_dark",
        color_discrete_sequence=["#E63946"],
        labels={"seats_won":"Women MPs","year":"Year"},
    )
    fig_women.update_traces(textposition="outside")
    fig_women.update_layout(
        plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_women, width="stretch")

    # First-time MPs trend
    st.markdown('<div class="section-header">First-time MPs & Incumbents Re-elected</div>', unsafe_allow_html=True)
    fresh_df  = demo_df[demo_df["category"] == "First-time MPs"]
    inc_r_df  = demo_df[demo_df["category"] == "Incumbent MPs re-elected"]
    fig_fresh = go.Figure()
    fig_fresh.add_trace(go.Bar(
        x=fresh_df["year"], y=fresh_df["seats_won"],
        name="First-time MPs", marker_color="#0070C0",
    ))
    fig_fresh.add_trace(go.Bar(
        x=inc_r_df["year"], y=inc_r_df["seats_won"],
        name="Incumbents Re-elected", marker_color="#FF6B00",
    ))
    fig_fresh.update_layout(
        barmode="group", template="plotly_dark",
        plot_bgcolor="rgba(13,13,25,0.8)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Year", yaxis_title="Number of MPs",
    )
    st.plotly_chart(fig_fresh, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 9 — INSIGHTS
# ════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">🔮 Key Insights & Analysis</div>', unsafe_allow_html=True)

    # Auto-generate insights from data
    yr_data = overview_df[overview_df["year"] == selected_year].iloc[0]
    prev_years = overview_df[overview_df["year"] < selected_year].sort_values("year", ascending=False)
    highest_turn = overview_df.loc[overview_df["turnout_pct"].idxmax()]
    lowest_turn  = overview_df.loc[overview_df["turnout_pct"].idxmin()]

    bjp_hist = party_df[party_df["party"] == "BJP"].sort_values("year")
    inc_hist  = party_df[party_df["party"] == "INC"].sort_values("year")

    insights = [
        f"📈 **{selected_year} turnout ({yr_data['turnout_pct']}%)** is "
        + ("above" if yr_data["turnout_pct"] > overview_df["turnout_pct"].mean() else "below")
        + f" the all-time average of {overview_df['turnout_pct'].mean():.1f}%.",

        f"🏆 **Highest ever turnout** was {highest_turn['turnout_pct']}% in {int(highest_turn['year'])}, "
        f"and the lowest was {lowest_turn['turnout_pct']}% in {int(lowest_turn['year'])}.",

        f"📉 **INC's decline**: From a peak of 404 seats in 1984 to {inc_hist.iloc[-1]['seats_won']} seats in 2024 — "
        f"a fall of {404 - inc_hist.iloc[-1]['seats_won']} seats over 40 years.",

        f"🚀 **BJP's rise**: From just 2 seats in 1984 to {bjp_hist.iloc[-1]['seats_won']} seats in 2024 — "
        f"one of the most dramatic political ascents in democratic history.",

        f"🗳️ **India's electorate** has grown from 376 million in 1984 to 969 million in 2024 — "
        f"more than doubling in 40 years, making it the world's largest democratic electorate.",

        f"🤝 **Coalition era (1989–2014)**: For 25 years India was governed exclusively by coalitions, "
        f"until BJP secured an outright majority in 2014.",

        f"⚖️ **Seat-vote disconnect**: BJP won 31% of votes in 2014 but 52% of seats — "
        f"illustrating how India's first-past-the-post system amplifies majorities.",

        f"👩 **Women representation**: Women MPs have grown from approximately 42 in 1989 "
        f"to 74 in 2024, though still far from gender parity in parliament.",
    ]

    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Seat Efficiency Heatmap (All Elections)</div>', unsafe_allow_html=True)
    top6 = ["BJP","INC","TMC","SP","BSP","DMK"]
    heat_data = party_df[party_df["party"].isin(top6)].copy()
    heat_data = heat_data[heat_data["seats_contested"] > 0]
    heat_data["efficiency"] = (heat_data["seats_won"] / heat_data["seats_contested"] * 100).round(1)
    heat_pivot = heat_data.pivot_table(index="party", columns="year", values="efficiency", fill_value=0)
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=["#1a1a2e","#FF6B00","#FFD700"],
        title="Win Strike Rate (%) — Seats Won / Seats Contested",
        template="plotly_dark",
        text_auto=".1f",
        labels={"color":"Strike Rate (%)"},
    )
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat, width="stretch")


# ════════════════════════════════════════════════════════════════════
# TAB 10 — PREDICTION (SCIKIT-LEARN)
# ════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-header">🤖 Election Seat Prediction using Scikit-learn</div>', unsafe_allow_html=True)

    st.info(
        "This is a machine-learning demonstration based only on the historical data in this dashboard. "
        "It is not an official election forecast."
    )

    pred_party = st.selectbox(
        "Select a party to predict",
        sorted(party_df["party"].unique()),
        key="prediction_party"
    )

    party_history = party_df[
        party_df["party"] == pred_party
    ][["year", "vote_share_pct", "seats_contested", "seats_won"]].copy()

    party_history = party_history.dropna()
    party_history = party_history[
        party_history["seats_contested"] > 0
    ].sort_values("year")

    if len(party_history) < 5:
        st.warning("Not enough historical data for a reliable model for this party.")
    else:
        # Features used by the ML model.
        X = party_history[["year", "vote_share_pct", "seats_contested"]]
        y = party_history["seats_won"]

        model_type = st.radio(
            "Choose ML model",
            ["Linear Regression", "Random Forest"],
            horizontal=True,
            key="prediction_model"
        )

        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=5,
                min_samples_leaf=1
            )

        # Keep the final historical election as a test point.
        # The model is trained on earlier elections and evaluated on the latest one.
        if len(party_history) >= 6:
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_test = X.iloc[-1:]
            y_test = y.iloc[-1:]

            model.fit(X_train, y_train)
            test_prediction = model.predict(X_test)

            mae = mean_absolute_error(y_test, test_prediction)
            r2 = r2_score(y_test, test_prediction) if len(y_test) > 1 else None

            m1, m2, m3 = st.columns(3)
            m1.metric("Historical elections", len(party_history))
            m2.metric("Test MAE", f"{mae:.1f} seats")
            m3.metric("Latest actual seats", f"{int(y_test.iloc[0])}")

        # Refit on all available historical data before making the future prediction.
        model.fit(X, y)

        latest_year = int(party_history["year"].max())
        next_year = st.number_input(
            "Election year to predict",
            min_value=latest_year + 1,
            max_value=2100,
            value=latest_year + 5,
            step=1,
            key="prediction_year"
        )

        latest_vote_share = float(party_history.iloc[-1]["vote_share_pct"])
        latest_contested = int(party_history.iloc[-1]["seats_contested"])

        pc1, pc2 = st.columns(2)

        with pc1:
            future_vote_share = st.number_input(
                "Expected vote share (%)",
                min_value=0.0,
                max_value=100.0,
                value=latest_vote_share,
                step=0.5,
                key="future_vote_share"
            )

        with pc2:
            future_contested = st.number_input(
                "Seats contested",
                min_value=1,
                max_value=543,
                value=min(latest_contested, 543),
                step=1,
                key="future_contested"
            )

        future_data = pd.DataFrame({
            "year": [next_year],
            "vote_share_pct": [future_vote_share],
            "seats_contested": [future_contested]
        })

        predicted_seats = float(model.predict(future_data)[0])
        predicted_seats = max(0, min(543, predicted_seats))

        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Party", pred_party)
        r2.metric("Predicted Seats", f"{predicted_seats:.0f}")
        r3.metric("Prediction Range", f"0–{future_contested} contested")

        # Historical actual vs fitted values.
        history_plot = party_history.copy()
        history_plot["predicted_seats"] = model.predict(X)

        fig_prediction = go.Figure()

        fig_prediction.add_trace(go.Scatter(
            x=history_plot["year"],
            y=history_plot["seats_won"],
            mode="lines+markers",
            name="Actual Seats",
            line=dict(width=3)
        ))

        fig_prediction.add_trace(go.Scatter(
            x=history_plot["year"],
            y=history_plot["predicted_seats"],
            mode="lines+markers",
            name="Model Prediction",
            line=dict(dash="dash", width=2)
        ))

        fig_prediction.add_trace(go.Scatter(
            x=[next_year],
            y=[predicted_seats],
            mode="markers",
            name="Future Prediction",
            marker=dict(size=14)
        ))

        fig_prediction.add_hline(
            y=272,
            line_dash="dash",
            annotation_text="Majority 272"
        )

        fig_prediction.update_layout(
            template="plotly_dark",
            title=f"{pred_party} — Actual vs ML Predicted Seats",
            xaxis_title="Election Year",
            yaxis_title="Seats Won",
            plot_bgcolor="rgba(13,13,25,0.8)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_prediction, width="stretch")

        st.markdown('<div class="section-header">Model Input & Prediction</div>', unsafe_allow_html=True)

        prediction_table = future_data.copy()
        prediction_table["predicted_seats"] = predicted_seats
        prediction_table.columns = [
            "Election Year",
            "Expected Vote Share (%)",
            "Seats Contested",
            "Predicted Seats"
        ]

        st.dataframe(
            prediction_table,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "The model learns relationships from the historical party data. "
            "Changing the expected "
            ""
            " share or seats contested changes the prediction."
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    📊 India Election Analysis Dashboard &nbsp;|&nbsp;
    Data sourced from Election Commission of India &nbsp;|&nbsp;
    Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)