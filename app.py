import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Multi-League Hidden Gem Finder", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("final_multileague_hidden_gems.csv")


df = load_data()

# Clean display names from build script
if "team_final" in df.columns and "team" not in df.columns:
    df["team"] = df["team_final"]

if "age_final" in df.columns and "age" not in df.columns:
    df["age"] = df["age_final"]

if "position_final" in df.columns and "position" not in df.columns:
    df["position"] = df["position_final"]

st.title("⚽ Multi-League Hidden Gem Finder")
st.write(
    "Find undervalued attacking players across the Premier League and La Liga using xG, xA, production, age, and market value."
)

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("Filters")

league_options = ["All"] + sorted(df["league"].dropna().unique().tolist())
selected_league = st.sidebar.selectbox("League", league_options)

position_options = ["All"] + sorted(df["position"].dropna().astype(str).unique().tolist())
selected_position = st.sidebar.selectbox("Position", position_options)

min_minutes = st.sidebar.slider(
    "Minimum Minutes",
    min_value=0,
    max_value=int(df["min"].max()),
    value=900,
    step=100,
)

max_age = st.sidebar.slider(
    "Max Age",
    min_value=int(df["age"].min()),
    max_value=int(df["age"].max()),
    value=min(25, int(df["age"].max())),
)

max_value = st.sidebar.slider(
    "Max Market Value (€)",
    min_value=0,
    max_value=int(df["market_value"].max()),
    value=min(60000000, int(df["market_value"].max())),
    step=1000000,
)

top_n = st.sidebar.slider("Rows to show", min_value=5, max_value=50, value=25, step=5)

# =====================================================
# FILTER DATA
# =====================================================
filtered = df.copy()

filtered = filtered[
    (filtered["min"] >= min_minutes)
    & (filtered["age"] <= max_age)
    & (filtered["market_value"] <= max_value)
]

if selected_league != "All":
    filtered = filtered[filtered["league"] == selected_league]

if selected_position != "All":
    filtered = filtered[filtered["position"].astype(str) == selected_position]

filtered = filtered.sort_values("hidden_gem_score", ascending=False)

# =====================================================
# KPIs
# =====================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Players Shown", len(filtered))
col2.metric("Top Hidden Gem Score", f"{filtered['hidden_gem_score'].max():.1f}" if len(filtered) else "N/A")
col3.metric("Lowest Market Value", f"€{int(filtered['market_value'].min()):,}" if len(filtered) else "N/A")
col4.metric("Best xG90", f"{filtered['xg90'].max():.2f}" if len(filtered) else "N/A")

# =====================================================
# TABLE
# =====================================================
st.subheader("Top Hidden Gems")

display_cols = [
    "player",
    "team",
    "league",
    "position",
    "age",
    "market_value",
    "goals",
    "assists",
    "xg90",
    "xa90",
    "g_a_per90",
    "performance_score_100",
    "hidden_gem_score",
]

table_df = filtered[display_cols].head(top_n).copy()
table_df["market_value"] = table_df["market_value"].map(lambda x: f"€{int(x):,}")
table_df["performance_score_100"] = table_df["performance_score_100"].round(1)
table_df["hidden_gem_score"] = table_df["hidden_gem_score"].round(1)
table_df["xg90"] = table_df["xg90"].round(2)
table_df["xa90"] = table_df["xa90"].round(2)
table_df["g_a_per90"] = table_df["g_a_per90"].round(2)

st.dataframe(table_df, use_container_width=True)

# =====================================================
# CHARTS
# =====================================================
left, right = st.columns(2)

with left:
    st.subheader("Performance vs Market Value")
    fig1 = px.scatter(
        filtered,
        x="market_value",
        y="performance_score_100",
        color="league",
        size="min",
        hover_name="player",
        hover_data=["team", "age", "position", "xg90", "xa90", "hidden_gem_score"],
        log_x=True,
    )
    fig1.update_layout(xaxis_title="Market Value (€)", yaxis_title="Performance Score")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("xG90 vs xA90")
    fig2 = px.scatter(
        filtered,
        x="xg90",
        y="xa90",
        color="hidden_gem_score",
        size="min",
        hover_name="player",
        hover_data=["team", "league", "market_value", "age"],
    )
    fig2.update_layout(xaxis_title="xG per 90", yaxis_title="xA per 90")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top 10 Hidden Gem Scores")
top10 = filtered.head(10).sort_values("hidden_gem_score")
fig3 = px.bar(
    top10,
    x="hidden_gem_score",
    y="player",
    orientation="h",
    color="league",
    hover_data=["team", "age", "market_value"],
)
fig3.update_layout(xaxis_title="Hidden Gem Score", yaxis_title="")
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# VALUE INEFFICIENCY
# =====================================================
if "value_inefficiency_score" in filtered.columns:
    st.subheader("Best Value Inefficiency")
    ineff = (
        filtered.sort_values("value_inefficiency_score", ascending=False)[
            ["player", "team", "league", "age", "market_value", "performance_score_100", "value_inefficiency_score"]
        ]
        .head(15)
        .copy()
    )
    ineff["market_value"] = ineff["market_value"].map(lambda x: f"€{int(x):,}")
    ineff["performance_score_100"] = ineff["performance_score_100"].round(1)
    ineff["value_inefficiency_score"] = ineff["value_inefficiency_score"].round(3)
    st.dataframe(ineff, use_container_width=True)
