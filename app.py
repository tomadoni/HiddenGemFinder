import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hidden Gem Finder", layout="wide")


# =====================================================
# HELPERS
# =====================================================
def read_csv_safe(path, sep=",", header="infer"):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, header=header)
        except Exception as e:
            last_error = e
    raise last_error


def normalize_name(value: str) -> str:
    if pd.isna(value):
        return value
    return str(value).strip()


def format_market_value(x):
    if pd.isna(x):
        return ""
    try:
        value_millions = float(x) / 1_000_000
        return f"€{value_millions:.1f}M"
    except Exception:
        return str(x)


def youtube_search_link(player_name):
    query = str(player_name).replace(" ", "+") + "+highlights"
    return f"https://www.youtube.com/results?search_query={query}"


def fbref_search_link(player_name):
    query = str(player_name).replace(" ", "+")
    return f"https://fbref.com/en/search/search.fcgi?search={query}"


def assign_role(position):
    if pd.isna(position):
        return "Other"

    pos = str(position).upper().strip()

    if "FW" in pos and "MF" not in pos:
        return "Striker"

    if "FW,MF" in pos or "MF,FW" in pos:
        return "Winger"

    if pos == "MF":
        return "Attacking Mid"

    return "Other"


def load_stats_file(path, league_name):
    df = read_csv_safe(path, sep=";")
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    if "a" in df.columns:
        df = df.rename(columns={"a": "assists"})

    df["league"] = league_name
    return df


def load_info_file(path, league_name):
    df = read_csv_safe(path, header=1)
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    df = df.rename(
        columns={
            "player": "player",
            "squad": "team",
            "pos": "position",
            "nation": "nationality",
            "age": "age",
        }
    )

    if "age" in df.columns:
        df["age"] = df["age"].astype(str).str.split("-").str[0]
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["league"] = league_name
    return df


def compute_percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


# =====================================================
# LOAD + BUILD DATA
# =====================================================
@st.cache_data
def load_data():
    stats_files = [
        ("league-players.csv", "Premier League"),
        ("league-players (1).csv", "La Liga"),
        ("bundesliga_xg_player.csv", "Bundesliga"),
        ("seriea_xg_player.csv", "Serie A"),
        ("ligue1_xg_player.csv", "Ligue 1"),
    ]

    info_files = [
        ("premier_league_player_info.csv", "Premier League"),
        ("laliga_player_info.csv", "La Liga"),
        ("bundesliga_player_info.csv", "Bundesliga"),
        ("seriea_player_info.csv", "Serie A"),
        ("ligue1_player_info.csv", "Ligue 1"),
    ]

    stats = pd.concat([load_stats_file(path, league) for path, league in stats_files], ignore_index=True)
    player_info = pd.concat([load_info_file(path, league) for path, league in info_files], ignore_index=True)

    tm = read_csv_safe("transfermarkt_player_values.csv")
    tm.columns = (
        tm.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    tm = tm.rename(
        columns={
            "name": "player",
            "league_name": "league",
            "current_value_eur": "market_value",
            "age": "age_tm",
            "nationality": "nationality_tm",
        }
    )

    keep_tm = [c for c in ["player", "league", "market_value", "age_tm", "nationality_tm"] if c in tm.columns]
    tm = tm[keep_tm].copy()

    if "market_value" in tm.columns:
        tm["market_value"] = pd.to_numeric(tm["market_value"], errors="coerce")

    stats["player_merge"] = stats["player"].apply(normalize_name)
    player_info["player_merge"] = player_info["player"].apply(normalize_name)
    tm["player_merge"] = tm["player"].apply(normalize_name)

    df = stats.merge(
        player_info[["player_merge", "league", "team", "position", "nationality", "age"]],
        on=["player_merge", "league"],
        how="left",
        suffixes=("", "_info"),
    )

    if "team_info" in df.columns:
        df["team"] = df["team_info"].fillna(df["team"])

    df = df.merge(
        tm[["player_merge", "league", "market_value"]],
        on=["player_merge", "league"],
        how="left",
    )

    numeric_cols = ["apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90", "age", "market_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["player", "team", "apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90"]).copy()
    df = df[df["min"] >= 900].copy()

    if "position" in df.columns:
        df = df[df["position"].astype(str).str.contains("FW|MF", na=False)].copy()

    if df["market_value"].notna().any():
        df["market_value"] = df["market_value"].fillna(df["market_value"].median())
    else:
        df["market_value"] = 0

    # Features
    df["g_a"] = df["goals"] + df["assists"]
    df["g_a_per90"] = df["g_a"] / (df["min"] / 90)
    df["goals_minus_xg"] = df["goals"] - df["xg"]
    df["assists_minus_xa"] = df["assists"] - df["xa"]

    # Base z-scored performance features
    performance_features = ["xg90", "xa90", "g_a_per90", "goals_minus_xg", "assists_minus_xa"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[performance_features])

    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"{c}_z" for c in performance_features],
        index=df.index,
    )
    df = pd.concat([df, scaled_df], axis=1)

    df["performance_score"] = (
        0.30 * df["xg90_z"]
        + 0.25 * df["xa90_z"]
        + 0.25 * df["g_a_per90_z"]
        + 0.10 * df["goals_minus_xg_z"]
        + 0.10 * df["assists_minus_xa_z"]
    )
    df["performance_score_100"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df[["performance_score"]])

    # Roles
    df["role"] = df["position"].apply(assign_role)

    # Role-specific raw scoring
    df["striker_score_raw"] = (
        0.40 * df["xg90"]
        + 0.25 * df["g_a_per90"]
        + 0.20 * df["goals_minus_xg"]
        + 0.15 * df["goals"]
    )

    df["winger_score_raw"] = (
        0.30 * df["xa90"]
        + 0.25 * df["g_a_per90"]
        + 0.20 * df["assists"]
        + 0.15 * df["xg90"]
        + 0.10 * df["assists_minus_xa"]
    )

    df["att_mid_score_raw"] = (
        0.35 * df["xa90"]
        + 0.25 * df["assists"]
        + 0.20 * df["g_a_per90"]
        + 0.10 * df["xg90"]
        + 0.10 * df["assists_minus_xa"]
    )

    def pick_role_score(row):
        if row["role"] == "Striker":
            return row["striker_score_raw"]
        elif row["role"] == "Winger":
            return row["winger_score_raw"]
        elif row["role"] == "Attacking Mid":
            return row["att_mid_score_raw"]
        return row["performance_score"]

    df["role_score_raw"] = df.apply(pick_role_score, axis=1)

    # League strength adjustment
    league_strength = {
        "Premier League": 1.00,
        "La Liga": 0.96,
        "Bundesliga": 0.93,
        "Serie A": 0.91,
        "Ligue 1": 0.88,
    }
    df["league_strength"] = df["league"].map(league_strength).fillna(0.90)
    df["adjusted_role_score_raw"] = df["role_score_raw"] * df["league_strength"]

    # Age / value / upside
    if df["age"].notna().any():
        age_nonnull = df["age"].fillna(df["age"].median())
        df["age_scaled"] = MinMaxScaler().fit_transform(age_nonnull.to_frame())
        df["age_bonus"] = 1 - df["age_scaled"]
    else:
        df["age_bonus"] = 0.5

    df["value_scaled"] = MinMaxScaler().fit_transform(df[["market_value"]])
    df["value_bonus"] = 1 - df["value_scaled"]

    # Development / upside model
    # Peaks around young productive players
    df["age_for_upside"] = df["age"].fillna(df["age"].median())
    df["development_upside_raw"] = ((24 - df["age_for_upside"]).clip(lower=0) / 6) * df["performance_score_100"]
    df["development_upside_score"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        df[["development_upside_raw"]]
    )

    # Final hidden gem score
    df["role_score_scaled"] = MinMaxScaler().fit_transform(df[["adjusted_role_score_raw"]])

    df["hidden_gem_score_raw"] = (
        0.60 * df["role_score_scaled"]
        + 0.20 * df["value_bonus"]
        + 0.10 * df["age_bonus"]
        + 0.10 * (df["development_upside_score"] / 100)
    )

    max_score = df["hidden_gem_score_raw"].max()
    if max_score == 0:
        df["hidden_gem_score"] = 0.0
    else:
        df["hidden_gem_score"] = ((df["hidden_gem_score_raw"] / max_score) * 99).round(1)

    # Percentiles within role
    percentile_metrics = ["xg90", "xa90", "g_a_per90", "goals_minus_xg", "assists_minus_xa", "hidden_gem_score"]
    for metric in percentile_metrics:
        df[f"{metric}_pct"] = df.groupby("role")[metric].transform(compute_percentile).round(1)

    # Value inefficiency
    safe_market = df["market_value"].replace(0, 1)
    df["value_inefficiency"] = (df["performance_score_100"] / safe_market) * 1_000_000

    return df.sort_values("hidden_gem_score", ascending=False)


# =====================================================
# LOAD DATA
# =====================================================
df = load_data()

# =====================================================
# UI FILTERS
# =====================================================
st.title("⚽ Multi-League Hidden Gem Finder")
st.write("Find undervalued attacking players across Europe’s top leagues.")

col_a, col_b, col_c = st.columns(3)

with col_a:
    league_options = ["All"] + sorted(df["league"].dropna().unique().tolist())
    selected_league = st.selectbox("League", league_options)

with col_b:
    role_options = ["All"] + sorted(df["role"].dropna().unique().tolist())
    selected_role = st.selectbox("Role", role_options)

with col_c:
    target_lists = [
        "Custom",
        "Best U23 Wingers Under €30M",
        "Best U23 Strikers Under €30M",
        "Best U25 Attacking Mids Under €40M",
        "Best Breakout Players Under €20M",
    ]
    selected_target_list = st.selectbox("Recruitment Target List", target_lists)

min_minutes = st.slider("Minimum Minutes", 0, int(df["min"].max()), 900, 100)

max_age_default = int(df["age"].dropna().max()) if df["age"].notna().any() else 30
max_age = st.slider("Max Age", 16, max(40, max_age_default), min(25, max_age_default))

max_value_m = st.slider(
    "Max Market Value (€M)",
    min_value=0,
    max_value=int(df["market_value"].max() / 1_000_000) if len(df) > 0 else 100,
    value=min(int(df["market_value"].max() / 1_000_000), 60) if len(df) > 0 else 60,
    step=1,
)

max_value = max_value_m * 1_000_000

filtered = df.copy()

# Recruitment target presets
if selected_target_list == "Best U23 Wingers Under €30M":
    filtered = filtered[(filtered["role"] == "Winger") & (filtered["age"] <= 23) & (filtered["market_value"] <= 30_000_000)]
elif selected_target_list == "Best U23 Strikers Under €30M":
    filtered = filtered[(filtered["role"] == "Striker") & (filtered["age"] <= 23) & (filtered["market_value"] <= 30_000_000)]
elif selected_target_list == "Best U25 Attacking Mids Under €40M":
    filtered = filtered[(filtered["role"] == "Attacking Mid") & (filtered["age"] <= 25) & (filtered["market_value"] <= 40_000_000)]
elif selected_target_list == "Best Breakout Players Under €20M":
    filtered = filtered[(filtered["age"] <= 23) & (filtered["market_value"] <= 20_000_000)]

filtered = filtered[(filtered["min"] >= min_minutes) & (filtered["market_value"] <= max_value)].copy()

if filtered["age"].notna().any():
    filtered = filtered[filtered["age"].fillna(99) <= max_age]

if selected_league != "All":
    filtered = filtered[filtered["league"] == selected_league]

if selected_role != "All":
    filtered = filtered[filtered["role"] == selected_role]

filtered = filtered.sort_values("hidden_gem_score", ascending=False)

# =====================================================
# KPI ROW
# =====================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Players", len(filtered))
k2.metric("Top Hidden Gem Score", f"{filtered['hidden_gem_score'].max():.1f}" if len(filtered) else "N/A")
k3.metric("Avg Market Value", format_market_value(filtered["market_value"].mean()) if len(filtered) else "N/A")
k4.metric("Avg Age", f"{filtered['age'].mean():.1f}" if len(filtered) and filtered["age"].notna().any() else "N/A")

# =====================================================
# TOP TABLE
# =====================================================
st.subheader("Top Hidden Gems")

show_cols = [
    "player",
    "team",
    "league",
    "position",
    "role",
    "age",
    "market_value",
    "goals",
    "assists",
    "xg90",
    "xa90",
    "g_a_per90",
    "hidden_gem_score",
]
display_df = filtered[show_cols].head(25).copy()

display_df["market_value"] = display_df["market_value"].apply(format_market_value)
display_df["xg90"] = display_df["xg90"].round(2)
display_df["xa90"] = display_df["xa90"].round(2)
display_df["g_a_per90"] = display_df["g_a_per90"].round(2)
display_df["hidden_gem_score"] = display_df["hidden_gem_score"].round(1)

st.dataframe(display_df, use_container_width=True)

# =====================================================
# CHARTS
# =====================================================
left, right = st.columns(2)

with left:
    st.subheader("Top 15 Hidden Gem Scores")
    top15 = filtered.head(15).copy()
    top15["label"] = top15["player"] + " (" + top15["league"] + ")"

    fig = px.bar(
        top15,
        x="hidden_gem_score",
        y="label",
        orientation="h",
        title="Top 15 Hidden Gem Scores",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Performance vs Market Value")
    fig2 = px.scatter(
        filtered,
        x="market_value",
        y="performance_score_100",
        color="league",
        size="min",
        hover_name="player",
        hover_data=["team", "role", "age", "hidden_gem_score"],
        log_x=True,
    )
    fig2.update_layout(xaxis_title="Market Value (€)", yaxis_title="Performance Score")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("xG90 vs xA90")
fig3 = px.scatter(
    filtered,
    x="xg90",
    y="xa90",
    color="role",
    size="min",
    hover_name="player",
    hover_data=["team", "league", "age", "market_value", "hidden_gem_score"],
)
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# PLAYER SIMILARITY SEARCH
# =====================================================
st.subheader("Player Similarity Search")

if len(filtered) > 0:
    player_choice = st.selectbox("Choose a player to find similar players", filtered["player"].dropna().unique().tolist())

    similarity_features = ["xg90", "xa90", "g_a_per90", "goals_minus_xg", "assists_minus_xa", "performance_score_100"]
    sim_df = filtered.dropna(subset=similarity_features).copy()

    if player_choice in sim_df["player"].values:
        scaler = StandardScaler()
        sim_matrix = scaler.fit_transform(sim_df[similarity_features])
        similarity = cosine_similarity(sim_matrix)

        sim_index = sim_df.index[sim_df["player"] == player_choice][0]
        matrix_row = sim_df.index.get_loc(sim_index)

        sim_scores = similarity[matrix_row]
        sim_df = sim_df.copy()
        sim_df["similarity_score"] = sim_scores
        sim_df = sim_df[sim_df["player"] != player_choice]
        sim_df = sim_df.sort_values("similarity_score", ascending=False).head(10)

        sim_display = sim_df[["player", "team", "league", "role", "age", "market_value", "similarity_score"]].copy()
        sim_display["market_value"] = sim_display["market_value"].apply(format_market_value)
        sim_display["similarity_score"] = sim_display["similarity_score"].round(3)
        st.dataframe(sim_display, use_container_width=True)

# =====================================================
# RADAR CHART + PERCENTILES
# =====================================================
st.subheader("Player Radar & Percentiles")

if len(filtered) > 1:
    radar_players = st.multiselect(
        "Select up to 2 players to compare",
        filtered["player"].dropna().unique().tolist(),
        default=filtered["player"].head(2).tolist(),
        max_selections=2,
    )

    radar_metrics = [
        ("xg90_pct", "xG/90"),
        ("xa90_pct", "xA/90"),
        ("g_a_per90_pct", "G+A/90"),
        ("goals_minus_xg_pct", "Goals-xG"),
        ("assists_minus_xa_pct", "Ast-xA"),
        ("hidden_gem_score_pct", "Gem Score"),
    ]

    radar_df = filtered[filtered["player"].isin(radar_players)].copy()

    if len(radar_df) > 0:
        fig_radar = go.Figure()

        categories = [label for _, label in radar_metrics]
        categories_closed = categories + [categories[0]]

        for _, row in radar_df.iterrows():
            values = [row[col] for col, _ in radar_metrics]
            values_closed = values + [values[0]]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    name=row["player"],
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        pct_cols = ["player", "role"] + [col for col, _ in radar_metrics]
        pct_display = radar_df[pct_cols].copy()
        pct_display.columns = ["player", "role", "xG/90 pct", "xA/90 pct", "G+A/90 pct", "Goals-xG pct", "Ast-xA pct", "Gem pct"]
        pct_display.iloc[:, 2:] = pct_display.iloc[:, 2:].round(1)
        st.dataframe(pct_display, use_container_width=True)

# =====================================================
# VIDEO / SCOUTING LINKS
# =====================================================
st.subheader("Scouting Links")

if len(filtered) > 0:
    scouting_player = st.selectbox("Choose a player for scouting links", filtered["player"].dropna().unique().tolist(), key="scouting_links")
    st.markdown(f"[YouTube Highlights]({youtube_search_link(scouting_player)})")
    st.markdown(f"[FBref Search]({fbref_search_link(scouting_player)})")

# =====================================================
# RECRUITMENT TARGET LISTS
# =====================================================
st.subheader("Recruitment Target List Export")

target_df = filtered[
    ["player", "team", "league", "role", "age", "market_value", "xg90", "xa90", "g_a_per90", "hidden_gem_score", "development_upside_score"]
].head(20).copy()

target_df["market_value"] = target_df["market_value"].apply(format_market_value)
target_df["xg90"] = target_df["xg90"].round(2)
target_df["xa90"] = target_df["xa90"].round(2)
target_df["g_a_per90"] = target_df["g_a_per90"].round(2)
target_df["hidden_gem_score"] = target_df["hidden_gem_score"].round(1)
target_df["development_upside_score"] = target_df["development_upside_score"].round(1)

st.dataframe(target_df, use_container_width=True)
