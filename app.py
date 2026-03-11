import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(page_title="Hidden Gem Finder", layout="wide")


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
        return f"€{int(round(float(x))):,}"
    except Exception:
        return str(x)


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
    # These info files have the real header on row 2
    df = read_csv_safe(path, header=1)
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    df = df.rename(columns={
        "player": "player",
        "squad": "team",
        "pos": "position",
        "nation": "nationality",
        "age": "age",
    })

    if "age" in df.columns:
        df["age"] = df["age"].astype(str).str.split("-").str[0]
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["league"] = league_name
    return df


@st.cache_data
def load_data():
    # -----------------------------
    # LOAD STATS FILES
    # -----------------------------
    stats_files = [
        ("league-players.csv", "Premier League"),
        ("league-players (1).csv", "La Liga"),
        ("bundesliga_xg_player.csv", "Bundesliga"),
        ("seriea_xg_player.csv", "Serie A"),
        ("ligue1_xg_player.csv", "Ligue 1"),
    ]

    stats_dfs = []
    for path, league_name in stats_files:
        stats_dfs.append(load_stats_file(path, league_name))

    stats = pd.concat(stats_dfs, ignore_index=True)

    # -----------------------------
    # LOAD INFO FILES
    # -----------------------------
    info_files = [
        ("premier_league_player_info.csv", "Premier League"),
        ("laliga_player_info.csv", "La Liga"),
        ("bundesliga_player_info.csv", "Bundesliga"),
        ("seriea_player_info.csv", "Serie A"),
        ("ligue1_player_info.csv", "Ligue 1"),
    ]

    info_dfs = []
    for path, league_name in info_files:
        info_dfs.append(load_info_file(path, league_name))

    player_info = pd.concat(info_dfs, ignore_index=True)

    # -----------------------------
    # LOAD TRANSFERMARKT FILE
    # -----------------------------
    tm = read_csv_safe("transfermarkt_player_values.csv")
    tm.columns = (
        tm.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    tm = tm.rename(columns={
        "name": "player",
        "league_name": "league",
        "current_value_eur": "market_value",
        "age": "age_tm",
        "nationality": "nationality_tm",
    })

    keep_tm = [c for c in ["player", "league", "market_value", "age_tm", "nationality_tm"] if c in tm.columns]
    tm = tm[keep_tm].copy()

    if "market_value" in tm.columns:
        tm["market_value"] = pd.to_numeric(tm["market_value"], errors="coerce")

    # -----------------------------
    # NORMALIZE NAMES
    # -----------------------------
    stats["player_merge"] = stats["player"].apply(normalize_name)
    player_info["player_merge"] = player_info["player"].apply(normalize_name)
    tm["player_merge"] = tm["player"].apply(normalize_name)

    # -----------------------------
    # MERGE STATS + INFO
    # -----------------------------
    df = stats.merge(
        player_info[["player_merge", "league", "team", "position", "nationality", "age"]],
        on=["player_merge", "league"],
        how="left",
        suffixes=("", "_info"),
    )

    if "team_info" in df.columns:
        df["team"] = df["team_info"].fillna(df["team"])

    # -----------------------------
    # MERGE TRANSFERMARKT VALUES
    # -----------------------------
    df = df.merge(
        tm[["player_merge", "league", "market_value"]],
        on=["player_merge", "league"],
        how="left",
    )

    # -----------------------------
    # CLEAN NUMERIC FIELDS
    # -----------------------------
    numeric_cols = ["apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90", "age", "market_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["player", "team", "apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90"]).copy()
    df = df[df["min"] >= 900].copy()

    if "position" in df.columns:
        df = df[df["position"].astype(str).str.contains("FW|MF", na=False)].copy()

    # Fill missing market values with median so app doesn't break
    if df["market_value"].notna().any():
        df["market_value"] = df["market_value"].fillna(df["market_value"].median())
    else:
        df["market_value"] = 0

    # -----------------------------
    # FEATURES
    # -----------------------------
    df["g_a"] = df["goals"] + df["assists"]
    df["g_a_per90"] = df["g_a"] / (df["min"] / 90)
    df["goals_minus_xg"] = df["goals"] - df["xg"]
    df["assists_minus_xa"] = df["assists"] - df["xa"]

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

    if df["age"].notna().any():
        age_nonnull = df["age"].fillna(df["age"].median())
        df["age_scaled"] = MinMaxScaler().fit_transform(age_nonnull.to_frame())
        df["age_bonus"] = 1 - df["age_scaled"]
    else:
        df["age_bonus"] = 0.5

    df["value_scaled"] = MinMaxScaler().fit_transform(df[["market_value"]])
    df["value_bonus"] = 1 - df["value_scaled"]

    df["performance_scaled"] = MinMaxScaler().fit_transform(df[["performance_score"]])

    df["hidden_gem_score_raw"] = (
        0.70 * df["performance_scaled"]
        + 0.20 * df["value_bonus"]
        + 0.10 * df["age_bonus"]
    )

    df["hidden_gem_score"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        df[["hidden_gem_score_raw"]]
    )

    return df.sort_values("hidden_gem_score", ascending=False)


df = load_data()

st.title("⚽ Multi-League Hidden Gem Finder")
st.write("Find undervalued attacking players across Europe’s top leagues.")

league_options = ["All"] + sorted(df["league"].dropna().unique().tolist())
selected_league = st.selectbox("League", league_options)

min_minutes = st.slider("Minimum Minutes", 0, int(df["min"].max()), 900, 100)

max_age_default = int(df["age"].dropna().max()) if df["age"].notna().any() else 30
max_age = st.slider("Max Age", 16, max(40, max_age_default), min(25, max_age_default))

max_value_m = st.slider(
    "Max Market Value (€M)",
    min_value=0.0,
    max_value=float(df["market_value"].max() / 1_000_000) if len(df) > 0 else 100.0,
    value=min(float(df["market_value"].max() / 1_000_000), 60.0) if len(df) > 0 else 60.0,
    step=1.0,
)

max_value = max_value_m * 1_000_000

filtered = df[(df["min"] >= min_minutes) & (df["market_value"] <= max_value)].copy()

if "age" in filtered.columns and filtered["age"].notna().any():
    filtered = filtered[filtered["age"].fillna(99) <= max_age]

if selected_league != "All":
    filtered = filtered[filtered["league"] == selected_league]

st.subheader("Top Hidden Gems")

show_cols = [
    "player", "team", "league", "position", "age", "market_value",
    "goals", "assists", "xg90", "xa90", "g_a_per90", "hidden_gem_score"
]
existing_cols = [c for c in show_cols if c in filtered.columns]

display_df = filtered[existing_cols].head(25).copy()

if "market_value" in display_df.columns:
    display_df["market_value"] = display_df["market_value"].apply(format_market_value)

if "xg90" in display_df.columns:
    display_df["xg90"] = display_df["xg90"].round(2)

if "xa90" in display_df.columns:
    display_df["xa90"] = display_df["xa90"].round(2)

if "g_a_per90" in display_df.columns:
    display_df["g_a_per90"] = display_df["g_a_per90"].round(2)

if "hidden_gem_score" in display_df.columns:
    display_df["hidden_gem_score"] = display_df["hidden_gem_score"].round(1)

st.dataframe(display_df, use_container_width=True)

st.subheader("Top 15 Hidden Gem Scores")

top15 = filtered.head(15).copy()
top15["label"] = top15["player"] + " (" + top15["league"] + ")"
top15["hidden_gem_score"] = top15["hidden_gem_score"].round(1)

fig = px.bar(
    top15,
    x="hidden_gem_score",
    y="label",
    orientation="h",
    title="Top 15 Hidden Gem Scores"
)

fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)
