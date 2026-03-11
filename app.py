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


@st.cache_data
def load_data():
    # -----------------------------
    # LOAD STATS FILES
    # -----------------------------
    pl_stats = read_csv_safe("league-players.csv", sep=";")
    laliga_stats = read_csv_safe("league-players (1).csv", sep=";")

    for df in [pl_stats, laliga_stats]:
        df.columns = (
            df.columns.astype(str)
            .str.replace('"', "", regex=False)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )
        if "a" in df.columns:
            df.rename(columns={"a": "assists"}, inplace=True)

    pl_stats["league"] = "Premier League"
    laliga_stats["league"] = "La Liga"
    stats = pd.concat([pl_stats, laliga_stats], ignore_index=True)

    # -----------------------------
    # LOAD PLAYER INFO FILES
    # real header is row 2, so use header=1
    # -----------------------------
    pl_info = read_csv_safe("premier_league_player_info.csv", header=1)
    laliga_info = read_csv_safe("laliga_player_info.csv", header=1)
    tm = read_csv_safe("transfermarkt_player_values.csv")

    # Clean info headers
    pl_info.columns = (
        pl_info.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    laliga_info.columns = (
        laliga_info.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    tm.columns = (
        tm.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    # Rename player info columns
    pl_info = pl_info.rename(columns={
        "player": "player",
        "squad": "team",
        "pos": "position",
        "nation": "nationality",
        "age": "age",
    })
    pl_info["league"] = "Premier League"

    laliga_info = laliga_info.rename(columns={
        "player": "player",
        "squad": "team",
        "pos": "position",
        "nation": "nationality",
        "age": "age",
    })
    laliga_info["league"] = "La Liga"

    # age like 27-081 -> 27
    for df in [pl_info, laliga_info]:
        if "age" in df.columns:
            df["age"] = df["age"].astype(str).str.split("-").str[0]
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Transfermarkt
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
    pl_info["player_merge"] = pl_info["player"].apply(normalize_name)
    laliga_info["player_merge"] = laliga_info["player"].apply(normalize_name)
    tm["player_merge"] = tm["player"].apply(normalize_name)

    player_info = pd.concat([pl_info, laliga_info], ignore_index=True)

    # -----------------------------
    # MERGE
    # -----------------------------
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

    df["market_value"] = df["market_value"].fillna(df["market_value"].median())

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
st.write("Find undervalued attacking players across the Premier League and La Liga.")

league_options = ["All"] + sorted(df["league"].dropna().unique().tolist())
selected_league = st.selectbox("League", league_options)

min_minutes = st.slider("Minimum Minutes", 0, int(df["min"].max()), 900, 100)
max_age_default = int(df["age"].dropna().max()) if df["age"].notna().any() else 30
max_age = st.slider("Max Age", 16, max(40, max_age_default), min(25, max_age_default))

max_value = st.slider(
    "Max Market Value (€)",
    0,
    int(df["market_value"].max()),
    min(int(df["market_value"].max()), 60000000),
    1000000,
)

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
st.dataframe(filtered[existing_cols].head(25), use_container_width=True)

st.subheader("Top 15 Hidden Gem Scores")
top15 = filtered.head(15).set_index("player")["hidden_gem_score"]
st.bar_chart(top15)
