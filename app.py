import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hidden Gem Finder", layout="wide")

@st.cache_data
def load_data():

    # Load player performance data
    prem_stats = pd.read_csv("premier_league_players.csv")
    laliga_stats = pd.read_csv("league-players.csv")

    # Load player info
    prem_info = pd.read_csv("premier_league_player_info.csv")
    laliga_info = pd.read_csv("laliga_player_info.csv")

    # Load transfer values
    values = pd.read_csv("transfermarkt_player_values.csv")

    # Merge leagues
    prem = prem_stats.merge(prem_info, on="player", how="left")
    laliga = laliga_stats.merge(laliga_info, on="player", how="left")

    prem["league"] = "Premier League"
    laliga["league"] = "La Liga"

    df = pd.concat([prem, laliga], ignore_index=True)

    # Merge values
    df = df.merge(values, on="player", how="left")

    # Fill missing values
    df["market_value"] = df["market_value"].fillna(df["market_value"].median())

    # Hidden gem score
    df["hidden_gem_score"] = (
        df["goals_per90"] * 2
        + df["assists_per90"] * 2
        + df["progressive_passes_per90"]
    ) / (df["market_value"] + 1)

    return df


df = load_data()

st.title("⚽ Hidden Gem Finder")

league = st.selectbox("Select League", df["league"].unique())

filtered = df[df["league"] == league]

st.dataframe(
    filtered.sort_values("hidden_gem_score", ascending=False)[
        ["player", "team", "age", "market_value", "hidden_gem_score"]
    ],
    use_container_width=True
)

st.subheader("Top Hidden Gems")

top = filtered.sort_values("hidden_gem_score", ascending=False).head(15)

st.bar_chart(top.set_index("player")["hidden_gem_score"])
