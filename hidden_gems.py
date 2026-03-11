import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =====================================================
# HELPERS
# =====================================================
def normalize_name(value: str) -> str:
    """Basic normalization to improve merge quality across files."""
    if pd.isna(value):
        return value

    s = str(value).strip()

    replacements = {
        "Kylian Mbappe": "Kylian Mbappé",
        "Mbappe-Lottin": "Mbappé",
        "Vinicius Junior": "Vinicius Júnior",
        "Julian Alvarez": "Julián Álvarez",
        "Alex Baena": "Álex Baena",
        "Alex Berenguer": "Álex Berenguer",
        "Inaki Williams": "Iñaki Williams",
        "Jose Luis Morales": "José Luis Morales",
        "Jose Luis Gaya": "José Luis Gayà",
        "Martin Odegaard": "Martin Ødegaard",
        "Joao Pedro": "João Pedro",
        "Joao Palhinha": "João Palhinha",
        "Joao Felix": "João Félix",
        "Mikel Oyarzabal Ugarte": "Mikel Oyarzabal",
    }

    return replacements.get(s, s)


def clean_stats(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    df = df.rename(columns={"a": "assists"})

    numeric_cols = ["number", "apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "player" not in df.columns or "team" not in df.columns:
        raise ValueError(f"{league_name} stats file must contain 'player' and 'team' columns.")

    df["league"] = league_name
    df["player_merge"] = df["player"].apply(normalize_name)
    return df


def clean_pl_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    rename_map = {
        "player name": "player",
        "club": "team",
        "position": "position",
        "age": "age",
        "nationality": "nationality",
    }
    df = df.rename(columns=rename_map)

    required = ["player", "team", "position", "age"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"premier_league_player_info.csv is missing columns: {missing}")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["league"] = "Premier League"
    df["player_merge"] = df["player"].apply(normalize_name)

    keep_cols = ["player", "player_merge", "team", "position", "nationality", "age", "league"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep_cols].drop_duplicates(subset=["player_merge", "league"])


def clean_laliga_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    rename_map = {
        "player": "player",
        "squad": "team",
        "pos": "position",
        "nation": "nationality",
        "age": "age",
    }
    df = df.rename(columns=rename_map)

    required = ["player", "team", "position", "age"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"laliga_player_info.csv is missing columns: {missing}")

    # age may look like 27-081
    df["age"] = df["age"].astype(str).str.split("-").str[0]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["league"] = "La Liga"
    df["player_merge"] = df["player"].apply(normalize_name)

    keep_cols = ["player", "player_merge", "team", "position", "nationality", "age", "league"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep_cols].drop_duplicates(subset=["player_merge", "league"])


def clean_transfermarkt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    rename_map = {
        "name": "player",
        "league_name": "league",
        "current_value_eur": "market_value",
        "age": "age",
        "nationality": "nationality",
    }
    df = df.rename(columns=rename_map)

    required = ["player", "league", "market_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"transfermarkt_player_values.csv is missing columns: {missing}")

    df = df[df["league"].isin(["Premier League", "La Liga"])].copy()

    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = pd.NA

    if "nationality" not in df.columns:
        df["nationality"] = pd.NA

    df["player_merge"] = df["player"].apply(normalize_name)

    keep_cols = ["player", "player_merge", "league", "market_value", "age", "nationality"]
    return df[keep_cols].drop_duplicates(subset=["player_merge", "league"])


# =====================================================
# LOAD FILES
# =====================================================
pl_stats = pd.read_csv("league-players.csv", sep=";")
laliga_stats = pd.read_csv("league-players (1).csv", sep=";")
pl_info = pd.read_csv("premier_league_player_info.csv")
laliga_info = pd.read_csv("laliga_player_info.csv")
transfermarkt = pd.read_csv("transfermarkt_player_values.csv")

# =====================================================
# CLEAN FILES
# =====================================================
pl_stats = clean_stats(pl_stats, "Premier League")
laliga_stats = clean_stats(laliga_stats, "La Liga")
stats = pd.concat([pl_stats, laliga_stats], ignore_index=True)

pl_info = clean_pl_info(pl_info)
laliga_info = clean_laliga_info(laliga_info)
player_info = pd.concat([pl_info, laliga_info], ignore_index=True)

tm = clean_transfermarkt(transfermarkt)

# =====================================================
# MERGE STATS + PLAYER INFO
# =====================================================
df = stats.merge(
    player_info,
    on=["player_merge", "league"],
    how="left",
    suffixes=("", "_info"),
)

# Fill some nice display columns from info if present
df["team_final"] = df["team_info"].fillna(df["team"]) if "team_info" in df.columns else df["team"]
df["position_final"] = df["position"].fillna("Unknown") if "position" in df.columns else "Unknown"
df["nationality_final"] = df["nationality"].fillna(pd.NA) if "nationality" in df.columns else pd.NA
df["age_final"] = df["age"]

# =====================================================
# MERGE TRANSFERMARKT VALUES
# =====================================================
df = df.merge(
    tm[["player_merge", "league", "market_value", "age", "nationality"]],
    on=["player_merge", "league"],
    how="left",
    suffixes=("", "_tm"),
)

# Prefer Transfermarkt age if info age missing
df["age_final"] = df["age_final"].fillna(df["age_tm"]) if "age_tm" in df.columns else df["age_final"]
df["nationality_final"] = (
    df["nationality_final"].fillna(df["nationality_tm"])
    if "nationality_tm" in df.columns
    else df["nationality_final"]
)

# =====================================================
# BASIC CLEANING
# =====================================================
required_stats_cols = ["player", "team", "apps", "min", "goals", "assists", "xg", "xa", "xg90", "xa90"]
df = df.dropna(subset=required_stats_cols).copy()

# reasonable sample filter
df = df[df["min"] >= 900].copy()

# attacking / advanced midfield filter
if "position_final" in df.columns:
    attacking_mask = df["position_final"].astype(str).str.contains("FW|MF", na=False)
    df = df[attacking_mask].copy()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["g_a"] = df["goals"] + df["assists"]
df["g_a_per90"] = df["g_a"] / (df["min"] / 90)
df["goals_minus_xg"] = df["goals"] - df["xg"]
df["assists_minus_xa"] = df["assists"] - df["xa"]

# =====================================================
# PERFORMANCE SCORE
# =====================================================
performance_features = [
    "xg90",
    "xa90",
    "g_a_per90",
    "goals_minus_xg",
    "assists_minus_xa",
]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[performance_features])

scaled_df = pd.DataFrame(
    scaled,
    columns=[f"{col}_z" for col in performance_features],
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

df["performance_score_100"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
    df[["performance_score"]]
)

# Save overall performance file
df.sort_values("performance_score_100", ascending=False).to_csv(
    "multileague_performance_rankings.csv", index=False
)

# =====================================================
# AGE-ADJUSTED SCOUTING
# =====================================================
age_df = df.dropna(subset=["age_final"]).copy()

if not age_df.empty:
    age_df["age_scaled"] = MinMaxScaler().fit_transform(age_df[["age_final"]])
    age_df["age_bonus"] = 1 - age_df["age_scaled"]
    age_df["performance_scaled"] = MinMaxScaler().fit_transform(age_df[["performance_score"]])

    age_df["age_adjusted_score_raw"] = (
        0.85 * age_df["performance_scaled"] + 0.15 * age_df["age_bonus"]
    )

    age_df["age_adjusted_score"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        age_df[["age_adjusted_score_raw"]]
    )

    age_df = age_df.sort_values("age_adjusted_score", ascending=False)
    age_df.to_csv("multileague_age_adjusted_scouting.csv", index=False)
else:
    age_df = df.copy()
    age_df["age_bonus"] = pd.NA
    age_df["performance_scaled"] = MinMaxScaler().fit_transform(age_df[["performance_score"]])
    age_df["age_adjusted_score"] = age_df["performance_score_100"]
    age_df.to_csv("multileague_age_adjusted_scouting.csv", index=False)

# =====================================================
# TRUE HIDDEN GEM SCORE
# =====================================================
gem_df = age_df.dropna(subset=["market_value"]).copy()

if not gem_df.empty:
    gem_df["value_scaled"] = MinMaxScaler().fit_transform(gem_df[["market_value"]])
    gem_df["value_bonus"] = 1 - gem_df["value_scaled"]

    gem_df["hidden_gem_score_raw"] = (
        0.70 * gem_df["performance_scaled"]
        + 0.20 * gem_df["value_bonus"]
        + 0.10 * gem_df["age_bonus"]
    )

    gem_df["hidden_gem_score"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        gem_df[["hidden_gem_score_raw"]]
    )

    # extra nice metric
    gem_df["value_inefficiency_score"] = (
        gem_df["performance_score_100"] / gem_df["market_value"]
    ) * 1_000_000

    gem_df = gem_df.sort_values("hidden_gem_score", ascending=False)
    gem_df.to_csv("final_multileague_hidden_gems.csv", index=False)
else:
    raise ValueError(
        "No players matched to transfermarkt values. Check player names / league names in the CSVs."
    )

# =====================================================
# PREVIEW
# =====================================================
print("\n=== Top Multi-League Performance Players ===")
print(
    df.sort_values("performance_score_100", ascending=False)[
        ["player", "team_final", "league", "apps", "min", "xg90", "xa90", "g_a_per90", "performance_score_100"]
    ].head(20)
)

print("\n=== Top Age-Adjusted Scouting Players ===")
print(
    age_df[
        ["player", "team_final", "league", "age_final", "xg90", "xa90", "g_a_per90", "age_adjusted_score"]
    ].head(20)
)

print("\n=== Top Hidden Gems ===")
print(
    gem_df[
        [
            "player",
            "team_final",
            "league",
            "age_final",
            "market_value",
            "xg90",
            "xa90",
            "g_a_per90",
            "hidden_gem_score",
        ]
    ].head(20)
)

print("\nSaved files:")
print("- multileague_performance_rankings.csv")
print("- multileague_age_adjusted_scouting.csv")
print("- final_multileague_hidden_gems.csv")
