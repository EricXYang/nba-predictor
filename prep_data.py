import pandas as pd

df = pd.read_csv("data/game_logs.csv")

# Fix the date (drop the fake time at the end)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

# Keep only the columns we need
keep = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "PTS", "FGA", "FTA", "OREB", "TOV",
]
df = df[keep].copy()

# Home or away, and the opponent, from the MATCHUP text
df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
df["OPP"] = df["MATCHUP"].str.split().str[-1]

# Sort oldest to newest within each player
df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

df.to_csv("data/clean_logs.csv", index=False)

print("Shape (rows, columns):", df.shape)
print()
print("Home vs away counts (should be about equal):")
print(df["IS_HOME"].value_counts())
print()
print("Number of unique opponents (should be 30):", df["OPP"].nunique())
print()
print("Sample: first 8 games of one player, oldest to newest")
one = df[df["PLAYER_NAME"] == "Luka Dončić"]
print(one[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "IS_HOME", "OPP", "MIN", "PTS"]].head(8))