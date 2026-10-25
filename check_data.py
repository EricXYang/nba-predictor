import pandas as pd

df = pd.read_csv("data/game_logs.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

print("1. SHAPE (rows, columns):", df.shape)

print()
print("2. Duplicate player-game rows:", df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum())

print()
print("3. Missing values in key columns:")
print(df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "PTS"]].isna().sum())

print()
print("4. Rows with 0 minutes:", (df["MIN"] == 0).sum())
print("   Rows with under 5 minutes:", (df["MIN"] < 5).sum())
print("   Minutes summary:")
print(df["MIN"].describe())

print()
print("5. Points summary:")
print(df["PTS"].describe())
print("   Top 5 scoring games:")
print(df.nlargest(5, "PTS")[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "MIN", "PTS"]])

print()
print("6. Games per team per season (teams should be 30, games around 82):")
g = df.groupby(["SEASON_YEAR", "TEAM_ABBREVIATION"])["GAME_ID"].nunique()
print(g.groupby("SEASON_YEAR").agg(["count", "min", "max"]))

print()
print("7. Unique players per season:")
print(df.groupby("SEASON_YEAR")["PLAYER_ID"].nunique())

print()
print("8. Teams per game (every game should have exactly 2 teams):")
print(df.groupby("GAME_ID")["TEAM_ID"].nunique().value_counts())