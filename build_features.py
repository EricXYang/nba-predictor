import pandas as pd

df = pd.read_csv("data/clean_logs.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

# Group by player so one player's games never mix with another's
by_player = df.groupby("PLAYER_ID")


def past_rolling_mean(col, window):
    # shift(1) first = only games BEFORE this one. Then average the last N of those.
    return by_player[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )


df["PTS_L5"] = past_rolling_mean("PTS", 5)
df["PTS_L10"] = past_rolling_mean("PTS", 10)
df["MIN_L5"] = past_rolling_mean("MIN", 5)
df["MIN_L10"] = past_rolling_mean("MIN", 10)
df["MIN_TREND"] = df["MIN_L5"] - df["MIN_L10"]

# Season-to-date average points, before this game (resets each season)
df["PTS_SEASON_AVG"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["PTS"].transform(
    lambda s: s.shift(1).expanding(min_periods=3).mean()
)

# How many games the player has played before this one
df["GAMES_BEFORE"] = by_player.cumcount()

df.to_csv("data/features_step1.csv", index=False)

print("Shape:", df.shape)
print()
print("Missing values in new columns (early-career rows will be empty, that's expected):")
new_cols = ["PTS_L5", "PTS_L10", "MIN_L5", "MIN_L10", "MIN_TREND", "PTS_SEASON_AVG"]
print(df[new_cols].isna().sum())
print()
print("LEAK CHECK on Luka's first 8 games.")
print("PTS_L5 on row N should equal the average of PTS on the rows before it.")
one = df[df["PLAYER_NAME"] == "Luka Dončić"].head(8)
print(one[["GAME_DATE", "PTS", "MIN", "PTS_L5", "MIN_L5", "PTS_SEASON_AVG", "GAMES_BEFORE"]])