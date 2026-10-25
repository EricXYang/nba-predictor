import pandas as pd

df = pd.read_csv("data/features_step2.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

features_needed = [
    "PTS_L5", "PTS_L10", "MIN_L5", "MIN_L10", "MIN_TREND", "PTS_SEASON_AVG",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]

n0 = len(df)
df = df.dropna(subset=features_needed)
n1 = len(df)
df = df[df["MIN_L10"] >= 10]
n2 = len(df)

print("Rows at start:", n0)
print("After dropping rows with missing features:", n1, "(removed", n0 - n1, ")")
print("After dropping low-minutes players (MIN_L10 < 10):", n2, "(removed", n1 - n2, ")")

df.to_csv("data/model_data.csv", index=False)

train = df[df["SEASON_YEAR"] != "2025-26"]
test = df[df["SEASON_YEAR"] == "2025-26"]

print()
print("Train rows:", len(train), "| dates", train["GAME_DATE"].min().date(), "to", train["GAME_DATE"].max().date())
print("Test rows: ", len(test), "| dates", test["GAME_DATE"].min().date(), "to", test["GAME_DATE"].max().date())
print("Train ends before test starts:", train["GAME_DATE"].max() < test["GAME_DATE"].min())


def mae(pred, actual):
    return (pred - actual).abs().mean()


print()
print("BASELINE ERRORS (MAE = average miss in points, lower is better)")
print("Test set (2025-26):")
print("  Guess = last 5 games avg:      ", round(mae(test["PTS_L5"], test["PTS"]), 3))
print("  Guess = last 10 games avg:     ", round(mae(test["PTS_L10"], test["PTS"]), 3))
print("  Guess = season-to-date avg:    ", round(mae(test["PTS_SEASON_AVG"], test["PTS"]), 3))
print("  Guess = same number for everyone (train avg):", round(mae(pd.Series(train["PTS"].mean(), index=test.index), test["PTS"]), 3))
print()
print("Average points in test set:", round(test["PTS"].mean(), 2))
print("Std dev of points in test set:", round(test["PTS"].std(), 2))