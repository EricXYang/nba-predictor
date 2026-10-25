import pandas as pd

df = pd.read_csv("data/features_step1.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
rows_before = len(df)

# ---------- Group 1: rest ----------
gap = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
df["DAYS_REST"] = gap.clip(upper=7).fillna(7)
df["BACK_TO_BACK"] = (df["DAYS_REST"] == 1).astype(int)

# ---------- Group 2: team table built from player logs ----------
team = (
    df.groupby(["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"], as_index=False)[
        ["PTS", "FGA", "FTA", "OREB", "TOV"]
    ].sum()
)
team["POSS"] = team["FGA"] + 0.44 * team["FTA"] - team["OREB"] + team["TOV"]

# Attach the other team's numbers from the same game
other = team[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(
    columns={"TEAM_ID": "OTHER_TEAM_ID", "PTS": "PTS_ALLOWED", "POSS": "OTHER_POSS"}
)
team = team.merge(other, on="GAME_ID")
team = team[team["TEAM_ID"] != team["OTHER_TEAM_ID"]].copy()

team["GAME_POSS"] = (team["POSS"] + team["OTHER_POSS"]) / 2
team["DRTG"] = team["PTS_ALLOWED"] / team["OTHER_POSS"] * 100

# Rolling last 10 games, only games BEFORE this one
team = team.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
by_team = team.groupby("TEAM_ID")
for col, name in [
    ("PTS_ALLOWED", "OPP_DEF_PTS_L10"),
    ("GAME_POSS", "OPP_PACE_L10"),
    ("DRTG", "OPP_DRTG_L10"),
]:
    team[name] = by_team[col].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean()
    )

# Attach to each player row: the OPPONENT's numbers for that game
opp_feats = team[
    ["GAME_ID", "TEAM_ABBREVIATION", "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10"]
].rename(columns={"TEAM_ABBREVIATION": "OPP"})
df = df.merge(opp_feats, on=["GAME_ID", "OPP"], how="left")

df.to_csv("data/features_step2.csv", index=False)

# ---------- Checks ----------
print("1. Row count before and after (must match):", rows_before, len(df))
print("   Shape:", df.shape)
print()
print("2. Team sanity (NBA teams score about 110 to 115, about 95 to 105 possessions):")
print("   Avg team points:", round(team["PTS"].mean(), 1))
print("   Avg team possessions:", round(team["POSS"].mean(), 1))
print()
print("3. Missing values in new columns:")
new_cols = ["DAYS_REST", "BACK_TO_BACK", "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10"]
print(df[new_cols].isna().sum())
print()
print("4. LEAK CHECK: Knicks first 6 games.")
print("   DEF_PTS_L10 on game 4 should be the average of PTS_ALLOWED in games 1 to 3.")
nyk = team[team["TEAM_ABBREVIATION"] == "NYK"].head(6)
print(nyk[["GAME_DATE", "PTS", "PTS_ALLOWED", "OPP_DEF_PTS_L10"]])
print()
print("5. Luka sample (rest and opponent columns):")
one = df[df["PLAYER_NAME"] == "Luka Dončić"].head(8)
print(one[["GAME_DATE", "OPP", "DAYS_REST", "BACK_TO_BACK", "OPP_DEF_PTS_L10", "OPP_PACE_L10"]])