import pandas as pd

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG",
    "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]


def load_logs():
    df = pd.read_csv("data/clean_logs.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def build_team_table(logs):
    """One row per team per game: points allowed, game pace, defensive rating."""
    team = logs.groupby(
        ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"], as_index=False
    )[["PTS", "FGA", "FTA", "OREB", "TOV"]].sum()
    team["POSS"] = team["FGA"] + 0.44 * team["FTA"] - team["OREB"] + team["TOV"]
    other = team[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(
        columns={"TEAM_ID": "OTHER_TEAM_ID", "PTS": "PTS_ALLOWED", "POSS": "OTHER_POSS"}
    )
    team = team.merge(other, on="GAME_ID")
    team = team[team["TEAM_ID"] != team["OTHER_TEAM_ID"]].copy()
    team["GAME_POSS"] = (team["POSS"] + team["OTHER_POSS"]) / 2
    team["DRTG"] = team["PTS_ALLOWED"] / team["OTHER_POSS"] * 100
    return team.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def player_features(logs, player_id, is_home, days_rest, opp, team_table, as_of=None):
    """Features for ONE game. Only uses games before as_of (or all games if as_of is None)."""
    g = logs[logs["PLAYER_ID"] == player_id]
    t = team_table
    if as_of is not None:
        g = g[g["GAME_DATE"] < as_of]
        t = t[t["GAME_DATE"] < as_of]
    g = g.sort_values(["GAME_DATE", "GAME_ID"])
    if len(g) < 3:
        raise ValueError("Not enough games for this player")

    season = g["SEASON_YEAR"].iloc[-1]
    f = {
        "PTS_L5": g["PTS"].tail(5).mean(),
        "PTS_L10": g["PTS"].tail(10).mean(),
        "PTS_SEASON_AVG": g[g["SEASON_YEAR"] == season]["PTS"].mean(),
        "MIN_L5": g["MIN"].tail(5).mean(),
        "MIN_L10": g["MIN"].tail(10).mean(),
    }
    f["MIN_TREND"] = f["MIN_L5"] - f["MIN_L10"]
    f["IS_HOME"] = int(is_home)
    f["DAYS_REST"] = float(min(days_rest, 7))
    f["BACK_TO_BACK"] = int(days_rest == 1)

    o = t[t["TEAM_ABBREVIATION"] == opp].sort_values(["GAME_DATE", "GAME_ID"]).tail(10)
    if len(o) < 3:
        raise ValueError("Not enough games for this opponent")
    f["OPP_DEF_PTS_L10"] = o["PTS_ALLOWED"].mean()
    f["OPP_PACE_L10"] = o["GAME_POSS"].mean()
    f["OPP_DRTG_L10"] = o["DRTG"].mean()
    return f