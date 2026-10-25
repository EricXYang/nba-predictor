from features_live import load_logs, build_team_table

team = build_team_table(load_logs())
d = team[team["TEAM_ABBREVIATION"] == "DAL"].sort_values("GAME_DATE").tail(10)
print(d[["GAME_DATE", "PTS", "PTS_ALLOWED"]].to_string(index=False))
print("Average points allowed:", round(d["PTS_ALLOWED"].mean(), 1))