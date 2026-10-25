from features_live import load_logs, build_team_table

logs = load_logs()
team = build_team_table(logs)
t = team[team["GAME_DATE"] >= "2025-10-01"]
by = t.groupby("TEAM_ABBREVIATION")["GAME_POSS"].mean().sort_values()

print("League average pace, 2025-26:", round(t["GAME_POSS"].mean(), 1))
print()
print("Slowest 5 teams:")
print(by.head(5).round(1).to_string())
print()
print("Fastest 5 teams:")
print(by.tail(5).round(1).to_string())
print()
print("NYK season average:", round(by["NYK"], 1))
nyk = t[t["TEAM_ABBREVIATION"] == "NYK"].sort_values("GAME_DATE").tail(10)
print("NYK last 10 games: ", round(nyk["GAME_POSS"].mean(), 1))
print("NYK last 10 dates: ", nyk["GAME_DATE"].min().date(), "to", nyk["GAME_DATE"].max().date())