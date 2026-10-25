from features_live import load_logs

logs = load_logs()
cur = logs[logs["GAME_DATE"] >= "2025-10-01"]
last = logs.sort_values(["GAME_DATE", "GAME_ID"]).groupby("PLAYER_ID").tail(1)
last = last[last["GAME_DATE"] >= "2025-10-01"].set_index("PLAYER_ID")

rows = []
for pid, g in logs.groupby("PLAYER_ID"):
    if pid not in last.index:
        continue
    g = g.sort_values(["GAME_DATE", "GAME_ID"])
    rows.append({
        "name": last.loc[pid, "PLAYER_NAME"],
        "total_games": len(g),
        "min_last10": g["MIN"].tail(10).mean(),
        "games_2526": int((g["GAME_DATE"] >= "2025-10-01").sum()),
    })
import pandas as pd
d = pd.DataFrame(rows)

print("TRUE BENCH (about 11 to 15 minutes a game):")
print(d[(d["min_last10"] > 11) & (d["min_last10"] < 15) & (d["total_games"] >= 30)].head(5)[["name", "min_last10"]].round(1).to_string(index=False))
print()
print("UNDER 10 MINUTES (should show the warning):")
print(d[(d["min_last10"] < 8) & (d["total_games"] >= 20)].head(4)[["name", "min_last10"]].round(1).to_string(index=False))
print()
print("VERY FEW GAMES EVER (should show a 'not enough games' message):")
print(d[d["total_games"] < 3].head(4)[["name", "total_games"]].to_string(index=False))