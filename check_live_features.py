import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from features_live import FEATURES, load_logs, build_team_table, player_features

logs = load_logs()
team = build_team_table(logs)

md = pd.read_csv("data/model_data.csv")
md["GAME_DATE"] = pd.to_datetime(md["GAME_DATE"])
test = md[md["SEASON_YEAR"] == "2025-26"].reset_index(drop=True)

model = XGBRegressor()
model.load_model("models/xgb_points.json")

# ---------- Test 1: 200 random games, rebuild features, compare to saved ones ----------
idx = np.random.default_rng(3).choice(len(test), size=200, replace=False)
worst_by_feature = {k: 0.0 for k in FEATURES}
bad_rows = 0
for i in idx:
    r = test.loc[int(i)]
    f = player_features(logs, int(r["PLAYER_ID"]), int(r["IS_HOME"]), float(r["DAYS_REST"]),
                        r["OPP"], team, as_of=r["GAME_DATE"])
    row_worst = 0.0
    for k in FEATURES:
        d = abs(float(f[k]) - float(r[k]))
        worst_by_feature[k] = max(worst_by_feature[k], d)
        row_worst = max(row_worst, d)
    if row_worst > 0.001:
        bad_rows += 1

print("TEST 1: rebuilt features vs saved features, 200 random 2025-26 games")
print("  Rows where any feature differs by more than 0.001:", bad_rows, "of 200")
print("  Largest difference per feature:")
for k, v in worst_by_feature.items():
    print(f"    {k:18s} {v:.6f}")

# ---------- Test 2: do we reproduce the three earlier predictions? ----------
print()
print("TEST 2: predictions from rebuilt features vs the ones we saw earlier")
for first, date, earlier in [("Luka", "2026-03-31", 32.5),
                             ("Valanciunas", "2025-11-29", 7.0),
                             ("LeBron", "2026-01-13", 22.7)]:
    m = test["PLAYER_NAME"].str.contains(first) & (test["GAME_DATE"].dt.strftime("%Y-%m-%d") == date)
    r = test[m].iloc[0]
    f = player_features(logs, int(r["PLAYER_ID"]), int(r["IS_HOME"]), float(r["DAYS_REST"]),
                        r["OPP"], team, as_of=r["GAME_DATE"])
    p = float(model.predict(pd.DataFrame([f])[FEATURES])[0])
    print(f"  {r['PLAYER_NAME']:22s} rebuilt {p:5.1f} | earlier {earlier:5.1f}")

# ---------- Test 3: what the app will do (all games, no as_of) ----------
print()
print("TEST 3: a made-up next game using ALL data (this is what the app does)")
luka_id = int(logs[logs["PLAYER_NAME"].str.contains("Don")].iloc[0]["PLAYER_ID"])
name = logs[logs["PLAYER_ID"] == luka_id]["PLAYER_NAME"].iloc[0]
f = player_features(logs, luka_id, 1, 2, "NYK", team, as_of=None)
p = float(model.predict(pd.DataFrame([f])[FEATURES])[0])
print(f"  {name}, home vs NYK, 2 days rest, predicted {p:.1f}")
for k in FEATURES:
    print(f"    {k:18s} {float(f[k]):.2f}")