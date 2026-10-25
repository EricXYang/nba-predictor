import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

os.makedirs("charts", exist_ok=True)

df = pd.read_csv("data/model_data.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG",
    "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]

train = df[df["SEASON_YEAR"] != "2025-26"]
test = df[df["SEASON_YEAR"] == "2025-26"].copy()
sub_train = df[df["SEASON_YEAR"].isin(["2021-22", "2022-23", "2023-24"])]
val = df[df["SEASON_YEAR"] == "2024-25"]
y = test["PTS"].values

shared = dict(
    learning_rate=0.03, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

preds = {"Baseline (season avg)": test["PTS_SEASON_AVG"].values}
preds["Linear regression"] = LinearRegression().fit(train[FEATURES], train["PTS"]).predict(test[FEATURES])

for name, obj in [("XGBoost (squared)", "reg:squarederror"), ("XGBoost (absolute)", "reg:absoluteerror")]:
    finder = XGBRegressor(n_estimators=2000, objective=obj, eval_metric="mae",
                          early_stopping_rounds=50, **shared)
    finder.fit(sub_train[FEATURES], sub_train["PTS"],
               eval_set=[(val[FEATURES], val["PTS"])], verbose=False)
    final = XGBRegressor(n_estimators=finder.best_iteration + 1, objective=obj, **shared)
    final.fit(train[FEATURES], train["PTS"])
    preds[name] = final.predict(test[FEATURES])

# ---------- 1. Side-by-side table ----------
abs_err = {k: np.abs(v - y) for k, v in preds.items()}
base_mae = abs_err["Baseline (season avg)"].mean()
print("1. MAE ON TEST SET (2025-26), lower is better")
for k, e in abs_err.items():
    print(f"   {k:24s} {e.mean():.3f}   ({(base_mae - e.mean()) / base_mae * 100:+.2f}% vs baseline)")

# ---------- 2. Is the edge real or noise? ----------
print()
print("2. PAIRED CHECK: per game, baseline error minus model error (positive = model better)")
for k in ["Linear regression", "XGBoost (absolute)"]:
    d = abs_err["Baseline (season avg)"] - abs_err[k]
    se = d.std() / np.sqrt(len(d))
    print(f"   {k:24s} avg gain {d.mean():.3f} pts/game, 95% range {d.mean() - 1.96 * se:.3f} to {d.mean() + 1.96 * se:.3f}")
d = abs_err["Linear regression"] - abs_err["XGBoost (absolute)"]
se = d.std() / np.sqrt(len(d))
print(f"   XGB(abs) vs Linear:      avg gain {d.mean():.3f} pts/game, 95% range {d.mean() - 1.96 * se:.3f} to {d.mean() + 1.96 * se:.3f}")
print("   (If the range includes 0, the edge could just be noise. Rows are not fully independent, so the range is a bit optimistic.)")

# ---------- 3. Error by player type ----------
print()
print("3. ERROR BY PLAYER TYPE (average minutes over last 10 games)")
test["BASE_ERR"] = abs_err["Baseline (season avg)"]
test["LIN_ERR"] = abs_err["Linear regression"]
test["XGB_ERR"] = abs_err["XGBoost (absolute)"]
test["TYPE"] = pd.cut(test["MIN_L10"], [0, 20, 28, 100],
                      labels=["Bench (10-20 min)", "Rotation (20-28 min)", "Starters (28+ min)"])
by_type = test.groupby("TYPE", observed=True).agg(
    games=("PTS", "size"), avg_pts=("PTS", "mean"),
    baseline_mae=("BASE_ERR", "mean"), linear_mae=("LIN_ERR", "mean"), xgb_mae=("XGB_ERR", "mean"),
).round(3)
by_type["xgb_gain_%"] = ((by_type["baseline_mae"] - by_type["xgb_mae"]) / by_type["baseline_mae"] * 100).round(2)
print(by_type.to_string())

# ---------- Save predictions ----------
out = test[["PLAYER_NAME", "GAME_DATE", "OPP", "MIN", "MIN_L10", "PTS"]].copy()
out["PRED_BASELINE"] = preds["Baseline (season avg)"]
out["PRED_LINEAR"] = preds["Linear regression"]
out["PRED_XGB"] = preds["XGBoost (absolute)"]
out.to_csv("data/test_predictions.csv", index=False)

# ---------- Chart 1: MAE by model ----------
names = list(abs_err.keys())
vals = [abs_err[k].mean() for k in names]
plt.figure(figsize=(8, 5))
bars = plt.bar(names, vals, color=["#999999", "#4C72B0", "#55A868", "#C44E52"])
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.3f}", ha="center")
plt.ylim(0, 5.6)
plt.ylabel("Average miss in points (MAE)")
plt.title("Test set 2025-26: average miss per player-game (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("charts/mae_by_model.png", dpi=150)
plt.close()

# ---------- Chart 2: predicted vs actual ----------
plt.figure(figsize=(7, 6))
for k, c in [("Linear regression", "#4C72B0"), ("XGBoost (absolute)", "#C44E52")]:
    t = pd.DataFrame({"p": preds[k], "a": y})
    t["bin"] = pd.qcut(t["p"], 15, duplicates="drop")
    g = t.groupby("bin", observed=True).mean()
    plt.plot(g["p"], g["a"], marker="o", label=k, color=c)
lim = max(preds["Linear regression"].max(), 30)
plt.plot([0, lim], [0, lim], "--", color="gray", label="Perfect")
plt.xlabel("Predicted points (groups of similar predictions)")
plt.ylabel("Average actual points in that group")
plt.title("Predicted vs actual (closer to dashed line is better)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/predicted_vs_actual.png", dpi=150)
plt.close()

print()
print("Saved: charts/mae_by_model.png, charts/predicted_vs_actual.png, data/test_predictions.csv")