import os
import json
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/model_data.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG",
    "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]

train = df[df["SEASON_YEAR"] != "2025-26"]
test = df[df["SEASON_YEAR"] == "2025-26"].reset_index(drop=True)
sub_train = df[df["SEASON_YEAR"].isin(["2021-22", "2022-23", "2023-24"])]
val = df[df["SEASON_YEAR"] == "2024-25"]

shared = dict(
    learning_rate=0.03, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
obj = "reg:absoluteerror"

finder = XGBRegressor(n_estimators=2000, objective=obj, eval_metric="mae",
                      early_stopping_rounds=50, **shared)
finder.fit(sub_train[FEATURES], sub_train["PTS"],
           eval_set=[(val[FEATURES], val["PTS"])], verbose=False)
trees = finder.best_iteration + 1

model = XGBRegressor(n_estimators=trees, objective=obj, **shared)
model.fit(train[FEATURES], train["PTS"])

model.save_model("models/xgb_points.json")
with open("models/features.json", "w") as f:
    json.dump(FEATURES, f)

X_test = test[FEATURES]
pred = model.predict(X_test)
test_mae = float(np.abs(pred - test["PTS"].values).mean())
print("Trees:", trees, "| Test MAE (should be about 4.75):", round(test_mae, 3))

# ---------- SHAP ----------
explainer = shap.TreeExplainer(model)
sv = explainer(X_test)
base = np.array(sv.base_values).reshape(-1)
vals = sv.values

# ---------- Check 1: do the pieces add up to the prediction? ----------
rebuilt = base + vals.sum(axis=1)
diff = np.abs(rebuilt - pred)
print()
print("CHECK: base + all feature pieces vs the model's real prediction")
print("  Largest gap over all games:", round(float(diff.max()), 4), "points")
print("  Average gap:", round(float(diff.mean()), 5), "points")
print("  Base value (the model's starting point):", round(float(base[0]), 2))
print("  (Gaps near 0 mean SHAP is working. Anything over about 0.1 is a problem.)")

# ---------- Check 2: which features matter most ----------
imp = pd.Series(np.abs(vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print()
print("AVERAGE SIZE OF EACH FEATURE'S PUSH (points, ignoring up or down):")
print(imp.round(3).to_string())

# ---------- Save SHAP for later steps ----------
out = test[["PLAYER_NAME", "GAME_DATE", "OPP"]].copy()
out["PRED"] = pred
out["BASE"] = base
for i, name in enumerate(FEATURES):
    out["SHAP_" + name] = vals[:, i]
out.to_csv("data/shap_test.csv", index=False)


# ---------- Three example games ----------
def show(i, label):
    row = test.iloc[i]
    print()
    print("=" * 60)
    print(label)
    print(f"{row['PLAYER_NAME']} vs {row['OPP']} on {row['GAME_DATE'].date()}")
    print(f"Predicted: {pred[i]:.1f} | Actual: {row['PTS']:.0f} | Base: {base[i]:.1f}")
    t = pd.DataFrame({
        "value": X_test.iloc[i].values,
        "push_pts": vals[i],
    }, index=FEATURES)
    t = t.reindex(t["push_pts"].abs().sort_values(ascending=False).index)
    print(t.round(2).to_string())
    print("Sum of pushes:", round(float(vals[i].sum()), 2),
          "| Base + pushes:", round(float(base[i] + vals[i].sum()), 2))


show(int(np.argmax(pred)), "EXAMPLE 1: highest predicted game")
bench = np.where(test["MIN_L10"] < 16)[0]
show(int(bench[100]), "EXAMPLE 2: a bench player")
b2b = np.where((test["BACK_TO_BACK"] == 1) & (test["MIN_L10"] > 28))[0]
show(int(b2b[0]), "EXAMPLE 3: a starter on a back-to-back")

print()
print("Saved: models/xgb_points.json, models/features.json, data/shap_test.csv")