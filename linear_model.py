import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/model_data.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG",
    "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]

train = df[df["SEASON_YEAR"] != "2025-26"]
test = df[df["SEASON_YEAR"] == "2025-26"]

X_train, y_train = train[FEATURES], train["PTS"]
X_test, y_test = test[FEATURES], test["PTS"]

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
model_mae = (pd.Series(pred, index=test.index) - y_test).abs().mean()
baseline_mae = (test["PTS_SEASON_AVG"] - y_test).abs().mean()

print("Train rows:", len(train), "| Test rows:", len(test))
print()
print("Baseline (season-to-date avg) MAE:", round(baseline_mae, 3))
print("Linear regression MAE:            ", round(model_mae, 3))
change = (baseline_mae - model_mae) / baseline_mae * 100
print("Improvement vs baseline:           ", round(change, 2), "%")
print()
print("Weights the model learned (points added per 1 unit of each feature):")
weights = pd.Series(model.coef_, index=FEATURES).round(3)
print(weights)
print("Intercept:", round(model.intercept_, 3))
print()
print("Sanity: average prediction vs average actual in test set")
print("  Predicted:", round(pred.mean(), 2), "| Actual:", round(y_test.mean(), 2))
print("  Lowest prediction:", round(pred.min(), 1), "| Highest:", round(pred.max(), 1))