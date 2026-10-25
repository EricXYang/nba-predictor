import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

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
# For picking the number of trees only: earlier seasons train, 2024-25 validates
sub_train = df[df["SEASON_YEAR"].isin(["2021-22", "2022-23", "2023-24"])]
val = df[df["SEASON_YEAR"] == "2024-25"]

X_train, y_train = train[FEATURES], train["PTS"]
X_test, y_test = test[FEATURES], test["PTS"]


def mae(pred, actual):
    return float(abs(pd.Series(pred, index=actual.index) - actual).mean())


baseline = mae(test["PTS_SEASON_AVG"], y_test)
linear = LinearRegression().fit(X_train, y_train)
linear_mae = mae(linear.predict(X_test), y_test)

print("Baseline (season-to-date avg) MAE:", round(baseline, 3))
print("Linear regression MAE:            ", round(linear_mae, 3))
print()

shared = dict(
    learning_rate=0.03, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

for obj in ["reg:squarederror", "reg:absoluteerror"]:
    # Step A: find how many trees are enough using the validation season
    finder = XGBRegressor(
        n_estimators=2000, objective=obj, eval_metric="mae",
        early_stopping_rounds=50, **shared,
    )
    finder.fit(
        sub_train[FEATURES], sub_train["PTS"],
        eval_set=[(val[FEATURES], val["PTS"])], verbose=False,
    )
    best_trees = finder.best_iteration + 1

    # Step B: retrain on all training seasons with that many trees
    final = XGBRegressor(n_estimators=best_trees, objective=obj, **shared)
    final.fit(X_train, y_train)

    train_mae = mae(final.predict(X_train), y_train)
    test_mae = mae(final.predict(X_test), y_test)
    pred = final.predict(X_test)

    print("XGBoost objective:", obj)
    print("  Trees used:", best_trees)
    print("  Train MAE:", round(train_mae, 3), "| Test MAE:", round(test_mae, 3))
    print("  Improvement vs baseline:", round((baseline - test_mae) / baseline * 100, 2), "%")
    print("  Improvement vs linear:  ", round((linear_mae - test_mae) / linear_mae * 100, 2), "%")
    print("  Avg prediction:", round(pred.mean(), 2), "| Avg actual:", round(y_test.mean(), 2))
    print("  Lowest:", round(pred.min(), 1), "| Highest:", round(pred.max(), 1))
    print()