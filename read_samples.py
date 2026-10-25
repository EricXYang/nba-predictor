import numpy as np
import pandas as pd

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG", "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]
df = pd.read_csv("data/model_data.csv")
test = df[df["SEASON_YEAR"] == "2025-26"].reset_index(drop=True)
sh = pd.read_csv("data/shap_test.csv")
res = pd.read_csv("data/explainer_check.csv")
picks = np.random.default_rng(7).choice(len(test), size=50, replace=False)

flagged = [k for k in range(len(res)) if res.loc[k, "strict_flags"] != "[]"]
others = [k for k in range(len(res)) if k not in flagged]
chosen = flagged + [int(x) for x in np.random.default_rng(1).choice(others, size=4, replace=False)]

for k in chosen:
    i = int(picks[k])
    assert test.loc[i, "PLAYER_NAME"] == res.loc[k, "player"], "rows do not line up"
    f = test.loc[i, FEATURES]
    s = {n: sh.loc[i, "SHAP_" + n] for n in FEATURES}
    form = s["PTS_L5"] + s["PTS_L10"] + s["PTS_SEASON_AVG"]
    mins = s["MIN_L5"] + s["MIN_L10"] + s["MIN_TREND"]
    rest = s["IS_HOME"] + s["DAYS_REST"] + s["BACK_TO_BACK"]
    opp = s["OPP_DEF_PTS_L10"] + s["OPP_PACE_L10"] + s["OPP_DRTG_L10"]
    print("=" * 60)
    print(f"{res.loc[k, 'player']} vs {res.loc[k, 'opp']} | predicted {sh.loc[i, 'PRED']:.1f} | typical {sh.loc[i, 'BASE']:.1f}")
    print(f"season {f['PTS_SEASON_AVG']:.1f} | last10 {f['PTS_L10']:.1f} | last5 {f['PTS_L5']:.1f} | min last5 {f['MIN_L5']:.1f}")
    print(f"EFFECTS: form {form:+.1f} | minutes {mins:+.1f} | rest/home {rest:+.1f} | opp {opp:+.1f}")
    print("STRICT FLAGS:", res.loc[k, "strict_flags"])
    print("CLAUDE SAID:", res.loc[k, "text"])