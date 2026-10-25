import time
from itertools import combinations
import numpy as np
import pandas as pd
from explainer import build_facts, explain, numbers_in

FEATURES = [
    "PTS_L5", "PTS_L10", "PTS_SEASON_AVG", "MIN_L5", "MIN_L10", "MIN_TREND",
    "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
    "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
]
BANNED = ["injur", "streak", "slump", "hot ", "cold", "career", "playoff",
          "trade", "coach", "lineup", "teammate", "star", "momentum", "revenge"]
N = 50

df = pd.read_csv("data/model_data.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
test = df[df["SEASON_YEAR"] == "2025-26"].reset_index(drop=True)
sh = pd.read_csv("data/shap_test.csv")
assert (test["PLAYER_NAME"] == sh["PLAYER_NAME"]).all(), "files do not line up"

rng = np.random.default_rng(7)
picks = rng.choice(len(test), size=N, replace=False)

rows = []
for n, i in enumerate(picks, 1):
    i = int(i)
    f = test.loc[i, FEATURES]
    s = {k: sh.loc[i, "SHAP_" + k] for k in FEATURES}
    pred, base = sh.loc[i, "PRED"], sh.loc[i, "BASE"]
    facts = build_facts(test.loc[i, "PLAYER_NAME"], test.loc[i, "OPP"], f, s, pred, base)

    effects = [
        s["PTS_L5"] + s["PTS_L10"] + s["PTS_SEASON_AVG"],
        s["MIN_L5"] + s["MIN_L10"] + s["MIN_TREND"],
        s["IS_HOME"] + s["DAYS_REST"] + s["BACK_TO_BACK"],
        s["OPP_DEF_PTS_L10"] + s["OPP_PACE_L10"] + s["OPP_DRTG_L10"],
    ]
    try:
        text, strict_extra = explain(facts)
    except Exception as e:
        print("Request", n, "failed:", str(e)[:100])
        continue

    # Fair check: allow rounding, sums of effects, and prediction minus typical player
    allowed = set(numbers_in(facts))
    allowed |= {float(round(x)) for x in list(allowed)}
    cands = list(effects) + [pred - base]
    for r in (2, 3, 4):
        for c in combinations(effects, r):
            cands.append(sum(c))
    for v in cands:
        allowed |= {round(abs(v), 1), float(round(abs(v)))}
    fair_extra = [x for x in strict_extra if not any(abs(x - a) < 0.051 for a in allowed)]

    sentences = [x for x in text.replace("\n", " ").replace("?", ".").replace("!", ".").split(". ") if x.strip()]
    banned = [w.strip() for w in BANNED if w in text.lower()]
    rows.append({
        "player": test.loc[i, "PLAYER_NAME"], "opp": test.loc[i, "OPP"],
        "date": test.loc[i, "GAME_DATE"].date(), "pred": round(pred, 1),
        "actual": test.loc[i, "PTS"], "strict_flags": strict_extra, "fair_flags": fair_extra,
        "banned_words": banned, "sentences": len(sentences), "text": text,
    })
    time.sleep(0.3)

res = pd.DataFrame(rows)
res.to_csv("data/explainer_check.csv", index=False)

total = len(res)
print()
print("RESULTS OVER", total, "EXPLANATIONS")
print("  Strict check, no unexpected numbers:", int((res["strict_flags"].str.len() == 0).sum()), "of", total)
print("  Fair check, no unexpected numbers:  ", int((res["fair_flags"].str.len() == 0).sum()), "of", total)
print("  No outside-the-data words:          ", int((res["banned_words"].str.len() == 0).sum()), "of", total)
print("  Exactly two sentences:              ", int((res["sentences"] == 2).sum()), "of", total)

bad = res[(res["fair_flags"].str.len() > 0) | (res["banned_words"].str.len() > 0) | (res["sentences"] != 2)]
print()
print("ANSWERS TO READ (failed the fair check, used a banned word, or not two sentences):", len(bad))
for _, r in bad.head(10).iterrows():
    print("-" * 60)
    print(r["player"], "vs", r["opp"], r["date"], "| predicted", r["pred"], "| sentences:", r["sentences"])
    print("Fair flags:", r["fair_flags"], "| Banned words:", r["banned_words"])
    print(r["text"])
print()
print("All 50 saved to data/explainer_check.csv")