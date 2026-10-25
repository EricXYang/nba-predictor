import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBRegressor

from features_live import FEATURES, load_logs, build_team_table, player_features
from explainer import build_facts, explain

st.set_page_config(page_title="NBA Points Predictor")


@st.cache_data
def load_data():
    logs = load_logs()
    return logs, build_team_table(logs)


@st.cache_resource
def load_model():
    m = XGBRegressor()
    m.load_model("models/xgb_points.json")
    return m, shap.TreeExplainer(m)


def fallback_text(name, pred, groups):
    big = max(groups, key=lambda k: abs(groups[k]))
    direction = "raised" if groups[big] > 0 else "lowered"
    return (
        f"{name} is predicted to score {pred:.1f} points. The biggest factor was "
        f"{big.lower()}, which {direction} the prediction by about {abs(groups[big]):.1f} "
        f"points compared with the typical player."
    )


logs, team = load_data()
model, shap_explainer = load_model()
data_end = logs["GAME_DATE"].max().date()

st.title("NBA Player Points Predictor")
st.write(
    "Pick a player and a matchup. The model predicts his points for one game, "
    "and Claude explains why in plain English."
)
st.caption(
    f"Data ends {data_end}. The model does not know about injuries, lineups, trades "
    "or news after that. Not betting advice."
)

# Players who appeared in the 2025-26 season
last_game = logs.sort_values(["GAME_DATE", "GAME_ID"]).groupby("PLAYER_ID").tail(1)
current = last_game[last_game["GAME_DATE"] >= "2025-10-01"].set_index("PLAYER_ID")
names = current["PLAYER_NAME"].sort_values()

pid = st.selectbox(
    "Player",
    options=list(names.index),
    index=None,
    format_func=lambda i: names[i],
    placeholder="Type a name to search",
)
if pid is None:
    st.stop()

name = names[pid]
own_team = current.loc[pid, "TEAM_ABBREVIATION"]
opps = sorted(t for t in team["TEAM_ABBREVIATION"].unique() if t != own_team)

c1, c2, c3 = st.columns(3)
opp = c1.selectbox("Opponent", opps)
where = c2.radio("Game location", ["Home", "Away"])
rest = c3.slider("Days of rest", 1, 7, 2)
st.caption(
    f"{name} last played for {own_team} in the data "
    f"(last game {current.loc[pid, 'GAME_DATE'].date()}). "
    "1 day of rest means he played yesterday."
)

if st.button("Predict"):
    try:
        f = player_features(logs, pid, where == "Home", rest, opp, team, as_of=None)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    X = pd.DataFrame([f])[FEATURES]
    pred = float(model.predict(X)[0])
    sv = shap_explainer(X)
    base = float(np.array(sv.base_values).reshape(-1)[0])
    s = dict(zip(FEATURES, sv.values[0]))

    if abs(base + sum(s.values()) - pred) > 0.1:
        st.warning("The breakdown does not add up to the prediction. Do not trust the explanation.")

    groups = {
        "Scoring form": s["PTS_L5"] + s["PTS_L10"] + s["PTS_SEASON_AVG"],
        "Minutes": s["MIN_L5"] + s["MIN_L10"] + s["MIN_TREND"],
        "Rest and home court": s["IS_HOME"] + s["DAYS_REST"] + s["BACK_TO_BACK"],
        "Opponent": s["OPP_DEF_PTS_L10"] + s["OPP_PACE_L10"] + s["OPP_DRTG_L10"],
    }

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted points", f"{pred:.1f}")
    m2.metric("Season average", f"{f['PTS_SEASON_AVG']:.1f}")
    m3.metric("Last 10 games", f"{f['PTS_L10']:.1f}")
    st.caption(
        "On last season's test games, the typical miss was about 4.75 points. "
        "Single-game scoring is noisy."
    )

    if f["MIN_L10"] < 10:
        st.warning(
            "This player averages under 10 minutes a game. The model was only trained on "
            "players above that, so treat this prediction as less reliable."
        )

    st.subheader("What pushed the prediction")
    st.bar_chart(pd.DataFrame({"Points": list(groups.values())}, index=list(groups.keys())))
    st.caption(f"Points added or removed compared with the typical prediction across all players ({base:.1f}).")

    st.subheader("In plain English")
    facts = build_facts(name, opp, f, s, pred, base)
    try:
        with st.spinner("Asking Claude..."):
            text, _ = explain(facts)
        st.write(text)
    except Exception as e:
        st.info("Claude's explanation is unavailable right now, so here is a plain version.")
        st.write(fallback_text(name, pred, groups))
        st.caption(f"Reason: {str(e)[:120]}")

    with st.expander("Exactly what Claude was given"):
        st.code(facts, wrap_lines=True)
    with st.expander("Features used for this prediction"):
        st.dataframe(pd.DataFrame({"value": f}).round(2))