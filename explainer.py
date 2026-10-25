import re
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env
MODEL = "claude-haiku-5-5"

SYSTEM = (
    "You explain a basketball points prediction to a casual fan. "
    "Use ONLY the numbers and facts in the data block. "
    "Do not add any other stats, player history, injuries, or news. "
    "Write exactly two sentences in plain English, with no jargon. "
    "Say what raised or lowered the prediction and by roughly how many points. "
    "The season, last-10 and last-5 scoring averages share ONE combined effect. "
    "Never say that a single one of those averages caused that effect; "
    "describe it as the player's scoring form as a whole. "
    "Do not say a player played more or fewer minutes than usual; "
    "just say whether the minutes factor raised or lowered the prediction."
    "Call the 10.8 figure 'the average prediction across players', never a league-wide or scoring average. "
    "Do not explain why an effect is zero or small; just state it."
)


def build_facts(name, opp, f, s, pred, base):
    """f = feature values for the game, s = SHAP pushes for the game."""
    form = s["PTS_L5"] + s["PTS_L10"] + s["PTS_SEASON_AVG"]
    minutes = s["MIN_L5"] + s["MIN_L10"] + s["MIN_TREND"]
    rest_home = s["IS_HOME"] + s["DAYS_REST"] + s["BACK_TO_BACK"]
    opponent = s["OPP_DEF_PTS_L10"] + s["OPP_PACE_L10"] + s["OPP_DRTG_L10"]
    lines = [
        f"Player: {name}",
        f"Opponent: {opp}",
        f"Home game: {'yes' if f['IS_HOME'] == 1 else 'no'}",
        f"Days of rest: {f['DAYS_REST']:.0f}",
        f"Back-to-back: {'yes' if f['BACK_TO_BACK'] == 1 else 'no'}",
        f"Predicted points: {pred:.1f}",
        f"Average prediction across all players: {base:.1f}",
        f"Season average so far: {f['PTS_SEASON_AVG']:.1f}",
        f"Average over last 10 games: {f['PTS_L10']:.1f}",
        f"Average over last 5 games: {f['PTS_L5']:.1f}",
        f"Average minutes over last 5 games: {f['MIN_L5']:.1f}",
        f"Combined effect of season average, last-10 average and last-5 average together: {form:+.1f} points",
        f"Effect of minutes: {minutes:+.1f} points",
        f"Effect of rest and home court: {rest_home:+.1f} points",
        f"Effect of the opponent: {opponent:+.1f} points",
    ]
    return "\n".join(lines)


def numbers_in(text):
    return {float(x) for x in re.findall(r"\d+\.?\d*", text)}


def explain(facts):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": "DATA:\n" + facts}],
    )
    # The reply is a list of blocks. Keep only the text ones, skip any thinking blocks.
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        text = "(no text came back, stop reason: " + str(resp.stop_reason) + ")"
    extra = sorted(numbers_in(text) - numbers_in(facts))
    return text, extra


if __name__ == "__main__":
    FEATURES = [
        "PTS_L5", "PTS_L10", "PTS_SEASON_AVG", "MIN_L5", "MIN_L10", "MIN_TREND",
        "IS_HOME", "DAYS_REST", "BACK_TO_BACK",
        "OPP_DEF_PTS_L10", "OPP_PACE_L10", "OPP_DRTG_L10",
    ]
    df = pd.read_csv("data/model_data.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    test = df[df["SEASON_YEAR"] == "2025-26"].reset_index(drop=True)
    sh = pd.read_csv("data/shap_test.csv")
    sh["GAME_DATE"] = pd.to_datetime(sh["GAME_DATE"])

    same = (test["PLAYER_NAME"] == sh["PLAYER_NAME"]).all() and (test["GAME_DATE"] == sh["GAME_DATE"]).all()
    print("Rows line up between the two files:", same)
    if not same:
        raise SystemExit("Files do not line up, stopping.")

    cases = [("Luka", "2026-03-31"), ("Valanciunas", "2025-11-29"), ("LeBron", "2026-01-13")]
    for first, date in cases:
        mask = test["PLAYER_NAME"].str.contains(first) & (test["GAME_DATE"].dt.strftime("%Y-%m-%d") == date)
        i = int(test.index[mask][0])
        f = test.loc[i, FEATURES]
        s = {k: sh.loc[i, "SHAP_" + k] for k in FEATURES}
        facts = build_facts(test.loc[i, "PLAYER_NAME"], test.loc[i, "OPP"], f, s,
                            sh.loc[i, "PRED"], sh.loc[i, "BASE"])
        text, extra = explain(facts)
        print()
        print("=" * 60)
        print(facts)
        print("-" * 60)
        print("ACTUAL points scored (not shown to Claude):", test.loc[i, "PTS"])
        print("CLAUDE SAYS:", text)
        print("Numbers in the answer that were NOT in the data:", extra if extra else "none")