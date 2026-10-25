# NBA Player Points Predictor

Predicts how many points an NBA player will score in one game, then uses the Claude API to explain the prediction in plain English using only the numbers the model produced.

Built as a learning and portfolio project. It is not betting advice, and it was not compared against sportsbook lines.

## Results

Trained on 2021-22 through 2024-25 and tested on the 2025-26 regular season (22,383 player-games the model never saw). The split is by date, not random. Error is mean absolute error (MAE): the average miss in points per player-game. Lower is better.

| Model | Test MAE | vs. baseline |
|---|---|---|
| Baseline: season-to-date average | 4.864 | n/a |
| Linear regression | 4.782 | 1.69% better |
| XGBoost (squared error) | 4.784 | 1.65% better |
| XGBoost (absolute error) | 4.750 | 2.34% better |

Other baselines: last-5-games average 5.007, last-10-games average 4.868, one league-wide number for everyone 6.85.

Honest read: most of the predictable signal is already in a player's scoring average. The model adds a small but consistent improvement on top of it.

- XGBoost (absolute error) beats the baseline by 0.114 points per game (95% range 0.097 to 0.131). It beats linear regression by 0.032 points per game (95% range 0.023 to 0.041). These ranges treat games as independent, but games from the same player and night are related, so the true ranges are wider.
- Gain over the baseline by player type (average minutes over the last 10 games): bench 3.09% (6,807 games), rotation 2.45% (7,901 games), starters 1.80% (7,675 games). Starters have the largest misses in points (5.63 per game).
- Train error (4.742) and test error (4.750) are almost the same, so the model is not overfitting.
- Predicted vs. actual (charts/predicted_vs_actual.png) tracks the diagonal closely, so predictions are well calibrated on average. Its highest prediction on the test set was 32.5, so it cannot call huge scoring nights (for example, it predicted 32.5 for a game where Luka Doncic scored 42).

Charts: `charts/mae_by_model.png` and `charts/predicted_vs_actual.png`.

## How it works

1. **Data.** Five regular seasons (2021-22 to 2025-26) of player game logs from the `nba_api` package: 131,291 player-games, 30 teams x 82 games each season, no duplicates, no missing key values. Pulled once and cached locally.
2. **Features, with no leakage.** Every feature for a game uses only games played before it (`shift(1)` before every rolling average). Features: last-5 and last-10 points and minutes, season-to-date points, minutes trend, home or away, days of rest, back-to-back flag, and opponent points allowed, defensive rating and pace over their last 10 games. Opponent stats are built from the player logs. Pace uses the standard possessions estimate (FGA + 0.44 x FTA - OREB + TOV), not official NBA pace figures.
3. **Filtering.** Rows without enough history (8,513) and players averaging under 10 minutes over their last 10 games (12,622) were dropped, leaving 110,156 rows: 87,773 for training and 22,383 for testing. The minutes filter uses the last-10 average, which is known before tip-off, not the minutes played in the game.
4. **Models.** Linear regression, then XGBoost. The number of trees was chosen on a validation season (2024-25) with early stopping, so the test season was never used for tuning.
5. **Explanations.** SHAP splits each prediction into per-feature pushes in points. Pushes were checked to add up to the prediction (largest gap 0.0 points over all test games). The three scoring averages share credit, so they are reported as one combined "scoring form" effect. The grouped totals (scoring form, minutes, rest and home court, opponent), not the raw per-feature numbers, go to Claude (`claude-haiku-5-5`) with the instruction to use only the numbers provided and write two sentences.
6. **App.** A Streamlit app: pick a player, an opponent, home or away, and days of rest. It shows the prediction, a chart of what pushed it, Claude's explanation, and the exact data Claude was given.

## What drives the predictions

Average size of each feature's push on the test set, in points: season average 2.74, last-10 average 1.47, last-5 average 0.73, last-5 minutes 0.37, then opponent points allowed 0.14, days of rest 0.13, opponent pace 0.08, opponent defensive rating 0.07, last-10 minutes 0.07, minutes trend 0.06, home or away 0.05, back-to-back 0.01. In practice the model mostly follows a player's own scoring history. Opponent, rest and home court barely move predictions.

## How the explanations were checked

- A script runs the explainer on 50 randomly chosen test games and pulls every number out of Claude's answer. 49 of 50 contained no number that was missing from the data provided, after allowing for rounding and for sums and differences of the provided effects. The one flagged answer was a sum of two displayed effects (not individually confirmed).
- 49 of 50 were exactly two sentences, and none used words like injury, streak or career.
- A number check cannot catch a sentence that is misleading, even when every number is real. In the first 3 examples I found the explainer crediting a combined effect to a single average, and changed the prompt. I then read 10 more answers by hand: all numbers were correct, with three small wording problems (calling the 10.8 starting point a league-wide average, inventing a reason for a zero effect, and comparing a player's scoring form to that 10.8). I changed the prompt again and re-ran the 50-game check, but after that last change I only read a few answers by hand, not all 50.
- The app's feature builder was tested against the training pipeline: 200 random test games rebuilt identically (largest difference 0.000000 on every feature).

## Limitations

- **No injuries, lineups or news.** This is the biggest reason it cannot match sportsbooks. Data ends 2026-04-12, so the app does not know about the current season or trades.
- **Single-game scoring is noisy.** The typical miss is about 4.75 points.
- **Not compared to sportsbook lines.** Historical player prop lines are paid, and I did not find a free source with enough coverage.
- **Weak for players near or under 10 minutes a game.** The model was trained only on players above that, and the app warns when a player is below it.
- **Opponent numbers are snapshots of each team's last 10 games**, which at the end of the data means the end of last season.
- **Effects are measured against a starting point of 10.8**, the model's average prediction across all players. "Scoring form lowered it by 3 points" means below the typical player, not a slump.
- **Explanations were checked on 50 games**, not all of them. That is evidence, not proof.
- **The headline number is slightly optimistic.** I chose the best of four models by looking at test error, with no separate final test season held back. The baseline is the best of three baselines, which makes the comparison harder, not easier.
- Package versions are not pinned. Built and tested on Python 3.14.

## Ideas for next steps

- Predict minutes first, since not knowing minutes is a large source of error.
- Add team spreads and totals as features. They reflect injuries and expected blowouts, and are known before tip-off. I have not tested whether this helps.
- Refresh the data for the current season so the app can predict upcoming games.
- Compare against sportsbook player prop lines on a sample, if a usable data source turns up.

## Run it

Needs Python and a Claude API key in a `.env` file (copy `.env.example` and add your key; the `.env` file is ignored by Git). The `data/` folder is also ignored, so build it first:

1. `python -m pip install -r requirements.txt`
2. `python pull_data.py` (downloads the game logs; slow, and the NBA site can rate limit)
3. `python prep_data.py`
4. `python build_features.py`
5. `python build_features2.py`
6. `python baseline.py`
7. `python linear_model.py` and `python xgb_model.py` (optional, to see each model)
8. `python compare_models.py` (results table and charts)
9. `python shap_explain.py` (trains and saves the final model)
10. `python -m streamlit run app.py`