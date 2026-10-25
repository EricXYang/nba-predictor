import time
import os
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "game_logs.csv")


def pull_season(season, tries=4):
    for attempt in range(1, tries + 1):
        try:
            print(f"Pulling {season} (attempt {attempt})...")
            logs = PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=90,
            )
            return logs.get_data_frames()[0]
        except Exception as e:
            print(f"  Failed: {e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"Could not pull {season} after {tries} tries")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    for season in SEASONS:
        cache = os.path.join(DATA_DIR, f"raw_{season}.csv")
        if os.path.exists(cache):
            print(f"{season}: using cached file")
            df = pd.read_csv(cache)
        else:
            df = pull_season(season)
            df.to_csv(cache, index=False)
            print(f"{season}: saved {len(df)} rows")
            time.sleep(5)
        frames.append(df)

    all_logs = pd.concat(frames, ignore_index=True)
    all_logs.to_csv(OUT_FILE, index=False)
    print()
    print("DONE")
    print("Rows, columns:", all_logs.shape)
    print("Column names:", list(all_logs.columns))
    print("Date range:", all_logs["GAME_DATE"].min(), "to", all_logs["GAME_DATE"].max())
    print("Rows per season:")
    print(all_logs.groupby("SEASON_YEAR").size())
    print()
    print(all_logs[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "MIN", "PTS"]].head(8))


main()