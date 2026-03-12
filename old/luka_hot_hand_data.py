"""
Hot Hand Fallacy - Data Collection Pipeline
CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS

Collects every field goal attempt by Luka Doncic in the 2024-25 season,
preserving game order and shot sequence for Hot Hand analysis.

Output: luka_2024_25_shots.csv
"""

import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, playergamelog


# ─── CONFIG ──────────────────────────────────────────────────────────────────

PLAYER_NAME  = "Luka Doncic"
SEASON_ID    = "2025-26"
SEASON_TYPE  = "Regular Season"

# nba_api can rate-limit; this delay (seconds) is applied between requests
REQUEST_DELAY = 0.6


# ─── STEP 1: RESOLVE PLAYER & TEAM ───────────────────────────────────────────

def get_player_info(player_name: str, season_id: str) -> tuple[int, int]:
    """Return (player_id, team_id) for the given player and season."""
    search_results = players.find_players_by_full_name(player_name)
    if not search_results:
        raise ValueError(f"Player not found: {player_name}")

    player_id = search_results[0]["id"]
    print(f"  Found player: {search_results[0]['full_name']} (id={player_id})")

    time.sleep(REQUEST_DELAY)
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    career_df = career.get_data_frames()[0]

    season_row = career_df[career_df["SEASON_ID"] == season_id]
    if season_row.empty:
        raise ValueError(f"No career data found for season {season_id}")

    team_id = int(season_row["TEAM_ID"].iloc[0])
    print(f"  Team ID for {season_id}: {team_id}")
    return player_id, team_id


# ─── STEP 2: PULL RAW SHOT CHART DATA ────────────────────────────────────────

def get_shot_chart(player_id: int, team_id: int, season_id: str) -> pd.DataFrame:
    """
    Pull every FGA from ShotChartDetail for the given player/season.

    Key columns returned by the endpoint:
      GAME_ID         – unique game identifier (chronological when sorted)
      GAME_DATE       – date of the game (YYYYMMDD string)
      GAME_EVENT_ID   – ordering ID within a game (preserves shot sequence)
      PERIOD          – quarter (1-4, 5+ = OT)
      MINUTES_REMAINING / SECONDS_REMAINING – clock at time of shot
      ACTION_TYPE     – e.g. "Jump Shot", "Layup"
      SHOT_TYPE       – "2PT Field Goal" or "3PT Field Goal"
      SHOT_ZONE_BASIC / SHOT_ZONE_AREA / SHOT_ZONE_RANGE – zone labels
      SHOT_DISTANCE   – feet from basket
      LOC_X / LOC_Y   – court coordinates (used by shot chart viz)
      SHOT_MADE_FLAG  – 1 = make, 0 = miss  ← primary target for Hot Hand
      EVENT_TYPE      – "Made Shot" or "Missed Shot" (redundant with above)
    """
    time.sleep(REQUEST_DELAY)

    shotchart_list = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        season_type_all_star=SEASON_TYPE,
        season_nullable=season_id,
        context_measure_simple="FGA"      # all field goal attempts
    ).get_data_frames()

    df = shotchart_list[0]
    print(f"  Raw shots pulled: {len(df)} FGAs")
    return df


# ─── STEP 3: BUILD ORDERED SHOT SEQUENCE ─────────────────────────────────────

def build_shot_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort shots into true chronological order and add sequence columns
    needed for Hot Hand analysis.

    Added columns:
      shot_number          – global shot index across the full season (1-based)
      game_shot_number     – shot index within a single game (1-based)
      rolling_3_fg_pct     – FG% over the previous 3 shots (recency window)
      rolling_5_fg_pct     – FG% over the previous 5 shots
      rolling_10_fg_pct    – FG% over the previous 10 shots
      streak               – current consecutive makes (+) or misses (-) at
                             each shot, calculated BEFORE this shot
                             e.g. +3 means the player hit 3 in a row going in
    """

    # Sort by game date, then by event order within the game
    df = df.sort_values(["GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)

    # Global shot number (season-level sequence)
    df["shot_number"] = df.index + 1

    # Per-game shot number
    df["game_shot_number"] = df.groupby("GAME_ID").cumcount() + 1

    # Rolling FG% windows (min_periods=1 so early shots aren't NaN)
    made = df["SHOT_MADE_FLAG"]
    df["rolling_3_fg_pct"]  = made.rolling(window=3,  min_periods=1).mean().shift(1)
    df["rolling_5_fg_pct"]  = made.rolling(window=5,  min_periods=1).mean().shift(1)
    df["rolling_10_fg_pct"] = made.rolling(window=10, min_periods=1).mean().shift(1)

    # Consecutive streak going INTO each shot
    streaks = []
    current_streak = 0
    for made_flag in made:
        streaks.append(current_streak)          # record streak BEFORE this shot
        if made_flag == 1:
            current_streak = current_streak + 1 if current_streak >= 0 else 1
        else:
            current_streak = current_streak - 1 if current_streak <= 0 else -1
    df["streak"] = streaks

    return df


# ─── STEP 4: SELECT & RENAME FINAL COLUMNS ───────────────────────────────────

FINAL_COLUMNS = {
    # Identifiers
    "GAME_ID"              : "game_id",
    "GAME_DATE"            : "game_date",
    "shot_number"          : "shot_number",
    "game_shot_number"     : "game_shot_number",

    # Shot context
    "PERIOD"               : "period",
    "MINUTES_REMAINING"    : "minutes_remaining",
    "SECONDS_REMAINING"    : "seconds_remaining",
    "ACTION_TYPE"          : "action_type",
    "SHOT_TYPE"            : "shot_type",
    "SHOT_DISTANCE"        : "shot_distance_ft",

    # Zone labels
    "SHOT_ZONE_BASIC"      : "shot_zone_basic",
    "SHOT_ZONE_AREA"       : "shot_zone_area",
    "SHOT_ZONE_RANGE"      : "shot_zone_range",

    # Court coordinates (for shot chart visualization)
    "LOC_X"                : "loc_x",
    "LOC_Y"                : "loc_y",

    # Outcome ← primary Hot Hand signal
    "SHOT_MADE_FLAG"       : "shot_made",
    "EVENT_TYPE"           : "event_type",

    # Derived Hot Hand columns
    "streak"               : "streak",
    "rolling_3_fg_pct"     : "rolling_3_fg_pct",
    "rolling_5_fg_pct"     : "rolling_5_fg_pct",
    "rolling_10_fg_pct"    : "rolling_10_fg_pct",
}

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in FINAL_COLUMNS if c in df.columns]
    out  = df[keep].rename(columns=FINAL_COLUMNS)

    # Format game_date as YYYY-MM-DD for readability
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    output_path = "luka_2025_26_shots.csv"

    print(f"\n{'='*55}")
    print(f"  Hot Hand Data Pipeline — {PLAYER_NAME} {SEASON_ID}")
    print(f"{'='*55}\n")

    print("[1/4] Resolving player and team info...")
    player_id, team_id = get_player_info(PLAYER_NAME, SEASON_ID)

    print("\n[2/4] Pulling shot chart from NBA API...")
    raw_df = get_shot_chart(player_id, team_id, SEASON_ID)

    print("\n[3/4] Building chronological shot sequence...")
    seq_df = build_shot_sequence(raw_df)

    print("\n[4/4] Finalizing and saving...")
    final_df = finalize(seq_df)
    final_df.to_csv(output_path, index=False)

    # ── Summary stats ──
    total  = len(final_df)
    made   = final_df["shot_made"].sum()
    games  = final_df["game_id"].nunique()
    pct    = made / total * 100 if total else 0

    print(f"\n  ✓ Saved {total} shots across {games} games → {output_path}")
    print(f"  ✓ Overall FG%: {made}/{total} ({pct:.1f}%)")
    print(f"  ✓ Columns: {list(final_df.columns)}\n")

    # ── Quick peek at streak distribution ──
    streak_counts = final_df["streak"].value_counts().sort_index()
    print("  Streak distribution going into each shot:")
    print("  (negative = miss streak, positive = make streak)\n")
    for streak_val, count in streak_counts.items():
        bar = "█" * min(count // 10, 40)
        print(f"  {streak_val:+3d}  {bar} {count}")

    return final_df


if __name__ == "__main__":
    df = main()
