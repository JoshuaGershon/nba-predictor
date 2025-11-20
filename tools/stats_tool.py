from __future__ import annotations
from typing import Dict, Optional
import os, time
import pandas as pd

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import TeamGameLog

from tools.cache import read_df_cache, write_df_cache

# Browser-like headers help avoid CDN blocks
DEFAULT_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
}

NBA_PROXY = os.getenv("NBA_PROXY")  # optional proxy like https://user:pass@host:port
CACHE_HOURS = float(os.getenv("NBA_CACHE_HOURS", "6"))

def _nba_sleep():
    time.sleep(0.6)

def _teams_by_fullname() -> Dict[str, Dict]:
    res = static_teams.get_teams()
    return {t["full_name"]: t for t in res}

def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    return float(df[col].mean()) if col in df.columns and not df[col].isna().all() else default

def _safe_latest(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    return float(df[col].iloc[0]) if col in df.columns and len(df) else default

def _fetch_team_gamelog(team_id: int, season: Optional[str], attempts: int = 2) -> pd.DataFrame:
    # 1) try fresh fetch with small retries
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            _nba_sleep()
            gl = TeamGameLog(
                team_id=team_id,
                season=season,
                headers=DEFAULT_HEADERS,
                proxy=NBA_PROXY,
                timeout=20,
            )
            df = gl.get_data_frames()[0]
            if not df.empty:
                write_df_cache(team_id, df)
                return df
        except Exception as e:
            last_exc = e
            time.sleep(1.0 + i)

    # 2) fallback: serve cached data even if stale
    cached = read_df_cache(team_id, max_age_hours=9999)  # any age ok as last resort
    if cached is not None:
        print(f"[warn] using stale cache for team_id={team_id} due to fetch error: {last_exc}")
        return cached

    # 3) give up
    print(f"[warn] nba_api fetch failed for team_id={team_id}: {last_exc}")
    return pd.DataFrame()

def rolling_team_features(
    team_full_name: str,
    season: str | None = None,
    last_n: int = 10,
) -> Dict[str, float]:
    tmap = _teams_by_fullname()
    if team_full_name not in tmap:
        return {}

    team_id = tmap[team_full_name]["id"]

    # Try valid cache first
    cached = read_df_cache(team_id, max_age_hours=CACHE_HOURS)
    if cached is not None:
        df = cached
    else:
        df = _fetch_team_gamelog(team_id, season=season, attempts=2)

    if df.empty:
        return {}

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=False)

    df = df.head(last_n).reset_index(drop=True)
    if df.empty:
        return {}

    feats = {
        "pts_avg": _safe_mean(df, "PTS"),
        "reb_avg": _safe_mean(df, "REB"),
        "ast_avg": _safe_mean(df, "AST"),
        "stl_avg": _safe_mean(df, "STL"),
        "blk_avg": _safe_mean(df, "BLK"),
        "tov_avg": _safe_mean(df, "TOV"),
        "fg_pct_avg": _safe_mean(df, "FG_PCT"),
        "fg3_pct_avg": _safe_mean(df, "FG3_PCT"),
        "ft_pct_avg": _safe_mean(df, "FT_PCT"),
        "plus_minus_avg": _safe_mean(df, "PLUS_MINUS"),
        "pace_proxy_avg": float(
            (
                (_safe_mean(df, "FGA") if "FGA" in df.columns else 0.0)
                + 0.44 * (_safe_mean(df, "FTA") if "FTA" in df.columns else 0.0)
                + _safe_mean(df, "TOV")
            )
        ),
        "last_game_pts": _safe_latest(df, "PTS"),
    }

    if "PLUS_MINUS" in df.columns:
        opp_pts = float((df["PTS"] - df["PLUS_MINUS"]).mean())
    else:
        opp_pts = 0.0
    feats["opp_pts_avg"] = opp_pts

    if "MATCHUP" in df.columns and df["MATCHUP"].dtype == object:
        home_mask = df["MATCHUP"].str.contains(" vs. ", na=False)
        away_mask = df["MATCHUP"].str.contains(" @ ", na=False)
        feats["home_rate"] = float(home_mask.mean())
        feats["away_rate"] = float(away_mask.mean())
    else:
        feats["home_rate"] = 0.5
        feats["away_rate"] = 0.5

    feats = {k: float(v) for k, v in feats.items()}
    return feats
