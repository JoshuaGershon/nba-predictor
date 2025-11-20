from __future__ import annotations
import os, time
import pandas as pd
from typing import Optional

CACHE_DIR = os.path.join(".cache", "nba")
os.makedirs(CACHE_DIR, exist_ok=True)

def _team_path(team_id: int) -> str:
    return os.path.join(CACHE_DIR, f"team_gamelog_{team_id}.csv")

def read_df_cache(team_id: int, max_age_hours: float) -> Optional[pd.DataFrame]:
    path = _team_path(team_id)
    if not os.path.exists(path):
        return None
    age_sec = time.time() - os.path.getmtime(path)
    if age_sec > max_age_hours * 3600:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def write_df_cache(team_id: int, df: pd.DataFrame) -> None:
    path = _team_path(team_id)
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass
