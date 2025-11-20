from __future__ import annotations
from typing import Dict, Any, Optional

import pandas as pd

# -------------------------------------------------------------------
# Team ID mapping using data/teams.csv from the Kaggle dataset
# -------------------------------------------------------------------

TEAM_NAME_TO_ID: Dict[str, int] = {}

try:
    teams_df = pd.read_csv("data/teams.csv")

    # Build full name as "CITY NICKNAME", e.g. "Denver Nuggets"
    if {"CITY", "NICKNAME", "TEAM_ID"}.issubset(teams_df.columns):
        for _, row in teams_df.iterrows():
            city = str(row["CITY"]).strip()
            nickname = str(row["NICKNAME"]).strip()
            if not city or not nickname:
                continue
            full_name = f"{city} {nickname}"  # e.g. "New Orleans Pelicans"
            TEAM_NAME_TO_ID[full_name] = int(row["TEAM_ID"])
        print(f"[features] Loaded {len(TEAM_NAME_TO_ID)} team name → id mappings from CITY+NICKNAME.")
    else:
        print("[features] WARNING: teams.csv missing CITY/NICKNAME/TEAM_ID columns.")
except Exception as e:
    print("[features] WARNING: could not load data/teams.csv:", e)


def get_team_id(team_name: str) -> Optional[int]:
    """Map team name from odds API (e.g. 'New Orleans Pelicans') to TEAM_ID."""
    if not team_name:
        return None
    name = team_name.strip()
    if name in TEAM_NAME_TO_ID:
        return TEAM_NAME_TO_ID[name]
    print(f"[features] WARNING: no TEAM_ID found for '{team_name}'")
    return None


# -------------------------------------------------------------------
# Odds helper functions (used by the chain)
# -------------------------------------------------------------------

def american_to_implied(odds: Optional[int]) -> Optional[float]:
    """Convert American odds (e.g. -150, +200) to implied probability."""
    if odds is None:
        return None
    odds = int(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return None


def ev_from_prob(p: float, odds: int) -> float:
    """
    Expected value per 1 unit stake given win prob p and American odds.

    EV = p * profit_if_win - (1 - p) * 1
    """
    odds = int(odds)
    if odds > 0:
        profit = odds / 100.0        # bet 100 to win odds
    else:
        profit = 100.0 / (-odds)     # bet -odds to win 100

    return p * profit - (1.0 - p)


def kelly_fraction(p: float, odds: int) -> float:
    """
    Kelly fraction of bankroll to bet for edge (p, odds).
    Returns 0 if edge is negative or invalid.
    """
    odds = int(odds)
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / (-odds)

    q = 1.0 - p
    numer = p * (b + 1.0) - 1.0
    denom = b
    if denom == 0:
        return 0.0
    f = numer / denom
    return max(0.0, f)


# -------------------------------------------------------------------
# Feature builder used by WinProbModel (team ID based)
# -------------------------------------------------------------------

def build_feature_row(
    home_team: str,
    away_team: str,
    home_feats: Optional[Dict[str, Any]] = None,
    away_feats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the feature row dict that the WinProbModel expects.

    Our RandomForest was trained on:
        HOME_TEAM_ID, VISITOR_TEAM_ID
    """
    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)

    # Fallback: if mapping is missing, use special values so it still runs
    if home_id is None:
        home_id = -1
    if away_id is None:
        away_id = -2

    return {
        "home_team_id": home_id,
        "away_team_id": away_id,
    }
