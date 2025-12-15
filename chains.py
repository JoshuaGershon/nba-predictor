from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from tools.odds_tool import OddsClient
from tools.stats_tool import rolling_team_features
from tools.features import (
    build_feature_row,
    american_to_implied,
    ev_from_prob,
    kelly_fraction,
)
from tools.injuries import adjust_home_prob
from tools.model import WinProbModel


# ============================
# One-time model bootstrap
# ============================

MODEL = WinProbModel()


# ============================
# Helpers
# ============================

def to_eastern(iso_str: str | None) -> str | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%Y-%m-%d %I:%M %p ET")
    except Exception:
        return iso_str


# ============================
# Pipeline stages
# ============================

def fetch_odds(_: Dict[str, Any]) -> Dict[str, Any]:
    oc = OddsClient()
    games = oc.get_nba_odds(markets=["h2h", "spreads", "totals"])

    # Keep only valid games
    games = [g for g in games if g.get("home_team") and g.get("away_team")]
    return {"games": games}


def add_team_features(inp: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("SKIP_STATS") == "1":
        return {
            "games": [
                {**g, "home_feats": {}, "away_feats": {}}
                for g in inp["games"]
            ]
        }

    out = []
    for g in inp["games"]:
        home = g["home_team"]
        away = g["away_team"]

        home_feats = rolling_team_features(home, season=None, last_n=10)
        away_feats = rolling_team_features(away, season=None, last_n=10)

        out.append({**g, "home_feats": home_feats, "away_feats": away_feats})

    return {"games": out}


def score_vs_market(inp: Dict[str, Any]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []

    for g in inp["games"]:
        home = g["home_team"]
        away = g["away_team"]

        # ============================
        # Model probability
        # ============================

        row = build_feature_row(home, away, g["home_feats"], g["away_feats"])
        p_home = MODEL.predict_proba(row)
        p_home = adjust_home_prob(home, away, p_home)
        p_away = 1.0 - p_home

        # ============================
        # MONEYLINE MARKET
        # ============================

        ml_home = g["markets"]["h2h"]["home_price"]
        ml_away = g["markets"]["h2h"]["away_price"]

        ev_ml_home = ev_ml_away = None
        k_ml_home = k_ml_away = None

        if ml_home is not None:
            ev_ml_home = ev_from_prob(p_home, ml_home)
            k_ml_home = kelly_fraction(p_home, ml_home)

        if ml_away is not None:
            ev_ml_away = ev_from_prob(p_away, ml_away)
            k_ml_away = kelly_fraction(p_away, ml_away)

        # ============================
        # SPREAD MARKET (new)
        # ============================

        spread = g["markets"]["spreads"]
        spread_home = spread.get("home_point")
        spread_away = spread.get("away_point")
        spread_odds_home = spread.get("home_price")
        spread_odds_away = spread.get("away_price")

        # Simple heuristic: reuse win prob as proxy
        # (we’ll improve this later with margin models)
        ev_spread_home = ev_spread_away = None
        k_spread_home = k_spread_away = None

        if spread_odds_home is not None:
            ev_spread_home = ev_from_prob(p_home, spread_odds_home)
            k_spread_home = kelly_fraction(p_home, spread_odds_home)

        if spread_odds_away is not None:
            ev_spread_away = ev_from_prob(p_away, spread_odds_away)
            k_spread_away = kelly_fraction(p_away, spread_odds_away)

        scored.append(
            {
                "commence_time": to_eastern(g.get("commence_time")),
                "home": home,
                "away": away,

                "model": {
                    "p_home": round(p_home, 4),
                    "p_away": round(p_away, 4),
                },

                "moneyline": {
                    "home": ml_home,
                    "away": ml_away,
                    "ev_home": ev_ml_home,
                    "ev_away": ev_ml_away,
                    "kelly_home": k_ml_home,
                    "kelly_away": k_ml_away,
                },

                "spread": {
                    "line_home": spread_home,
                    "line_away": spread_away,
                    "price_home": spread_odds_home,
                    "price_away": spread_odds_away,
                    "ev_home": ev_spread_home,
                    "ev_away": ev_spread_away,
                    "kelly_home": k_spread_home,
                    "kelly_away": k_spread_away,
                },
            }
        )

    return {"scored": scored}


def rank_recs(inp: Dict[str, Any]) -> Dict[str, Any]:
    ranked = []

    for g in inp["scored"]:
        candidates = []

        # Moneyline sides
        if g["moneyline"]["ev_home"] is not None:
            candidates.append(("moneyline", "home", g["moneyline"]["ev_home"]))
        if g["moneyline"]["ev_away"] is not None:
            candidates.append(("moneyline", "away", g["moneyline"]["ev_away"]))

        # Spread sides
        if g["spread"]["ev_home"] is not None:
            candidates.append(("spread", "home", g["spread"]["ev_home"]))
        if g["spread"]["ev_away"] is not None:
            candidates.append(("spread", "away", g["spread"]["ev_away"]))

        if not candidates:
            continue

        best_market, best_side, best_ev = max(candidates, key=lambda x: x[2])

        ranked.append(
            {
                **g,
                "best_market": best_market,
                "best_side": best_side,
                "best_ev": best_ev,
            }
        )

    ranked.sort(key=lambda r: r["best_ev"], reverse=True)
    return {"recommendations": ranked}


# ============================
# Public runnable
# ============================

NBA_CHAIN = (
    RunnablePassthrough()
    | RunnableLambda(fetch_odds)
    | RunnableLambda(add_team_features)
    | RunnableLambda(score_vs_market)
    | RunnableLambda(rank_recs)
)
