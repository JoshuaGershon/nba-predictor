from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
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

# ---------- one-time model bootstrap ----------
MODEL = WinProbModel()


# ---------- pipeline stages ----------

def fetch_odds(_: Dict[str, Any]) -> Dict[str, Any]:
    oc = OddsClient()
    games = oc.get_nba_odds(markets=["h2h", "spreads", "totals"])
    # keep only games with both team names
    games = [g for g in games if g.get("home_team") and g.get("away_team")]
    return {"games": games}


def add_team_features(inp: Dict[str, Any]) -> Dict[str, Any]:
    # Fast path: skip nba_api calls entirely if flagged
    if os.getenv("SKIP_STATS") == "1":
        out = []
        for g in inp["games"]:
            out.append({**g, "home_feats": {}, "away_feats": {}})
        return {"games": out}

    # Otherwise fetch rolling features normally
    out = []
    for g in inp["games"]:
        home = g["home_team"]
        away = g["away_team"]
        home_feats = rolling_team_features(home, season=None, last_n=10)
        away_feats = rolling_team_features(away, season=None, last_n=10)
        out.append({**g, "home_feats": home_feats, "away_feats": away_feats})
    return {"games": out}

def to_eastern(iso_str: str | None) -> str | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%Y-%m-%d %I:%M %p ET")
    except Exception:
        return iso_str

def score_vs_market(inp: Dict[str, Any]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []
    for g in inp["games"]:
        home = g["home_team"]
        away = g["away_team"]

        # build features for model
        row = build_feature_row(home, away, g["home_feats"], g["away_feats"])

        # model probabilities
        p_home = MODEL.predict_proba(row)

        # manual injury/absence adjustment from injuries.json
        p_home = adjust_home_prob(home, away, p_home)
        p_away = 1.0 - p_home

        # market
        home_ml = g["markets"]["h2h"]["home_price"]
        away_ml = g["markets"]["h2h"]["away_price"]
        imp_home = american_to_implied(home_ml)
        imp_away = american_to_implied(away_ml)

        # EV + Kelly
        ev_home = ev_away = None
        kelly_home = kelly_away = None
        if home_ml is not None:
            ev_home = ev_from_prob(p_home, home_ml)
            kelly_home = kelly_fraction(p_home, home_ml)
        if away_ml is not None:
            ev_away = ev_from_prob(p_away, away_ml)
            kelly_away = kelly_fraction(p_away, away_ml)

        rec = {
            "commence_time": to_eastern(g.get("commence_time")),
            "home": home,
            "away": away,
            "moneyline": {"home": home_ml, "away": away_ml},
            "spreads": g["markets"]["spreads"],
            "totals": g["markets"]["totals"],
            "model": {
                "home_win_prob": round(p_home, 4),
                "away_win_prob": round(p_away, 4),
            },
            "market_implied": {
                "home": round(imp_home, 4) if imp_home is not None else None,
                "away": round(imp_away, 4) if imp_away is not None else None,
            },
            "edge": {
                "home": round((p_home - (imp_home or 0.0)), 4) if imp_home is not None else None,
                "away": round((p_away - (imp_away or 0.0)), 4) if imp_away is not None else None,
            },
            "ev": {
                "home": round(ev_home, 4) if ev_home is not None else None,
                "away": round(ev_away, 4) if ev_away is not None else None,
            },
            "kelly": {
                "home": round(kelly_home, 4) if kelly_home is not None else None,
                "away": round(kelly_away, 4) if kelly_away is not None else None,
            },
        }
        scored.append(rec)
    return {"scored": scored}


def rank_recs(inp: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for g in inp["scored"]:
        home_ev = g["ev"]["home"]
        away_ev = g["ev"]["away"]
        if home_ev is not None or away_ev is not None:
            best_side = "home" if (home_ev or -9) >= (away_ev or -9) else "away"
            best_metric = home_ev if best_side == "home" else away_ev
            metric_name = "EV"
        else:
            home_edge = g["edge"]["home"]
            away_edge = g["edge"]["away"]
            best_side = "home" if (home_edge or -9) >= (away_edge or -9) else "away"
            best_metric = home_edge if best_side == "home" else away_edge
            metric_name = "edge"
        items.append({**g, "best_side": best_side, "best_metric": best_metric, "metric_name": metric_name})

    items.sort(key=lambda r: (r["best_metric"] if r["best_metric"] is not None else -999), reverse=True)
    return {"recommendations": items}


# ---------- public runnable ----------
NBA_CHAIN = (
    RunnablePassthrough()
    | RunnableLambda(fetch_odds)
    | RunnableLambda(add_team_features)
    | RunnableLambda(score_vs_market)
    | RunnableLambda(rank_recs)
)
