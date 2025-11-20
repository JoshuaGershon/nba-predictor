from __future__ import annotations
import json
import os

_DEFAULT_PATH = os.getenv("INJURIES_FILE", "injuries.json")

def _load_adjustments(path: str = _DEFAULT_PATH) -> dict[str, float]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return hi if x > hi else lo if x < lo else x

def adjust_home_prob(home_team: str, away_team: str, p_home: float) -> float:
    """
    Linear shift using per-team adjustments from injuries.json.
    Positive value favors that team by that many probability points.
    Example:
    {
      "Boston Celtics": -0.06,
      "Memphis Grizzlies": 0.03
    }
    """
    adj = _load_adjustments()
    dh = float(adj.get(home_team, 0.0))
    da = float(adj.get(away_team, 0.0))
    return clamp(p_home + dh - da, 0.01, 0.99)
