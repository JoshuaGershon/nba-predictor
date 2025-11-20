from __future__ import annotations
from typing import Dict, Any, List, Optional
import math
import httpx

from .utils import get_env

def _american_payout_per_dollar(american: int | float) -> float:
    """Return net payout for a $1 stake given American odds."""
    a = float(american)
    if a >= 100:
        return a / 100.0
    return 100.0 / abs(a)

def _choose_better_moneyline(current: Optional[int], new: Optional[int]) -> Optional[int]:
    """Pick the line with higher potential payout for the bettor."""
    if new is None:
        return current
    if current is None:
        return new
    return new if _american_payout_per_dollar(new) > _american_payout_per_dollar(current) else current

class OddsClient:
    """
    Thin client for The Odds API.
    Requires:
      ODDS_API_KEY in .env
      ODDS_API_BASE in .env (default https://api.the-odds-api.com/v4)
    """
    def __init__(self):
        self.base = get_env("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
        self.key = get_env("ODDS_API_KEY")

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        params = {"apiKey": self.key, **params}
        url = f"{self.base}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    def get_nba_odds(self, markets: List[str] = ["h2h", "spreads", "totals"]) -> List[Dict[str, Any]]:
        """
        Fetch today’s NBA odds. Free tier generally returns only upcoming events.
        markets: "h2h" (moneyline), "spreads", "totals"
        Returns a list of games with the best available prices across books.
        """
        data = self._get(
            "/sports/basketball_nba/odds",
            {
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )

        games: List[Dict[str, Any]] = []
        for g in data:
            home = g.get("home_team") or g.get("homeTeam")
            away = g.get("away_team") or g.get("awayTeam")

            if not away:
                teams = g.get("teams") or []
                if home and len(teams) == 2:
                    away = teams[0] if teams[0] != home else teams[1]


            rec = {
                "id": g.get("id"),
                "commence_time": g.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "markets": {
                    "h2h": {"home_price": None, "away_price": None},
                    "spreads": {"home_point": None, "home_price": None, "away_point": None, "away_price": None},
                    "totals": {"over_point": None, "over_price": None, "under_point": None, "under_price": None},
                },
            }

            # Aggregate best prices across all bookmakers
            for bk in g.get("bookmakers", []):
                for m in bk.get("markets", []):
                    key = m.get("key")
                    outcomes = m.get("outcomes", []) or []

                    if key == "h2h":
                        # two outcomes: home and away
                        for o in outcomes:
                            name = o.get("name")
                            price = o.get("price")
                            if name == home:
                                rec["markets"]["h2h"]["home_price"] = _choose_better_moneyline(
                                    rec["markets"]["h2h"]["home_price"], price
                                )
                            elif name == away:
                                rec["markets"]["h2h"]["away_price"] = _choose_better_moneyline(
                                    rec["markets"]["h2h"]["away_price"], price
                                )

                    elif key == "spreads":
                        # outcomes carry team name, point, and price
                        # pick best price observed for each side and keep its corresponding point
                        # note: points may vary across books; we keep the point attached to the best price
                        for o in outcomes:
                            name = o.get("name")
                            price = o.get("price")
                            point = o.get("point")
                            if name == home:
                                current = rec["markets"]["spreads"]["home_price"]
                                better = _choose_better_moneyline(current, price)
                                if better != current:
                                    rec["markets"]["spreads"]["home_price"] = better
                                    rec["markets"]["spreads"]["home_point"] = point
                            elif name == away:
                                current = rec["markets"]["spreads"]["away_price"]
                                better = _choose_better_moneyline(current, price)
                                if better != current:
                                    rec["markets"]["spreads"]["away_price"] = better
                                    rec["markets"]["spreads"]["away_point"] = point

                    elif key == "totals":
                        # outcomes are Over and Under with points and price
                        for o in outcomes:
                            desc = (o.get("name") or "").lower()
                            price = o.get("price")
                            point = o.get("point")
                            if "over" in desc:
                                current = rec["markets"]["totals"]["over_price"]
                                better = _choose_better_moneyline(current, price)
                                if better != current:
                                    rec["markets"]["totals"]["over_price"] = better
                                    rec["markets"]["totals"]["over_point"] = point
                            elif "under" in desc:
                                current = rec["markets"]["totals"]["under_price"]
                                better = _choose_better_moneyline(current, price)
                                if better != current:
                                    rec["markets"]["totals"]["under_price"] = better
                                    rec["markets"]["totals"]["under_point"] = point

            games.append(rec)

            # Optional sanity: if a team name is missing for some reason, you can filter it out
            # if not home or not away:
            #     continue

        return games

if __name__ == "__main__":
    # quick manual test: prints a couple of games
    oc = OddsClient()
    games = oc.get_nba_odds()
    for g in games[:5]:
        print(g["away_team"], "at", g["home_team"], g["commence_time"])
        print("  h2h:", g["markets"]["h2h"])
        print("  spreads:", g["markets"]["spreads"])
        print("  totals:", g["markets"]["totals"])
        print("-" * 40)
