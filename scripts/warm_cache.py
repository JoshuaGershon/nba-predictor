from __future__ import annotations
# ensure project root is on sys.path
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.odds_tool import OddsClient
from tools.stats_tool import rolling_team_features

def main():
    oc = OddsClient()
    games = oc.get_nba_odds(markets=["h2h"])
    teams = set()
    for g in games:
        if g.get("home_team"): teams.add(g["home_team"])
        if g.get("away_team"): teams.add(g["away_team"])

    teams = sorted(teams)
    print(f"warming cache for {len(teams)} teams...")

    ok = fail = 0
    for t in teams:
        try:
            feats = rolling_team_features(t, season=None, last_n=10)
            if feats:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"[warn] warm failed for {t}: {e}")
        time.sleep(0.2)  # be polite
    print(f"done. ok={ok} fail={fail} (some fails are normal; rerun once).")

if __name__ == "__main__":
    main()
