from __future__ import annotations
from pprint import pprint
import argparse, sys, traceback
from chains import NBA_CHAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="show top N recommendations")
    args = ap.parse_args()

    try:
        result = NBA_CHAIN.invoke({})
    except Exception as e:
        print("ERROR while running chain:")
        traceback.print_exc()
        sys.exit(1)

    recs = (result or {}).get("recommendations", [])
    print(f"total recommendations: {len(recs)}")
    if not recs:
        print("No games returned (could be API limits, no NBA games at this time, or key issue).")
        sys.exit(0)

    for r in recs[: args.top]:
        print("=" * 70)
        print(f"{r['away']} at {r['home']}  tip={r.get('commence_time')}")
        print(f"moneyline: home={r['moneyline']['home']}  away={r['moneyline']['away']}")
        print(f"spreads:   {r['spreads']}")
        print(f"totals:    {r['totals']}")
        print(
            f"model:     home_win_prob={r['model']['home_win_prob']:.3f}  "
            f"away_win_prob={r['model']['away_win_prob']:.3f}"
        )
        print(f"implied:   home={r['market_implied']['home']}  away={r['market_implied']['away']}")
        print(f"edge:      home={r['edge']['home']}  away={r['edge']['away']}")
        print(f"EV:        home={r['ev']['home']}   away={r['ev']['away']}")
        print(f"KELLY:     home={r['kelly']['home']}   away={r['kelly']['away']}")
        print(f"RECOMMEND: bet_{r['best_side']}  by {r['metric_name']}={r['best_metric']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
