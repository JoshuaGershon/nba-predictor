from __future__ import annotations
import os
import sys
import traceback
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import pytz

# Make sure we can import from the project root
sys.path.append(os.path.dirname(__file__))

from chains import NBA_CHAIN  # uses your trained model + odds API

# ---------- env + page setup ----------

# Load .env once (for default ODDS_API_KEY, BANKROLL, etc.)
load_dotenv(override=False)

st.set_page_config(page_title="NBA Game Predictor", layout="wide")

st.title("NBA Game Predictor")
st.markdown(
    """
This app:

- Uses a model trained on historical NBA games (Kaggle data).
- Pulls **live** NBA moneylines and spreads from The Odds API.
- Calculates win probabilities, expected value (EV), and Kelly bet sizes.
- Returns **recommended sides** based on EV and Kelly.
"""
)

# Time-zone helpers (Odds API returns UTC)
UTC = pytz.timezone("UTC")
EASTERN = pytz.timezone("US/Eastern")


def utc_to_eastern_str(ts: str) -> str:
    """Convert Odds API UTC timestamp to a nice local (ET) string."""
    if not ts:
        return ""
    try:
        dt_utc = datetime.fromisoformat(ts.replace("Z", "")).replace(tzinfo=UTC)
        dt_local = dt_utc.astimezone(EASTERN)
        return dt_local.strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        # Fallback: just return raw string if parsing fails
        return ts


# ---------- sidebar controls ----------

with st.sidebar:
    st.header("Settings")

    default_key = os.getenv("ODDS_API_KEY", "")
    api_key = st.text_input("Odds API Key", value=default_key, type="password")
    if api_key and api_key != os.getenv("ODDS_API_KEY"):
        os.environ["ODDS_API_KEY"] = api_key

    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=0.0,
        value=float(os.getenv("BANKROLL", "1000") or 1000.0),
    )
    os.environ["BANKROLL"] = str(bankroll)

    half_kelly = st.toggle("Use half Kelly", value=(os.getenv("HALF_KELLY", "1") == "1"))
    os.environ["HALF_KELLY"] = "1" if half_kelly else "0"

    min_ev = st.number_input("Filter: min EV", value=0.0, step=0.1)
    min_kelly = st.number_input("Filter: min Kelly fraction", value=0.0, step=0.05)


# ---------- helpers ----------

def passes_filters(rec: Dict[str, Any], min_ev: float, min_kelly: float) -> bool:
    ev_home = rec["ev"]["home"] if rec["ev"]["home"] is not None else -999.0
    ev_away = rec["ev"]["away"] if rec["ev"]["away"] is not None else -999.0

    k_home = rec["kelly"]["home"] if rec["kelly"]["home"] is not None else 0.0
    k_away = rec["kelly"]["away"] if rec["kelly"]["away"] is not None else 0.0

    return (max(ev_home, ev_away) >= min_ev) and (max(k_home, k_away) >= min_kelly)


# ---------- main action ----------

if st.button("Get Picks", type="primary"):
    with st.spinner("Running model on live odds..."):
        try:
            out = NBA_CHAIN.invoke({})
        except Exception as e:
            st.error(
                "Error while running the prediction chain:\n\n"
                + "".join(traceback.format_exception_only(type(e), e))
            )
            st.stop()

    recs: List[Dict[str, Any]] = out.get("recommendations", [])

    # Apply EV / Kelly filters
    recs = [r for r in recs if passes_filters(r, min_ev, min_kelly)]

    st.subheader(f"Total recommendations: {len(recs)}")

    if not recs:
        st.info("No bets passed the filters. Try lowering min EV or min Kelly.")
    else:
        hk = os.getenv("HALF_KELLY", "0") == "1"
        k_factor = 0.5 if hk else 1.0
        bk = float(os.getenv("BANKROLL", "0") or 0.0)

        rows: List[Dict[str, Any]] = []

        for r in recs:
            k_home = r["kelly"]["home"] or 0.0
            k_away = r["kelly"]["away"] or 0.0

            stake_home = round(bk * k_home * k_factor, 2) if bk else 0.0
            stake_away = round(bk * k_away * k_factor, 2) if bk else 0.0

            best_side = r.get("best_side", "home")

            rows.append(
                {
                    "Tip (ET)": r.get("commence_time"),  # already converted in chains
                    "Away": r["away"],
                    "Home": r["home"],
                    "ML home": r["moneyline"]["home"],
                    "ML away": r["moneyline"]["away"],
                    "P(home)": r["model"]["home_win_prob"],
                    "P(away)": r["model"]["away_win_prob"],
                    "EV home": r["ev"]["home"],
                    "EV away": r["ev"]["away"],
                    "Kelly home": r["kelly"]["home"],
                    "Kelly away": r["kelly"]["away"],
                    "Stake home ($)": stake_home,
                    "Stake away ($)": stake_away,
                    "Recommended pick": (
                        f"{r['home']} (HOME)" if best_side == "home"
                        else f"{r['away']} (AWAY)"
                    ),
                    "Rec EV": r["best_metric"],
                }
            )

        df = pd.DataFrame(rows)

        # Order columns nicely
        cols = [
            "Tip (ET)",
            "Away",
            "Home",
            "ML home",
            "ML away",
            "P(home)",
            "P(away)",
            "EV home",
            "EV away",
            "Kelly home",
            "Kelly away",
            "Stake home ($)",
            "Stake away ($)",
            "Recommended pick",
            "Rec EV",
        ]
        df = df[cols]

        # Highlight rows based on Rec EV
        def highlight_row(row):
            val = row["Rec EV"]
            if pd.isna(val):
                return [""] * len(row)
            if val > 0:
                color = "background-color: #d4f9d4"  # light green
            elif val < 0:
                color = "background-color: #fddede"  # light red
            else:
                color = ""
            return [color] * len(row)

        styled = df.style.apply(highlight_row, axis=1)

        st.dataframe(styled, use_container_width=True)

        # CSV download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV of picks",
            data=csv_bytes,
            file_name="nba_picks.csv",
            mime="text/csv",
        )

st.caption(
    "User only needs an Odds API key in the sidebar. "
    "All modeling, odds fetching, EV and Kelly math happens automatically."
)
