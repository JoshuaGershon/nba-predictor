from __future__ import annotations
import os
import sys
import traceback
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Make sure we can import from the project root
sys.path.append(os.path.dirname(__file__))

from chains import NBA_CHAIN  # model + odds pipeline

# ---------- ENV + PAGE SETUP ----------

load_dotenv(override=False)

st.set_page_config(page_title="NBA Game Predictor", layout="wide")

st.title("NBA Game Predictor")
st.markdown(
    """
This app:

- Uses a model trained on historical NBA games (Kaggle data)
- Pulls **live NBA moneylines and spreads** from The Odds API
- Calculates win probabilities, **expected value (EV)**, and **unit-based sizing**
- Returns **recommended bets** with clear unit sizing
"""
)

# ---------- HELPERS ----------

def fmt_american(odds: int | None) -> str | None:
    """Ensure + sign is shown for positive odds."""
    if odds is None:
        return None
    return f"+{odds}" if odds > 0 else str(odds)


def passes_filters(rec: Dict[str, Any], min_ev: float) -> bool:
    ev_home = rec["ev"]["home"] if rec["ev"]["home"] is not None else -999.0
    ev_away = rec["ev"]["away"] if rec["ev"]["away"] is not None else -999.0
    return max(ev_home, ev_away) >= min_ev


# ---------- SIDEBAR ----------

with st.sidebar:
    st.header("Settings")

    # Odds API
    api_key = st.text_input(
        "Odds API Key",
        value=os.getenv("ODDS_API_KEY", ""),
        type="password",
    )
    if api_key:
        os.environ["ODDS_API_KEY"] = api_key

    # Units instead of bankroll
    unit_size = st.number_input(
        "1 Unit = $",
        min_value=1.0,
        value=float(os.getenv("UNIT_SIZE", "100") or 100),
        step=10.0,
    )
    os.environ["UNIT_SIZE"] = str(unit_size)

    use_half_kelly = st.toggle(
        "Use half Kelly",
        value=(os.getenv("HALF_KELLY", "1") == "1"),
    )
    os.environ["HALF_KELLY"] = "1" if use_half_kelly else "0"

    min_ev = st.number_input("Filter: min EV", value=0.0, step=0.05)


# ---------- MAIN ----------

if st.button("Get Picks", type="primary"):
    with st.spinner("Running model on live odds..."):
        try:
            out = NBA_CHAIN.invoke({})
        except Exception as e:
            st.error(
                "Error while running prediction:\n\n"
                + "".join(traceback.format_exception_only(type(e), e))
            )
            st.stop()

    recs: List[Dict[str, Any]] = out.get("recommendations", [])
    recs = [r for r in recs if passes_filters(r, min_ev)]

    st.subheader(f"Total recommendations: {len(recs)}")

    if not recs:
        st.info("No bets passed the filter.")
        st.stop()

    rows: List[Dict[str, Any]] = []

    half_factor = 0.5 if os.getenv("HALF_KELLY") == "1" else 1.0

    for r in recs:
        best_side = r["best_side"]
        rec_ev = r["best_metric"]

        # Kelly → Units (capped + rounded)
        raw_kelly = r["kelly"][best_side] or 0.0
        units = round(raw_kelly * 10 * half_factor, 1)  # scale Kelly to units
        units = max(units, 0)

        dollar_example = round(units * unit_size, 2)

        rows.append(
            {
                "Tip (ET)": r["commence_time"],
                "Away": r["away"],
                "Home": r["home"],

                # Moneylines (formatted)
                "ML Away": fmt_american(r["moneyline"]["away"]),
                "ML Home": fmt_american(r["moneyline"]["home"]),

                # Model
                "P(Home)": r["model"]["home_win_prob"],
                "P(Away)": r["model"]["away_win_prob"],

                # EV
                "EV Home": r["ev"]["home"],
                "EV Away": r["ev"]["away"],

                # Recommendation
                "Recommended Bet": (
                    f"{r['home']} ML" if best_side == "home"
                    else f"{r['away']} ML"
                ),
                "Units": units,
                "Example $ Bet": f"${dollar_example}",
                "Rec EV": rec_ev,
            }
        )

    df = pd.DataFrame(rows)

    # Order columns
    df = df[
        [
            "Tip (ET)",
            "Away",
            "Home",
            "ML Away",
            "ML Home",
            "P(Home)",
            "P(Away)",
            "EV Away",
            "EV Home",
            "Recommended Bet",
            "Units",
            "Example $ Bet",
            "Rec EV",
        ]
    ]

    # ---------- STYLING ----------

    def highlight_row(row):
        val = row["Rec EV"]
        if pd.isna(val):
            return [""] * len(row)
        if val > 0:
            return ["background-color: #d4f9d4"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

    # Download
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "nba_picks.csv",
        "text/csv",
    )

st.caption(
    "Set your unit size in the sidebar. Model outputs bets in units with a dollar example."
)
