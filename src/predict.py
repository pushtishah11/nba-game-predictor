"""Predict upcoming NBA games with the trained model.

Usage:
    python src/predict.py                  # next 3 days of games
    python src/predict.py --days 7
    python src/predict.py --date 2026-12-25

Refresh first to get the latest results and schedule:
    python src/scrape.py && python src/features.py
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from features import build

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "model.joblib"


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict upcoming NBA games")
    ap.add_argument("--date", type=date.fromisoformat, help="first day to predict (default: today)")
    ap.add_argument("--days", type=int, default=3, help="how many days to cover")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_FILE)
    df = build()  # recompute so predictions use the latest results
    start = pd.Timestamp(args.date or date.today())
    end = start + timedelta(days=args.days - 1)
    games = df[~df["played"] & df["date"].between(start, end)].copy()

    if games.empty:
        print(f"No unplayed games between {start.date()} and {end.date()}.")
        print("Run `python src/scrape.py` to refresh the schedule.")
        return

    games["p_home"] = bundle["model"].predict_proba(games[bundle["features"]])[:, 1]
    print(f"Model trained through {bundle['trained_through']}\n")
    for day, grp in games.groupby("date"):
        print(day.strftime("%a %b %d, %Y"))
        for r in grp.itertuples():
            fav, p = (r.home, r.p_home) if r.p_home >= 0.5 else (r.away, 1 - r.p_home)
            print(f"  {r.away:>24} @ {r.home:<24}  pick: {fav} ({p:.0%})")
        print()


if __name__ == "__main__":
    main()
