"""Download pre-game betting lines from ESPN's odds feed.

Usage:
    python src/odds.py                 # every season with box scores
    python src/odds.py --season 2027

For each game we take every sportsbook's final pre-game moneyline,
convert it to a win probability with the bookmaker's margin removed, and
keep the median across books. That's "the market's" probability, the
benchmark the model gets compared against in compare.py.

Cached per season in data/raw/odds/<season>.csv.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from availability import BOX_DIR
from boxscores import API, HEADERS, get_json, team_name

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = ROOT / "data" / "raw" / "odds"
WORKERS = 8

# Not sportsbooks (pick sites, opening lines) or in-game odds, which are set after tip-off.
SKIP = re.compile(r"live|opening|numberfire|teamrankings|accuscore|consensus", re.I)

COLS = ["espn_id", "home_ml", "away_ml", "p_market", "spread", "books"]


def implied(ml: float) -> float:
    """American moneyline -> implied probability (still including the bookmaker's margin)."""
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def to_american(p: float) -> int:
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def game_odds(espn_id: str) -> dict:
    row = {"espn_id": espn_id, "home_ml": np.nan, "away_ml": np.nan, "p_market": np.nan,
           "spread": np.nan, "books": 0}
    try:
        items = requests.get(CORE.format(espn_id), headers=HEADERS, timeout=30).json().get("items", [])
    except (requests.RequestException, ValueError):
        return row
    probs, home_imp, away_imp, spreads = [], [], [], []
    for it in items:
        if SKIP.search(it.get("provider", {}).get("name", "")):
            continue
        h = (it.get("homeTeamOdds") or {}).get("moneyLine")
        a = (it.get("awayTeamOdds") or {}).get("moneyLine")
        if h is None or a is None or abs(h) < 100 or abs(a) < 100:
            continue
        ih, ia = implied(h), implied(a)
        if not 1.0 <= ih + ia <= 1.15:  # a real two-way market has a small margin
            continue
        probs.append(ih / (ih + ia))
        home_imp.append(ih)
        away_imp.append(ia)
        if it.get("spread") is not None:
            spreads.append(it["spread"])
    if probs:
        # Take medians in probability space. American odds jump from +100 to -100,
        # so the median of e.g. [-116, +100] as raw numbers is meaningless.
        row.update(home_ml=to_american(float(np.median(home_imp))), away_ml=to_american(float(np.median(away_imp))),
                   p_market=round(float(np.median(probs)), 4),
                   spread=float(np.median(spreads)) if spreads else np.nan, books=len(probs))
    return row


def scrape_season(season: int) -> pd.DataFrame:
    path = ODDS_DIR / f"{season}.csv"
    box = pd.read_csv(BOX_DIR / f"{season}.csv.gz", dtype={"espn_id": str}, usecols=["espn_id"])
    old = pd.read_csv(path, dtype={"espn_id": str}) if path.exists() else pd.DataFrame(columns=COLS)
    todo = sorted(set(box["espn_id"]) - set(old["espn_id"]))
    if not todo:
        print(f"{season}: up to date ({len(old)} games)")
        return old
    print(f"{season}: fetching odds for {len(todo)} games")
    with ThreadPoolExecutor(WORKERS) as pool:
        new = pd.DataFrame(list(pool.map(game_odds, todo)), columns=COLS)
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    out.sort_values("espn_id").to_csv(path, index=False)
    print(f"  {new['p_market'].notna().mean():.0%} of new games have a line")
    return out


def upcoming_odds(days: list[pd.Timestamp]) -> dict[tuple[pd.Timestamp, str], dict]:
    """Current lines for scheduled games, keyed by (date, home team)."""
    events = []
    for day in days:
        for e in get_json(f"{API}/scoreboard?dates={day:%Y%m%d}&limit=50").get("events", []):
            comp = e["competitions"][0]
            if comp["status"]["type"]["state"] != "pre":
                continue
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            events.append((day, team_name(home["team"]), e["id"]))
    with ThreadPoolExecutor(WORKERS) as pool:
        lines = list(pool.map(game_odds, [eid for _, _, eid in events]))
    return {(day, home): o for (day, home, _), o in zip(events, lines) if pd.notna(o["p_market"])}


def load_odds() -> pd.DataFrame:
    """Market odds joined to Basketball-Reference game ids (via the box scores)."""
    from features import load_games  # local import: features imports availability

    games = load_games()
    odds = pd.concat([pd.read_csv(f, dtype={"espn_id": str}) for f in sorted(ODDS_DIR.glob("*.csv"))])
    box = pd.concat([pd.read_csv(f, dtype={"espn_id": str}, usecols=["espn_id", "date", "team"],
                                 parse_dates=["date"]) for f in sorted(BOX_DIR.glob("*.csv.gz"))])
    box = box.drop_duplicates(["espn_id", "team"])
    ids = games[["game_id", "date", "home"]].merge(box, left_on=["date", "home"], right_on=["date", "team"])
    return ids[["game_id", "espn_id"]].drop_duplicates("game_id").merge(odds, on="espn_id")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download betting lines")
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()
    seasons = [args.season] if args.season else sorted(int(p.name[:4]) for p in BOX_DIR.glob("*.csv.gz"))
    for s in seasons:
        scrape_season(s)


if __name__ == "__main__":
    main()
