"""Download NBA game results from Basketball-Reference schedule pages.

Usage:
    python src/scrape.py                 # seasons 2015-2027 (default)
    python src/scrape.py --start 2010    # go further back

Each season is cached in data/raw/<season>.csv, so re-running only refetches
seasons that are still in progress. Season N means the N-1/N season
(e.g. 2024 = 2023-24). Basketball-Reference asks scrapers to stay under
20 requests per minute, so this sleeps between requests.
"""

from __future__ import annotations

import argparse
import io
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.basketball-reference.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-game-predictor; educational project)"}
DELAY = 3.5  # seconds between requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "games.csv"


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year + 1 if today.month >= 9 else today.year


def fetch(url: str) -> str:
    for attempt in range(5):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(DELAY)
        return resp.text
    raise RuntimeError(f"Gave up on {url}")


def month_pages(season: int) -> list[str]:
    html = fetch(f"{BASE}/leagues/NBA_{season}_games.html")
    pages = re.findall(rf"/leagues/NBA_{season}_games-[\w-]+\.html", html)
    return list(dict.fromkeys(pages))  # dedupe, keep order


def parse_month(html: str, season: int) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), attrs={"id": "schedule"})
    t = tables[0]
    t = t.rename(columns={
        "Date": "date", "Start (ET)": "start_et",
        "Visitor/Neutral": "away", "PTS": "away_pts",
        "Home/Neutral": "home", "PTS.1": "home_pts",
        "Unnamed: 7": "ot", "Notes": "notes",
    })
    # Drop repeated header rows and "Playoffs" divider rows (older seasons).
    t["date"] = pd.to_datetime(t["date"], format="%a, %b %d, %Y", errors="coerce")
    t = t[t["date"].notna()].copy()
    for col in ("away_pts", "home_pts"):
        t[col] = pd.to_numeric(t[col], errors="coerce")
    for col in ("start_et", "ot", "notes"):
        if col not in t:
            t[col] = pd.NA
    t["season"] = season
    return t[["season", "date", "start_et", "away", "away_pts", "home",
              "home_pts", "ot", "notes"]]


def mark_playoffs(df: pd.DataFrame) -> pd.Series:
    """Flag postseason games (the schedule pages don't label them).

    The regular season's final day always has 5+ games, while playoff days
    never do (8 first-round series -> at most 4 games a day). Play-in games
    are labeled in the notes column and are not counted as playoffs.
    """
    per_day = df.groupby("date").size()
    reg_end = per_day[per_day >= 5].index.max()
    play_in = df.loc[df["notes"].astype(str).str.contains("Play-In"), "date"]
    if not play_in.empty:
        reg_end = max(reg_end, play_in.max())
    return df["date"] > reg_end


def scrape_season(season: int) -> pd.DataFrame:
    frames = []
    for page in month_pages(season):
        print(f"  {page}")
        frames.append(parse_month(fetch(BASE + page), season))
    df = pd.concat(frames, ignore_index=True)
    df["playoffs"] = mark_playoffs(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=current_season())
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    live = current_season()
    frames = []
    for season in range(args.start, args.end + 1):
        cache = RAW_DIR / f"{season}.csv"
        if cache.exists() and season < live:
            print(f"{season}: cached")
            df = pd.read_csv(cache, parse_dates=["date"])
        else:
            print(f"{season}: downloading")
            df = scrape_season(season)
            df.to_csv(cache, index=False)
        frames.append(df)

    games = pd.concat(frames, ignore_index=True).sort_values(["date", "start_et"], kind="stable")
    games.to_csv(OUT_FILE, index=False)
    played = games["home_pts"].notna().sum()
    print(f"Saved {len(games)} games ({played} played) to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
