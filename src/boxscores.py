"""Download player box scores from ESPN's public JSON API.

Usage:
    python src/boxscores.py                # every season in data/games.csv
    python src/boxscores.py --season 2027  # just one season

Box scores tell us who actually played in each game, which is what the
player availability features in features.py are built from. One row per
player per game, including players who sat (dnp=True).

Each season is cached in data/raw/box/<season>.csv.gz. Re-running only
fetches games that have been played since the last run.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-game-predictor; educational project)"}
WORKERS = 4

ROOT = Path(__file__).resolve().parents[1]
GAMES_FILE = ROOT / "data" / "games.csv"
BOX_DIR = ROOT / "data" / "raw" / "box"

# ESPN display names that differ from Basketball-Reference's
NAME_FIX = {"LA Clippers": "Los Angeles Clippers"}

COLS = ["season", "date", "espn_id", "team", "opp", "player_id", "player",
        "starter", "dnp", "reason", "min", "pts", "gmsc"]


def get_json(url: str) -> dict:
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                raise LookupError(url)
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gave up on {url}")


def team_name(team: dict) -> str:
    return NAME_FIX.get(team["displayName"], team["displayName"])


def events_on(day: pd.Timestamp) -> list[dict]:
    """Finished games on one date: ESPN id plus home and away team names."""
    d = get_json(f"{API}/scoreboard?dates={day:%Y%m%d}&limit=50")
    out = []
    for e in d.get("events", []):
        comp = e["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue
        sides = {c["homeAway"]: team_name(c["team"]) for c in comp["competitors"]}
        out.append({"espn_id": e["id"], "date": day, "home": sides["home"], "away": sides["away"]})
    return out


def game_score(s: dict) -> float:
    """John Hollinger's Game Score: one number for a player's box score line."""
    fgm, fga = map(int, s["FG"].split("-"))
    ftm, fta = map(int, s["FT"].split("-"))
    return (int(s["PTS"]) + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm)
            + 0.7 * int(s["OREB"]) + 0.3 * int(s["DREB"]) + int(s["STL"])
            + 0.7 * int(s["AST"]) + 0.7 * int(s["BLK"]) - 0.4 * int(s["PF"]) - int(s["TO"]))


def box_score(event: dict) -> list[dict]:
    try:
        d = get_json(f"{API}/summary?event={event['espn_id']}")
    except LookupError:
        print(f"  no box score on ESPN for {event['away']} @ {event['home']} on {event['date']:%Y-%m-%d}, skipping")
        return []
    rows = []
    teams = d["boxscore"].get("players", [])
    names = [team_name(t["team"]) for t in teams]
    for t, team, opp in zip(teams, names, names[::-1]):
        block = t["statistics"][0]
        labels = block["labels"]
        for a in block["athletes"]:
            if "id" not in a.get("athlete", {}):
                continue  # rare placeholder rows with no player attached
            played = bool(a["stats"]) and not a.get("didNotPlay")
            s = dict(zip(labels, a["stats"])) if played else {}
            try:
                minutes = int(s.get("MIN", 0) or 0)
                gmsc = round(game_score(s), 1) if played else 0.0
                pts = int(s.get("PTS", 0) or 0)
            except (ValueError, KeyError):  # stat lines with "--"
                minutes, gmsc, pts = 0, 0.0, 0
            rows.append({"date": event["date"], "espn_id": event["espn_id"], "team": team, "opp": opp,
                         "player_id": a["athlete"]["id"], "player": a["athlete"]["displayName"],
                         "starter": bool(a.get("starter")), "dnp": not played or minutes == 0,
                         "reason": a.get("reason") if not played else None,
                         "min": minutes, "pts": pts, "gmsc": gmsc})
    return rows


def scrape_season(season: int, games: pd.DataFrame) -> pd.DataFrame:
    path = BOX_DIR / f"{season}.csv.gz"
    old = pd.read_csv(path, parse_dates=["date"], dtype={"espn_id": str, "player_id": str}) if path.exists() \
        else pd.DataFrame(columns=COLS)
    season_games = games[games["season"] == season]
    played = season_games[season_games["home_pts"].notna()]
    have = set(old["date"].dt.normalize()) if len(old) else set()
    # During a season, refetch the last cached day in case it was saved before
    # every game had finished.
    if have and season_games["home_pts"].isna().any():
        have.discard(max(have))
    days = sorted(set(played["date"]) - have)
    if not days:
        print(f"{season}: up to date ({old['espn_id'].nunique()} games)")
        return old

    with ThreadPoolExecutor(WORKERS) as pool:
        events = [e for day in pool.map(events_on, days) for e in day]
        print(f"{season}: fetching {len(events)} box scores from {len(days)} days")
        rows = [r for game in pool.map(box_score, events) for r in game]

    new = pd.DataFrame(rows, columns=[c for c in COLS if c != "season"]).assign(season=season)
    keep = old[~old["date"].isin(days)]
    out = (pd.concat([keep, new], ignore_index=True) if len(keep) else new)[COLS]
    out = out.sort_values(["date", "espn_id", "team", "starter", "min"], ascending=[True, True, True, False, False])
    BOX_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})  # mtime=0: same data, same bytes
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ESPN box scores")
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()
    games = pd.read_csv(GAMES_FILE, parse_dates=["date"])
    seasons = [args.season] if args.season else sorted(games["season"].unique())
    for s in seasons:
        scrape_season(int(s), games)


if __name__ == "__main__":
    main()
