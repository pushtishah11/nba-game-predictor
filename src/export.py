"""Export predictions and the model's live track record for the website.

Usage:
    python src/export.py                  # predict from today onward
    python src/export.py --today 2026-11-14

Writes docs/data.json (read by docs/index.html) and keeps a log of every
prediction in data/prediction_log.csv. A game's prediction keeps being
refreshed until it has a final score, then it is frozen. Because features
only use completed games, every logged pick was made before tip-off.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

import injuries
from availability import current_values, load_box, player_values
from explain import explain
from features import build

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "model.joblib"
LOG_FILE = ROOT / "data" / "prediction_log.csv"
METRICS_FILE = ROOT / "results" / "metrics.csv"
OUT_FILE = ROOT / "docs" / "data.json"

DAYS_AHEAD = 7
RECENT_DAYS = 7

ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
LOG_COLS = ["key", "date", "away", "home", "p_home", "predicted_at"]


def game_key(df: pd.DataFrame) -> pd.Series:
    return df["date"].dt.strftime("%Y-%m-%d") + "|" + df["away"] + "|" + df["home"]


def tipoff_utc(df: pd.DataFrame) -> pd.Series:
    """Scheduled start in UTC. Times are listed in US Eastern; assume noon if missing."""
    t = df["start_et"].fillna("12:00p").astype(str).str.replace("p", " PM").str.replace("a", " AM")
    local = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + t, format="%Y-%m-%d %I:%M %p")
    return local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def record(win_pct: float, games: float) -> str:
    wins = round(win_pct * games)
    return f"{wins}-{int(games) - wins}"


def update_log(upcoming: pd.DataFrame, played_keys: set[str], now: str) -> pd.DataFrame:
    new = upcoming.assign(key=game_key(upcoming), predicted_at=now)[LOG_COLS]
    # Never touch a prediction once its game has a final score.
    new = new[~new["key"].isin(played_keys)]
    if LOG_FILE.exists():
        log = pd.read_csv(LOG_FILE, parse_dates=["date"])
        log = pd.concat([log[~log["key"].isin(new["key"])], new], ignore_index=True)
    else:
        log = new.copy()
    log["date"] = pd.to_datetime(log["date"])
    log = log.sort_values(["date", "key"])
    log.to_csv(LOG_FILE, index=False, float_format="%.4f")
    return log


def apply_injuries(upcoming: pd.DataFrame, today: pd.Timestamp) -> dict:
    """Replace the assumed "usual" lineups with ones from ESPN's injury report.

    Returns {(row index, side): [missing players]}. If ESPN can't be reached,
    the usual lineups stay and the picks just ignore injuries for the day.
    """
    try:
        rosters, report = injuries.fetch_rosters(), injuries.fetch_injuries()
    except Exception as e:  # noqa: BLE001
        print(f"Injury report unavailable ({e}); assuming usual lineups")
        return {}
    players = player_values(load_box())
    values, cutoffs = current_values(players), injuries.recent_cutoffs(players)
    missing = {}
    for i, r in upcoming.iterrows():
        for side in ("home", "away"):
            value, out = injuries.lineup(r[side], r["date"], today, rosters, report, values, cutoffs)
            upcoming.loc[i, f"{side}_lineup"] = value
            missing[(i, side)] = out
    upcoming["lineup_diff"] = upcoming["home_lineup"] - upcoming["away_lineup"]
    upcoming["lineup_change_diff"] = ((upcoming["home_lineup"] - upcoming["home_lineup_usual"])
                                      - (upcoming["away_lineup"] - upcoming["away_lineup_usual"]))
    print(f"Injury report: {len(report)} players listed")
    return missing


def backtest_summary() -> dict:
    m = pd.read_csv(METRICS_FILE)
    lr = m[m["model"] == "Logistic regression"]
    home = m[m["model"] == "Always pick home team"]
    acc = lambda d: float((d["accuracy"] * d["games"]).sum() / d["games"].sum())  # noqa: E731
    return {"games": int(lr["games"].sum()), "accuracy": acc(lr), "home_baseline": acc(home),
            "first_season": f"{lr['season'].min() - 1}-{str(lr['season'].min())[-2:]}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export website data")
    ap.add_argument("--today", type=date.fromisoformat, default=None)
    args = ap.parse_args()

    # NBA schedules are in US Eastern time; servers usually run on UTC.
    today = pd.Timestamp(args.today or pd.Timestamp.now(tz="America/New_York").date())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle = joblib.load(MODEL_FILE)
    df = build()
    df["key"] = game_key(df)

    # Upcoming games
    upcoming = df[~df["played"] & df["date"].between(today, today + timedelta(days=DAYS_AHEAD - 1))].copy()
    # Only predict games that haven't tipped off yet. Started games keep the
    # last prediction made before tip-off (already in the log).
    if args.today is None:
        upcoming = upcoming[tipoff_utc(upcoming) > pd.Timestamp.now(tz="UTC")]
    missing = apply_injuries(upcoming, today) if args.today is None else {}
    upcoming["p_home"] = bundle["model"].predict_proba(upcoming[bundle["features"]])[:, 1]
    played = df[df["played"]]
    log = update_log(upcoming, set(played["key"]), now)

    # Score logged predictions against final results
    scored = log.merge(played[["key", "home_pts", "away_pts"]], on="key")
    scored["home_won"] = scored["home_pts"] > scored["away_pts"]
    scored["correct"] = (scored["p_home"] >= 0.5) == scored["home_won"]
    recent = scored[scored["date"] >= today - timedelta(days=RECENT_DAYS)].sort_values("date", ascending=False)

    # Current Elo for each team = pre-game rating of its next game (or last game)
    tv = pd.concat([
        df[["date", "game_id", "played"]].assign(team=df["home"], elo=df["home_elo"]),
        df[["date", "game_id", "played"]].assign(team=df["away"], elo=df["away_elo"]),
    ]).sort_values(["date", "game_id"])
    nxt = tv[~tv["played"]].groupby("team")["elo"].first()
    last = tv.groupby("team")["elo"].last()
    elo = nxt.combine_first(last).to_dict()
    season = int(df["season"].max())
    cur = df[df["season"] == season]
    wins = pd.concat([
        cur[cur["played"]].assign(team=cur["home"], w=cur["home_win"]),
        cur[cur["played"]].assign(team=cur["away"], w=1 - cur["home_win"]),
    ]).groupby("team")["w"].agg(["sum", "count"])

    def team_info(i, r: pd.Series, side: str) -> dict:
        games = int(r[f"{side}_games"])
        n = min(10, games)
        wins = round(r[f"{side}_last10_win"] * n)
        return {"name": r[side], "abbr": ABBR.get(r[side], r[side][:3].upper()),
                "record": record(r[f"{side}_win_pct"], games),
                "last_n": n, "last_record": f"{wins}-{n - wins}" if n else None,
                "last_margin": round(float(r[f"{side}_last10_margin"]), 1),
                "avg_margin": round(float(r[f"{side}_avg_margin"]), 1),
                "rest": int(r[f"{side}_rest"]),
                "elo": round(r[f"{side}_elo"]), "b2b": bool(r[f"{side}_b2b"]),
                "lineup_change": round(float(r[f"{side}_lineup"] - r[f"{side}_lineup_usual"]), 1),
                "missing": missing.get((i, side), [])}

    data = {
        "generated_at": now,
        "model_trained_through": bundle["trained_through"],
        "backtest": backtest_summary(),
        "track_record": {"games": int(len(scored)), "correct": int(scored["correct"].sum()),
                         "since": str(log["date"].min().date()) if len(log) else None},
        "upcoming": [
            {"date": str(r["date"].date()), "time": r["start_et"] if isinstance(r["start_et"], str) else None,
             "neutral": bool(r["neutral"]), "playoffs": bool(r["playoffs"]),
             "notes": r["notes"] if isinstance(r["notes"], str) else None,
             "p_home": round(float(r["p_home"]), 4), "home": team_info(i, r, "home"), "away": team_info(i, r, "away"),
             "explain": explain(r, bundle["model"], bundle["features"],
                                {s: missing.get((i, s), []) for s in ("home", "away")})}
            for i, r in upcoming.iterrows()
        ],
        "recent_results": [
            {"date": str(r.date.date()), "away": r.away, "home": r.home,
             "away_abbr": ABBR.get(r.away), "home_abbr": ABBR.get(r.home),
             "away_pts": int(r.away_pts), "home_pts": int(r.home_pts),
             "p_home": round(float(r.p_home), 4), "correct": bool(r.correct)}
            for r in recent.itertuples()
        ],
        "rankings": [
            {"rank": i + 1, "name": t, "abbr": ABBR.get(t), "elo": round(e),
             "record": f"{int(wins.loc[t, 'sum'])}-{int(wins.loc[t, 'count'] - wins.loc[t, 'sum'])}" if t in wins.index else "0-0"}
            for i, (t, e) in enumerate(sorted(elo.items(), key=lambda kv: -kv[1]))
        ],
    }
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=1))
    tr = data["track_record"]
    print(f"{len(data['upcoming'])} upcoming games, track record {tr['correct']}/{tr['games']} -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
