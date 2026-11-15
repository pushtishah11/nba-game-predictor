"""Turn raw game results into pre-game features.

Every feature for a game is computed only from games played *before* it,
so the model never sees the result it is trying to predict.

Usage:
    python src/features.py      # reads data/games.csv, writes data/features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAMES_FILE = ROOT / "data" / "games.csv"
OUT_FILE = ROOT / "data" / "features.csv"

# Elo settings (in the spirit of FiveThirtyEight's NBA Elo)
ELO_START = 1500
ELO_MEAN = 1505
ELO_K = 20
ELO_HOME = 100       # home-court advantage in Elo points
ELO_CARRY = 0.75     # fraction of rating kept between seasons

ROLL = 10            # window for "recent form" features

FEATURES = [
    "elo_diff", "elo_prob",
    "win_pct_diff", "margin_diff",
    "last10_win_diff", "last10_margin_diff",
    "home_rest", "away_rest", "home_b2b", "away_b2b",
    "home_games", "away_games",
    "neutral", "playoffs",
]


def load_games(path: Path = GAMES_FILE) -> pd.DataFrame:
    g = pd.read_csv(path, parse_dates=["date"])
    g = g.sort_values(["date", "start_et"], kind="stable").reset_index(drop=True)
    g["game_id"] = g.index
    g["played"] = g["home_pts"].notna()
    # Neutral-site games (Paris, Mexico City, NBA Cup final in Las Vegas...)
    g["neutral"] = g["notes"].astype(str).str.match(r"^at ").astype(int)
    g["playoffs"] = g["playoffs"].astype(int)
    g["home_win"] = np.where(g["played"], (g["home_pts"] > g["away_pts"]).astype(float), np.nan)
    return g


def elo_expected(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def add_elo(g: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    home_pre, away_pre = [], []

    for row in g.itertuples(index=False):
        for team in (row.home, row.away):
            if team not in ratings:
                ratings[team] = ELO_START
            elif last_season[team] != row.season:
                # Regress toward the mean at the start of each season.
                ratings[team] = ELO_CARRY * ratings[team] + (1 - ELO_CARRY) * ELO_MEAN
            last_season[team] = row.season

        rh, ra = ratings[row.home], ratings[row.away]
        home_pre.append(rh)
        away_pre.append(ra)
        if not row.played:
            continue

        hca = 0 if row.neutral else ELO_HOME
        exp_home = elo_expected(rh + hca - ra)
        margin = row.home_pts - row.away_pts
        winner_diff = (rh + hca - ra) if margin > 0 else (ra - rh - hca)
        # Margin-of-victory multiplier, damped for heavy favourites.
        mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = ELO_K * mult * ((1.0 if margin > 0 else 0.0) - exp_home)
        ratings[row.home] = rh + delta
        ratings[row.away] = ra - delta

    g["home_elo"], g["away_elo"] = home_pre, away_pre
    hca = np.where(g["neutral"] == 1, 0, ELO_HOME)
    g["elo_diff"] = g["home_elo"] + hca - g["away_elo"]
    g["elo_prob"] = elo_expected(g["elo_diff"])
    return g


def team_view(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from that team's perspective."""
    cols = ["game_id", "season", "date", "played"]
    home = g[cols].assign(team=g["home"], side="home",
                          pts_for=g["home_pts"], pts_against=g["away_pts"])
    away = g[cols].assign(team=g["away"], side="away",
                          pts_for=g["away_pts"], pts_against=g["home_pts"])
    t = pd.concat([home, away]).sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    t["win"] = np.where(t["played"], (t["pts_for"] > t["pts_against"]).astype(float), np.nan)
    t["margin"] = t["pts_for"] - t["pts_against"]
    return t


def add_form(g: pd.DataFrame) -> pd.DataFrame:
    t = team_view(g)
    key = ["team", "season"]

    # Stats *after* each played game, computed within the season...
    p = t[t["played"]].copy()
    grp = p.groupby(key)
    p["post_games"] = grp.cumcount() + 1
    p["post_win_pct"] = grp["win"].cumsum() / p["post_games"]
    p["post_avg_margin"] = grp["margin"].cumsum() / p["post_games"]
    p["post_last10_win"] = grp["win"].transform(lambda s: s.rolling(ROLL, min_periods=1).mean())
    p["post_last10_margin"] = grp["margin"].transform(lambda s: s.rolling(ROLL, min_periods=1).mean())
    post_cols = [c for c in p.columns if c.startswith("post_")]
    t = t.join(p[post_cols])

    # ...then each game's *pre-game* value is the latest post-game value
    # from an earlier game. This also works for games not yet played.
    pre = t.groupby(key)[post_cols].shift(1)
    pre = pre.groupby([t["team"], t["season"]]).ffill()
    pre.columns = [c.replace("post_", "") for c in post_cols]
    neutral_start = {"games": 0, "win_pct": 0.5, "avg_margin": 0.0,
                     "last10_win": 0.5, "last10_margin": 0.0}
    t = t.join(pre.fillna(neutral_start))

    # Days of rest since the team's previous game (capped at a week).
    prev = t.groupby("team")["date"].shift(1)
    t["rest"] = (t["date"] - prev).dt.days.sub(1).clip(0, 7).fillna(7)
    t["b2b"] = (t["rest"] == 0).astype(int)

    stats = ["games", "win_pct", "avg_margin", "last10_win", "last10_margin", "rest", "b2b"]
    for side in ("home", "away"):
        s = t[t["side"] == side].set_index("game_id")[stats].add_prefix(f"{side}_")
        g = g.join(s, on="game_id")

    g["win_pct_diff"] = g["home_win_pct"] - g["away_win_pct"]
    g["margin_diff"] = g["home_avg_margin"] - g["away_avg_margin"]
    g["last10_win_diff"] = g["home_last10_win"] - g["away_last10_win"]
    g["last10_margin_diff"] = g["home_last10_margin"] - g["away_last10_margin"]
    return g


def build() -> pd.DataFrame:
    g = load_games()
    g = add_elo(g)
    g = add_form(g)
    return g


def main() -> None:
    g = build()
    g.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(g)} rows x {len(FEATURES)} features to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
