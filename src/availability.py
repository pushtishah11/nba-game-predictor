"""Player availability: how much talent each team actually has on the floor.

The game-level features only know a team's results, so they can't tell
when a star is hurt. This module rates every player from his recent box
scores and measures each team's available talent for a game.

  player value   average Game Score over his last 20 games, shrunk toward 0
                 when he has only played a few (computed before the game)
  lineup         sum of the 8 best available players' values
  lineup change  tonight's lineup minus the team's average over its last
                 10 games. A star sitting out makes this negative; a star
                 coming back makes it positive.

For past games "available" means the player actually played. That is a
slightly optimistic stand-in for the pre-game injury report (late
scratches are rare but they exist). For upcoming games export.py builds
the lineup from ESPN's current injury report instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOX_DIR = ROOT / "data" / "raw" / "box"

PLAYER_WINDOW = 20   # games used to rate a player
SHRINK = 2           # pseudo-games at value 0, so a 1-game sample counts for little
TOP_N = 8            # rotation size
TEAM_WINDOW = 10     # games used for a team's "usual" lineup


def load_box() -> pd.DataFrame:
    files = sorted(BOX_DIR.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError("No box scores found. Run: python src/boxscores.py")
    box = pd.concat([pd.read_csv(f, parse_dates=["date"], dtype={"espn_id": str, "player_id": str})
                     for f in files], ignore_index=True)
    return box.sort_values(["date", "espn_id"], kind="stable").reset_index(drop=True)


def player_values(box: pd.DataFrame) -> pd.DataFrame:
    """Played rows with each player's value before (pre_value) and after (value) the game."""
    p = box[~box["dnp"]].sort_values(["player_id", "date"]).copy()
    grp = p.groupby("player_id")["gmsc"]
    total = grp.transform(lambda s: s.rolling(PLAYER_WINDOW, min_periods=1).sum())
    n = grp.transform(lambda s: s.rolling(PLAYER_WINDOW, min_periods=1).count())
    p["value"] = total / (n + SHRINK)
    p["pre_value"] = p.groupby("player_id")["value"].shift(1).fillna(0.0)
    return p


def top_sum(values: pd.Series, n: int = TOP_N) -> float:
    return float(values.nlargest(n).sum())


def team_lineups(players: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played game: the lineup value that took the floor."""
    lu = (players.groupby(["date", "team"])["pre_value"].apply(top_sum)
          .rename("lineup").reset_index().sort_values(["team", "date"]))
    # The team's usual lineup going into each game (across seasons, so an
    # opener is compared with how the roster looked at the end of last season).
    lu["usual"] = lu.groupby("team")["lineup"].transform(
        lambda s: s.shift(1).rolling(TEAM_WINDOW, min_periods=1).mean())
    lu["usual"] = lu["usual"].fillna(lu["lineup"])
    return lu


def latest_usual(lineups: pd.DataFrame) -> dict[str, float]:
    """Each team's usual lineup value going into its next game."""
    return (lineups.groupby("team")["lineup"]
            .apply(lambda s: s.tail(TEAM_WINDOW).mean()).to_dict())


def current_values(players: pd.DataFrame) -> pd.DataFrame:
    """Latest value for every player, and the team he last played for."""
    last = players.sort_values("date").groupby("player_id").tail(1)
    return last.set_index("player_id")[["player", "team", "date", "value"]]


def add_availability(g: pd.DataFrame, box: pd.DataFrame | None = None) -> pd.DataFrame:
    box = load_box() if box is None else box
    players = player_values(box)
    lu = team_lineups(players)
    usual_next = latest_usual(lu)

    for side in ("home", "away"):
        m = g[["date", side]].merge(lu, left_on=["date", side], right_on=["date", "team"], how="left")
        lineup, usual = m["lineup"].to_numpy(), m["usual"].to_numpy()
        # Games without a box score (not played yet): assume the usual lineup.
        nxt = g[side].map(usual_next).to_numpy(dtype=float)
        lineup = np.where(np.isnan(lineup), nxt, lineup)
        usual = np.where(np.isnan(usual), nxt, usual)
        g[f"{side}_lineup"] = lineup
        g[f"{side}_lineup_usual"] = usual

    # Seasons before box score coverage (or a missing game) get neutral values.
    for side in ("home", "away"):
        g[f"{side}_lineup"] = g[f"{side}_lineup"].fillna(g[f"{side}_lineup_usual"])
    g["lineup_diff"] = (g["home_lineup"] - g["away_lineup"]).fillna(0.0)
    g["lineup_change_diff"] = ((g["home_lineup"] - g["home_lineup_usual"])
                               - (g["away_lineup"] - g["away_lineup_usual"])).fillna(0.0)
    return g
