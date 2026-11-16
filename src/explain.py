"""Explain a single prediction from the logistic regression model.

The model's log-odds are a sum of (weight x feature) terms, so each
feature's push on the prediction can be measured exactly. We compare the
game to a reference matchup: two evenly matched, equally rested teams at
the same venue. That reference gives the "home court alone" starting
probability, and each group of features moves it from there.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features import ELO_HOME, elo_expected

GROUPS = [
    ("strength", "Team strength (Elo)", ["elo_diff", "elo_prob"]),
    ("season", "Season record", ["win_pct_diff", "margin_diff"]),
    ("recent", "Recent form (last 10)", ["last10_win_diff", "last10_margin_diff"]),
    ("rest", "Rest & schedule", ["home_rest", "away_rest", "home_b2b", "away_b2b", "home_games", "away_games"]),
]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def linear_parts(model, features: list[str]) -> tuple[pd.Series, float]:
    """Per-feature weights on raw (unscaled) values, plus the intercept."""
    scaler, lr = model[0], model[-1]
    w = lr.coef_[0] / scaler.scale_
    intercept = lr.intercept_[0] - float(np.dot(w, scaler.mean_))
    return pd.Series(w, index=features), intercept


def reference(row: pd.Series) -> pd.Series:
    """The same game, but with evenly matched, equally rested teams."""
    ref = row.copy()
    hca = 0 if row["neutral"] else ELO_HOME
    ref["elo_diff"] = hca
    ref["elo_prob"] = elo_expected(hca)
    for f in ("win_pct_diff", "margin_diff", "last10_win_diff", "last10_margin_diff"):
        ref[f] = 0.0
    ref["home_rest"] = ref["away_rest"] = 1.0
    ref["home_b2b"] = ref["away_b2b"] = 0
    ref["home_games"] = ref["away_games"] = row["home_games"]
    return ref


def _rec(win_pct: float, games: float) -> str:
    w = round(win_pct * games)
    return f"{w}\u2011{int(games) - w}"  # non-breaking hyphen keeps "4-9" on one line


def _short(name: str) -> str:
    """'Los Angeles Lakers' -> 'Lakers', 'Portland Trail Blazers' -> 'Trail Blazers'."""
    special = {"Portland Trail Blazers": "Trail Blazers"}
    return special.get(name, name.split()[-1])


def summary(row: pd.Series, deltas: dict[str, float], base: float) -> list[str]:
    """Plain-English reasons, biggest factors first."""
    h, a = _short(row["home"]), _short(row["away"])
    lines: list[tuple[float, str]] = []

    gap = row["home_elo"] - row["away_elo"]
    if abs(gap) >= 15:
        better, worse = (h, a) if gap > 0 else (a, h)
        degree = "much " if abs(gap) >= 100 else "slightly " if abs(gap) < 40 else ""
        lines.append((deltas["strength"], f"The {better} are rated {degree}stronger than the {worse} "
                      f"(Elo {round(max(row['home_elo'], row['away_elo']))} vs {round(min(row['home_elo'], row['away_elo']))})."))

    hg, ag = row["home_games"], row["away_games"]
    if hg >= 3 and ag >= 3 and abs(deltas["season"]) >= 0.01:
        teams = [(h, _rec(row["home_win_pct"], hg), row["home_avg_margin"]),
                 (a, _rec(row["away_win_pct"], ag), row["away_avg_margin"])]
        if deltas["season"] < 0:
            teams.reverse()
        (t1, r1, m1), (t2, r2, m2) = teams
        lines.append((deltas["season"], f"The {t1} have had the better season: "
                      f"{t1} {r1} ({m1:+.1f} pts/game), {t2} {r2} ({m2:+.1f})."))

    if hg >= 3 and ag >= 3 and abs(deltas["recent"]) >= 0.01:
        n_h, n_a = min(10, int(hg)), min(10, int(ag))
        wh, wa = round(row["home_last10_win"] * n_h), round(row["away_last10_win"] * n_a)
        teams = [(h, f"{wh}\u2011{n_h - wh}"), (a, f"{wa}\u2011{n_a - wa}")]
        if deltas["recent"] < 0:
            teams.reverse()
        (t1, r1), (t2, r2) = teams
        lines.append((deltas["recent"], f"The {t1} have been better lately: {t1} {r1}, "
                      f"{t2} {r2} in their last {min(n_h, n_a)}."))

    tired = [t for t, flag in ((h, row["home_b2b"]), (a, row["away_b2b"])) if flag]
    if len(tired) == 1:
        lines.append((deltas["rest"], f"The {tired[0]} are on the second night of a back-to-back."))
    elif len(tired) == 2:
        lines.append((deltas["rest"], "Both teams are on the second night of a back-to-back."))
    elif abs(row["home_rest"] - row["away_rest"]) >= 2:
        rested = h if row["home_rest"] > row["away_rest"] else a
        lines.append((deltas["rest"], f"The {rested} are better rested "
                      f"({int(max(row['home_rest'], row['away_rest']))} days off vs {int(min(row['home_rest'], row['away_rest']))})."))

    lines.sort(key=lambda x: -abs(x[0]))
    out = [text for _, text in lines[:3]]
    if row["neutral"]:
        out.append("It's at a neutral site, so neither team gets home-court advantage.")
    else:
        out.append(f"Home court is worth about {abs(base - 0.5) * 100:.0f} points of win probability "
                   f"to the {h} on its own.")
    if min(hg, ag) < 5:
        out.append("It's early in the season, so the model leans mostly on Elo, which carries over from last season.")
    return out


def explain(row: pd.Series, model, features: list[str]) -> dict:
    w, b = linear_parts(model, features)
    x = row[features].astype(float)
    ref = reference(row)[features].astype(float)

    z = b + float((w * ref).sum())
    base = sigmoid(z)
    steps, deltas = [], {}
    p_prev = base
    for key, label, cols in GROUPS:
        z += float((w[cols] * (x[cols] - ref[cols])).sum())
        p = sigmoid(z)
        deltas[key] = p - p_prev
        steps.append({"key": key, "label": label, "delta": round(p - p_prev, 4)})
        p_prev = p

    return {"base": round(base, 4), "steps": steps, "final": round(p_prev, 4),
            "summary": summary(row, deltas, base)}
