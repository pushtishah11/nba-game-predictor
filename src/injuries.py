"""Live rosters and injury report from ESPN, turned into tonight's lineup values.

Used by export.py for upcoming games. Past games use box scores instead
(see availability.py).
"""

from __future__ import annotations

import re

import pandas as pd

from availability import TOP_N
from boxscores import API, get_json, team_name

# Chance a player suits up, by injury status. ESPN's "Day-To-Day" covers
# everything from "probable" to "doubtful", so call it a coin flip.
PLAY_PROB = {"Out": 0.0, "Suspension": 0.0, "Day-To-Day": 0.5}


def fetch_rosters() -> dict[str, list[str]]:
    """Team name -> ESPN player ids on the current roster."""
    teams = get_json(f"{API}/teams")["sports"][0]["leagues"][0]["teams"]
    out = {}
    for t in teams:
        d = get_json(f"{API}/teams/{t['team']['id']}/roster")
        out[team_name(t["team"])] = [a["id"] for a in d["athletes"]]
    return out


def fetch_injuries() -> pd.DataFrame:
    d = get_json(f"{API}/injuries")
    rows = []
    for team in d["injuries"]:
        for i in team["injuries"]:
            href = i["athlete"]["links"][0]["href"] if i["athlete"].get("links") else ""
            pid = re.search(r"/id/(\d+)", href)
            details = i.get("details") or {}
            rows.append({"team": team_name(team), "player_id": pid.group(1) if pid else None,
                         "player": i["athlete"]["displayName"], "status": i["status"],
                         "injury": details.get("type"), "return_date": details.get("returnDate")})
    df = pd.DataFrame(rows, columns=["team", "player_id", "player", "status", "injury", "return_date"])
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")
    return df


def play_prob(inj: pd.Series | None, day: pd.Timestamp, today: pd.Timestamp) -> float:
    if inj is None:
        return 1.0
    if pd.notna(inj["return_date"]) and inj["return_date"] <= day:
        return 1.0  # expected back by then
    p = PLAY_PROB.get(inj["status"], 1.0)
    # A day-to-day tag says little about games several days away.
    if inj["status"] == "Day-To-Day" and day > today + pd.Timedelta(days=1):
        p = 1.0
    return p


def recent_cutoffs(players: pd.DataFrame, n: int = 3) -> dict[str, pd.Timestamp]:
    """Date of each team's n-th most recent game. Players who appeared since then are "recent"."""
    dates = players[["team", "date"]].drop_duplicates().sort_values("date")
    return dates.groupby("team")["date"].apply(lambda s: s.iloc[-n] if len(s) >= n else s.iloc[0]).to_dict()


def lineup(team: str, day: pd.Timestamp, today: pd.Timestamp, rosters: dict[str, list[str]],
           injuries: pd.DataFrame, values: pd.DataFrame,
           cutoffs: dict[str, pd.Timestamp] | None = None) -> tuple[float, list[dict]]:
    """Expected lineup value for one team on one day, and the notable players missing.

    Each missing player is flagged "new" if he played in one of the team's last
    3 games. Players out longer than that are already baked into the team's
    usual lineup and recent results, so they barely move the prediction.
    """
    inj = injuries.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    roster = rosters.get(team, [])
    vals = values["value"].reindex(roster).fillna(0.0)
    probs = pd.Series({pid: play_prob(inj.loc[pid] if pid in inj.index else None, day, today)
                       for pid in roster}, dtype=float)
    expected = (vals * probs).nlargest(TOP_N).sum()

    regulars = set(vals.nlargest(TOP_N).index)
    missing = []
    for pid in roster:
        if probs.get(pid, 1.0) < 1.0 and pid in regulars and vals[pid] > 0:
            r = inj.loc[pid]
            cutoff = (cutoffs or {}).get(team)
            new = bool(cutoff is not None and pid in values.index
                       and values.loc[pid, "team"] == team and values.loc[pid, "date"] >= cutoff)
            missing.append({"new": new, "player": r["player"], "status": "Questionable" if r["status"] == "Day-To-Day" else r["status"],
                            "injury": r["injury"] if isinstance(r["injury"], str) else None,
                            "value": round(float(vals[pid]), 1)})
    missing.sort(key=lambda m: -m["value"])
    return float(expected), missing
