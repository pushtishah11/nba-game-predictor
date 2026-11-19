"""Charts and numbers for the write-up (docs/writeup.html).

Usage:
    python src/analysis.py    # after features.py and train.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from features import OUT_FILE as FEATURES_FILE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
METRICS = ROOT / "results" / "metrics.csv"
ACCENT, GREY = "#e8590c", "#8a8a8a"


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def home_court(df: pd.DataFrame) -> None:
    g = df[(df["neutral"] == 0) & (df["playoffs"] == 0)]
    by = g.groupby("season")["home_win"].agg(["mean", "size"])
    by = by[by["size"] > 600]  # skip the season in progress
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [GREY if s < 2020 else ACCENT for s in by.index]
    ax.bar([f"{s - 1}-{str(s)[-2:]}" for s in by.index], by["mean"] * 100, color=colors)
    ax.set_ylim(50, 62)
    ax.set_ylabel("Home win %")
    ax.tick_params(axis="x", rotation=45)
    style(ax, "Home teams win less often since 2020")
    fig.tight_layout()
    fig.savefig(IMG / "home_court.png", dpi=150)
    plt.close(fig)
    pre, post = g[g["season"] < 2020]["home_win"].mean(), g[g["season"] >= 2021]["home_win"].mean()
    print(f"Home win %: {pre:.1%} before 2019-20, {post:.1%} from 2020-21 on")


def rest(df: pd.DataFrame) -> None:
    g = df[(df["neutral"] == 0) & (df["playoffs"] == 0)]
    cases = {
        "Both rested": g[(g["home_b2b"] == 0) & (g["away_b2b"] == 0)],
        "Road team on\nback-to-back": g[(g["away_b2b"] == 1) & (g["home_b2b"] == 0)],
        "Home team on\nback-to-back": g[(g["home_b2b"] == 1) & (g["away_b2b"] == 0)],
    }
    rates = {k: v["home_win"].mean() * 100 for k, v in cases.items()}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(rates.keys(), rates.values(), color=[GREY, ACCENT, ACCENT])
    for i, v in enumerate(rates.values()):
        ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", fontsize=11)
    ax.set_ylim(40, 66)
    ax.set_ylabel("Home win %")
    style(ax, "A back-to-back costs about 5-6 points of win %")
    fig.tight_layout()
    fig.savefig(IMG / "rest.png", dpi=150)
    plt.close(fig)
    print("Home win % by rest:", {k.replace(chr(10), " "): f"{v:.1f}" for k, v in rates.items()},
          {k.replace(chr(10), " "): len(v) for k, v in cases.items()})


def injuries(df: pd.DataFrame) -> None:
    g = df[df["season"] >= 2016].copy()
    edges = [-np.inf, -15, -8, -3, 3, 8, 15, np.inf]
    labels = ["< -15", "-15 to -8", "-8 to -3", "±3", "3 to 8", "8 to 15", "> 15"]
    g["bucket"] = pd.cut(g["lineup_change_diff"], edges, labels=labels)
    by = g.groupby("bucket", observed=True).agg(win=("home_win", "mean"), n=("home_win", "size"),
                                                model=("p_no_lineups", "mean"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(by))
    ax.bar(x, by["win"] * 100, color=ACCENT, label="Home team actually won")
    ax.plot(x, by["model"] * 100, "o-", color="#333", label="Model without lineups expected")
    ax.set_xticks(x, by.index)
    ax.set_xlabel("Home lineup change minus away lineup change (Game Score points)")
    ax.set_ylabel("Home win %")
    ax.set_ylim(20, 85)
    ax.legend(frameon=False, fontsize=9, loc="upper left", reverse=True)
    style(ax, "When a team is missing players, results-only models miss it")
    fig.tight_layout()
    fig.savefig(IMG / "injuries.png", dpi=150)
    plt.close(fig)
    print("By lineup change:\n", by.round(3).to_string())


def accuracy() -> None:
    m = pd.read_csv(METRICS)
    pick = ["Logistic regression", "Logistic regression (no lineups)", "Elo only", "Always pick home team"]
    pv = m.pivot(index="season", columns="model", values="accuracy")[pick] * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, color, ls in zip(pick, [ACCENT, "#333", GREY, GREY], ["-", "-", "--", ":"]):
        ax.plot([f"{s - 1}-{str(s)[-2:]}" for s in pv.index], pv[name], ls, marker="o", ms=4, color=color, label=name)
    ax.set_ylabel("Accuracy %")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    style(ax, "Walk-forward accuracy by season")
    fig.tight_layout()
    fig.savefig(IMG / "accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"], low_memory=False)
    df = df[df["home_win"].notna()]
    preds = pd.read_csv(ROOT / "results" / "predictions.csv")
    df = df.merge(preds, on="game_id", how="left")
    home_court(df)
    rest(df)
    injuries(df[df["p_no_lineups"].notna()])
    accuracy()
    print(f"Charts saved to {IMG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
