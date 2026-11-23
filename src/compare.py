"""How does the model stack up against the betting market?

Usage:
    python src/compare.py     # after train.py and odds.py

Uses the walk-forward (out-of-sample) predictions from train.py, so every
model probability here was made without seeing that season. Writes
results/market.md and docs/img/market.png.

Three questions:
  1. Head to head: accuracy, log loss and Brier score on the same games.
  2. Does the model know anything the market doesn't? Fit outcome ~
     market + model on earlier seasons and test on the next one. If the
     model's weight is ~0 and the score doesn't improve, the market
     already priced in everything the model sees.
  3. Could you bet on it? Bet 1 unit whenever the model's edge over the
     market's price clears a threshold, at the actual (median) moneyline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import OUT_FILE as FEATURES_FILE
from odds import load_odds
from train import FIRST_TEST_SEASON, score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"
ACCENT, GREY = "#e8590c", "#8a8a8a"
THRESHOLDS = [0.0, 0.03, 0.05, 0.10]


def logit(p: pd.Series) -> np.ndarray:
    p = np.clip(p.to_numpy(dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def decimal(ml: float) -> float:
    return 1 + (ml / 100 if ml > 0 else 100 / -ml)


def load() -> pd.DataFrame:
    f = pd.read_csv(FEATURES_FILE, usecols=["game_id", "season", "date", "home_win", "elo_prob"],
                    parse_dates=["date"])
    preds = pd.read_csv(RESULTS / "predictions.csv")
    df = f.merge(preds, on="game_id").merge(load_odds(), on="game_id")
    return df[df["p_market"].notna() & (df["season"] >= FIRST_TEST_SEASON)].reset_index(drop=True)


def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    y = df["home_win"].to_numpy()
    rows = [{"model": name, **score(y, df[col].to_numpy())} for name, col in [
        ("Betting market", "p_market"), ("My model", "p_lineups"),
        ("My model, no lineups", "p_no_lineups"), ("Elo only", "elo_prob")]]
    return pd.DataFrame(rows)


def blend(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Walk-forward: does adding the model to the market's probability help?"""
    out, weights = [], []
    for season in sorted(df["season"].unique())[1:]:
        train, test = df[df["season"] < season], df[df["season"] == season]
        X = lambda d: np.column_stack([logit(d["p_market"]), logit(d["p_lineups"])])  # noqa: E731
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X(train), train["home_win"])
        weights.append(lr.coef_[0][1] / lr.coef_[0].sum())
        out.append(test.assign(p_blend=lr.predict_proba(X(test))[:, 1]))
    res = pd.concat(out)
    y = res["home_win"].to_numpy()
    table = pd.DataFrame([{"model": "Market", **score(y, res["p_market"].to_numpy())},
                          {"model": "Market + my model", **score(y, res["p_blend"].to_numpy())}])
    return table, float(np.mean(weights))


def betting(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["home_ml"].notna() & df["away_ml"].notna()]
    rows = []
    for t in THRESHOLDS:
        profit, bets, wins = 0.0, 0, 0
        for r in d.itertuples():
            for p, ml, won in ((r.p_lineups, r.home_ml, r.home_win == 1), (1 - r.p_lineups, r.away_ml, r.home_win == 0)):
                dec = decimal(ml)
                if p * dec - 1 > t:  # expected profit per unit, by the model's own estimate
                    bets += 1
                    wins += won
                    profit += dec - 1 if won else -1
        rows.append({"edge_needed": t, "bets": bets, "win_rate": wins / bets if bets else np.nan,
                     "profit_units": profit, "roi": profit / bets if bets else np.nan})
    return pd.DataFrame(rows)


def disagreements(df: pd.DataFrame) -> dict:
    model_home, market_home = df["p_lineups"] >= 0.5, df["p_market"] >= 0.5
    dis = df[model_home != market_home]
    return {"games": len(df), "disagree": len(dis),
            "model_right": float(((dis["p_lineups"] >= 0.5) == (dis["home_win"] == 1)).mean())}


def chart(df: pd.DataFrame) -> None:
    by = df.groupby("season").apply(lambda g: pd.Series({
        "Betting market": score(g["home_win"].to_numpy(), g["p_market"].to_numpy())["accuracy"],
        "My model": score(g["home_win"].to_numpy(), g["p_lineups"].to_numpy())["accuracy"]}),
        include_groups=False) * 100
    counts = df.groupby("season").size()
    labels = [f"{s - 1}-{str(s)[-2:]}" + (" (so far)" if counts[s] < 1000 else "") for s in by.index]
    ahead = int((by["Betting market"] > by["My model"]).sum())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(labels, by["Betting market"], "o-", color="#333", label="Betting market")
    ax.plot(labels, by["My model"], "o-", color=ACCENT, label="My model")
    ax.set_ylabel("Accuracy %")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"The market is ahead in {ahead} of {len(by)} seasons", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "market.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load()
    h2h = head_to_head(df)
    bl, w = blend(df)
    bets = betting(df)
    dis = disagreements(df)
    chart(df)

    fmt = lambda t: t.to_string(index=False, float_format=lambda v: f"{v:.4f}")  # noqa: E731
    print(f"{len(df)} games with a market line, {df['season'].min()}-{df['season'].max()}\n")
    print(fmt(h2h), "\n")
    print(f"Disagreements: {dis['disagree']} games ({dis['disagree'] / dis['games']:.1%}), "
          f"model right in {dis['model_right']:.1%} of them\n")
    print(fmt(bl), f"\nAverage share of weight on my model: {w:.1%}\n")
    print(fmt(bets))

    lines = [f"Walk-forward comparison with the betting market on {len(df):,} games "
             f"({df['season'].min() - 1}-{str(df['season'].min())[-2:]} to now). "
             "Market probability = median no-vig probability across sportsbooks' final pre-game moneylines.",
             "", "| | Accuracy | Log loss | Brier |", "|---|---|---|---|"]
    lines += [f"| {r.model} | {r.accuracy:.1%} | {r.log_loss:.4f} | {r.brier:.4f} |" for r in h2h.itertuples()]
    lines += ["", f"The model and the market pick different winners in {dis['disagree']:,} games "
              f"({dis['disagree'] / dis['games']:.1%}); the model is right in {dis['model_right']:.1%} of those.",
              "", "**Market + model blend** (fit on earlier seasons, tested on the next):", "",
              "| | Accuracy | Log loss | Brier |", "|---|---|---|---|"]
    lines += [f"| {r.model} | {r.accuracy:.1%} | {r.log_loss:.4f} | {r.brier:.4f} |" for r in bl.itertuples()]
    lines += ["", f"Average weight on the model: {w:.0%}.", "",
              "**Betting simulation** (1 unit per bet at the median moneyline):", "",
              "| Edge needed | Bets | Win rate | Profit (units) | ROI |", "|---|---|---|---|---|"]
    lines += [f"| {r.edge_needed:.0%} | {r.bets:,} | {r.win_rate:.1%} | {r.profit_units:+.1f} | {r.roi:+.1%} |"
              for r in bets.itertuples()]
    (RESULTS / "market.md").write_text("\n".join(lines) + "\n")
    m, me = h2h.set_index("model").loc["Betting market"], h2h.set_index("model").loc["My model"]
    (RESULTS / "market_summary.json").write_text(json.dumps({
        "games": len(df), "market_accuracy": round(m["accuracy"], 4), "model_accuracy": round(me["accuracy"], 4),
        "disagree": dis["disagree"], "model_right_when_disagree": round(dis["model_right"], 4),
        "bet_roi": round(float(bets.loc[bets["edge_needed"] == 0.05, "roi"].iloc[0]), 4)}, indent=1))
    print(f"\nWrote {RESULTS.relative_to(ROOT)}/market.md and docs/img/market.png")


if __name__ == "__main__":
    main()
