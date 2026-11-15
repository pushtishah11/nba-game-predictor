"""Train and evaluate game-outcome models with walk-forward validation.

For each test season S, models are trained only on seasons before S and
scored on S - the same situation as predicting a season you haven't seen.

Usage:
    python src/train.py
Outputs:
    results/metrics.csv, results/metrics.md, results/*.png, models/model.joblib
"""

from __future__ import annotations

from pathlib import Path

import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from features import FEATURES, OUT_FILE as FEATURES_FILE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# numpy 2.x on macOS (Apple Accelerate) emits spurious "overflow in matmul"
# warnings during LogisticRegression fits; the results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_FILE = ROOT / "models" / "model.joblib"

FIRST_TRAIN_SEASON = 2016  # 2015 is only used to warm up the Elo ratings
FIRST_TEST_SEASON = 2018   # always at least two seasons of training data


def make_models() -> dict:
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.03,
                                                            l2_regularization=1.0, random_state=0),
    }


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": accuracy_score(y, p > 0.5),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p)}


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, preds = [], []
    for season in range(FIRST_TEST_SEASON, df["season"].max() + 1):
        train = df[(df["season"] >= FIRST_TRAIN_SEASON) & (df["season"] < season)]
        test = df[df["season"] == season]
        if test.empty:
            continue
        y = test["home_win"].to_numpy()
        probs = {
            "Always pick home team": np.full(len(test), train["home_win"].mean()),
            "Elo only": test["elo_prob"].to_numpy(),
        }
        for name, model in make_models().items():
            model.fit(train[FEATURES], train["home_win"])
            probs[name] = model.predict_proba(test[FEATURES])[:, 1]

        for name, p in probs.items():
            rows.append({"season": season, "model": name, "games": len(test), **score(y, p)})
        preds.append(test[["season", "home_win"]].assign(**{n: p for n, p in probs.items()}))
    return pd.DataFrame(rows), pd.concat(preds)


def summarize(metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    model_names = metrics["model"].unique()
    overall = pd.DataFrame([
        {"model": m, "games": len(preds), **score(preds["home_win"].to_numpy(), preds[m].to_numpy())}
        for m in model_names
    ])
    return overall.sort_values("log_loss")


def write_markdown(overall: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = ["| Model | Accuracy | Log loss | Brier |", "|---|---|---|---|"]
    for r in overall.itertuples():
        lines.append(f"| {r.model} | {r.accuracy:.1%} | {r.log_loss:.4f} | {r.brier:.4f} |")
    lines += ["", "Accuracy by test season:", ""]
    pivot = metrics.pivot(index="season", columns="model", values="accuracy")[overall["model"]]
    lines.append("| Season | " + " | ".join(pivot.columns) + " |")
    lines.append("|---" * (len(pivot.columns) + 1) + "|")
    for season, r in pivot.iterrows():
        label = f"{season - 1}-{str(season)[-2:]}"
        if metrics.loc[metrics["season"] == season, "games"].iloc[0] < 1000:
            label += " (in progress)"
        lines.append(f"| {label} | " + " | ".join(f"{v:.1%}" for v in r) + " |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n")


def plot_accuracy(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, grp in metrics.groupby("model", sort=False):
        ax.plot(grp["season"], grp["accuracy"] * 100, marker="o", label=name)
    ax.set_xlabel("Test season (year it ended)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Walk-forward accuracy by season")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_by_season.png", dpi=150)
    plt.close(fig)


def plot_calibration(preds: pd.DataFrame, name: str) -> None:
    frac, mean_p = calibration_curve(preds["home_win"], preds[name], n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_p, frac, marker="o", label=name)
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration.png", dpi=150)
    plt.close(fig)


def plot_coefficients(model) -> None:
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coefs.index, coefs.values, color=np.where(coefs > 0, "#2b8cbe", "#e34a33"))
    ax.set_xlabel("Standardized coefficient (positive = favors home team)")
    ax.set_title("What the logistic regression relies on")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_weights.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"], low_memory=False)
    df = df[df["home_win"].notna()]

    metrics, preds = walk_forward(df)
    overall = summarize(metrics, preds)
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    write_markdown(overall, metrics)
    plot_accuracy(metrics)

    print("Walk-forward results (all test seasons combined):")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Final model: logistic regression on every completed game. It matches
    # XGBoost closely, is better calibrated, and is easy to explain.
    final = make_models()["Logistic regression"]
    train = df[df["season"] >= FIRST_TRAIN_SEASON]
    final.fit(train[FEATURES], train["home_win"])
    plot_calibration(preds, "Logistic regression")
    plot_coefficients(final)

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES,
                 "trained_through": str(train["date"].max().date())}, MODEL_FILE)
    print(f"Saved final model to {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
