"""
Renders the figures of the README from the shipped artifacts (models/).

Everything is recomputed from the model and the history panel, with the same
functions the API uses, so the figures cannot drift from what is served.

Usage:
    python src/make_figures.py            # -> docs/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from context import Context  # noqa: E402
from data_pipeline import (  # noqa: E402
    ANCHOR,
    OD,
    PERIOD,
    TARGET,
    build_features,
    load_raw,
    predict_delays,
    to_panel,
)

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID = "#14181f", "#5d6675", "#e2e5ea"
MODEL_COLOR, BASE_COLOR = "#d9480f", "#868e96"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def test_predictions(model_dir: Path) -> tuple[pd.DataFrame, dict]:
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    # the evaluation model, under the scenario the test was scored with
    eval_path = model_dir / "model_eval.joblib"
    model = joblib.load(eval_path if eval_path.exists() else model_dir / "model.joblib")
    history = load_raw(str(model_dir / "history.csv.gz"))
    test_from = pd.Period(metrics["test"]["from"], freq="M")
    engineered = build_features(
        to_panel(history), Context.load(model_dir), "normal", climate_until=test_from
    )
    test = engineered[
        engineered[TARGET].notna()
        & engineered[ANCHOR].notna()
        & (engineered[PERIOD] >= pd.Period(metrics["test"]["from"], freq="M"))
    ].copy()
    test["model"] = predict_delays(model, test)
    test["baseline"] = test["roll12"].fillna(test[ANCHOR])
    return test, metrics


def plot_mae_by_month(test: pd.DataFrame, out: Path) -> None:
    err = test.assign(
        model_err=(test["model"] - test[TARGET]).abs(),
        base_err=(test["baseline"] - test[TARGET]).abs(),
    ).groupby(PERIOD)[["model_err", "base_err"]].mean()

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(err))
    ax.bar(x - 0.2, err["base_err"], 0.4, color=BASE_COLOR, label="Route's 12-month average")
    ax.bar(x + 0.2, err["model_err"], 0.4, color=MODEL_COLOR, label="Model")
    ax.set_xticks(x, [str(p) for p in err.index], rotation=45, ha="right")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Error per held-out month, all routes", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_routes(test: pd.DataFrame, engineered_history: pd.DataFrame, out: Path) -> None:
    busiest = (
        engineered_history.groupby(OD)["nb_train_prevu"].sum().nlargest(4).index.tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharey=False)
    for ax, od in zip(axes.flat, busiest, strict=True):
        hist = engineered_history[engineered_history[OD] == od].tail(36)
        part = test[test[OD] == od]
        ax.plot(hist[PERIOD].dt.to_timestamp(), hist[TARGET], color=INK, lw=1.4,
                label="Observed")
        ax.plot(part[PERIOD].dt.to_timestamp(), part["model"], color=MODEL_COLOR,
                lw=1.6, ls="--", label="Model (one month ahead)")
        ax.axvspan(part[PERIOD].min().to_timestamp(), part[PERIOD].max().to_timestamp(),
                   color="#0b6bcb", alpha=0.06, lw=0)
        ax.set_title(od.replace(" > ", " → "), loc="left", fontsize=9, color=INK)
        ax.set_ylim(bottom=0)
        locator = mdates.MonthLocator(bymonth=(1, 7))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=8)
    axes.flat[0].set_ylabel("minutes")
    axes.flat[2].set_ylabel("minutes")
    axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Busiest routes - shaded: held-out test period", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(model_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    test, _ = test_predictions(model_dir)
    history = load_raw(str(model_dir / "history.csv.gz"))
    plot_mae_by_month(test, out_dir / "mae_by_month.png")
    plot_routes(test, history, out_dir / "routes.png")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs"))
    args = parser.parse_args()
    main(Path(args.model_dir), Path(args.out_dir))
