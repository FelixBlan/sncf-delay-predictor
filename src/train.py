"""
Trains the monthly delay forecaster and writes everything the API needs to
serve it.

Because the data is a monthly panel, evaluation is chronological, never a
random split: the test set is the last `--test-months` months, and model
selection uses rolling-origin folds inside the training period only. Every
run also scores the naive baselines a forecaster has to beat -- an OD's
recent average is a strong predictor here, and a model that cannot improve
on it is not worth serving.

Weather is fitted on what was observed but scored on what a forecaster would
actually have: the climatology of the training years (`normal`). The score
with the observed weather is reported next to it, as is the score of the same
model without any weather or calendar input, so the contribution of the
context is measured rather than assumed.

Two models are written: `model_eval.joblib`, fitted before the test period
(it produces every backtest, so past forecasts stay out of sample), and
`model.joblib`, refitted on *all* the months once evaluation is done -- the
one that forecasts the coming month.

Usage:
    python src/fetch_data.py && python src/fetch_context.py   # once, network
    python src/train.py                       # real export + context in data/
    python src/train.py --data data/raw_regularity.csv --model-dir /tmp/models
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from context import CONTEXT_COLUMNS, Context, add_context
from data_pipeline import (
    ANCHOR,
    FEATURE_COLUMNS,
    OD,
    PERIOD,
    RAW_SEP,
    TARGET,
    build_feature_matrix,
    clean,
    engineer_features,
    load_raw,
    predict_delays,
    to_panel,
)

ROOT = Path(__file__).resolve().parent.parent

# The forest is deliberately kept shallow-ish: on this data a fully grown one
# (400 trees, depth 14, leaves of 4) scored 0.04 min better in walk-forward
# validation but weighed 13.5 MB against 1.9 MB, small enough to ship the
# trained model with the repository so the API runs straight from a clone.
MODELS = {
    "hgb": lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=42
    ),
    "rf": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=20, max_features=0.5,
        random_state=42, n_jobs=-1,
    ),
}


def _scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae_minutes": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_minutes": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "bias_minutes": round(float(np.mean(y_pred - y_true)), 3),
    }


def fit_model(kind: str, train: pd.DataFrame):
    """Fit on the deviation from the anchor rather than on the delay itself."""
    X, y, anchor = build_feature_matrix(train)
    model = MODELS[kind]()
    model.fit(X, y - anchor)
    return model


def without_context(df: pd.DataFrame) -> pd.DataFrame:
    """Blank out weather and calendar, for the ablation score."""
    return df.assign(**dict.fromkeys(CONTEXT_COLUMNS, np.nan))


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Forecasts available without any model, for reference."""
    fallback = test[ANCHOR]
    return {
        "train_mean": np.full(len(test), train[TARGET].mean()),
        "last_month": test["lag1"].fillna(fallback),
        "mean_6_months": fallback,
        "mean_12_months": test["roll12"].fillna(fallback),
    }


def rolling_origin_cv(
    features: pd.DataFrame, kinds: list[str], folds: int, horizon: int,
    scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Walk-forward validation: fit on everything before a cut-off month,
    score the `horizon` months that follow, then move the cut-off forward.
    `scored` (same rows, forecast-time weather) is what gets scored; it
    defaults to `features` itself.
    """
    scored = features if scored is None else scored
    months = np.sort(features[PERIOD].unique())
    rows = []
    for fold in range(folds, 0, -1):
        cut = months[-fold * horizon]
        train = features[features[PERIOD] < cut]
        test = scored[(scored[PERIOD] >= cut) & (scored[PERIOD] < cut + horizon)]
        if train.empty or test.empty:
            continue
        for name, preds in baselines(train, test).items():
            rows.append({"fold": str(cut), "model": f"baseline:{name}",
                         **_scores(test[TARGET], preds)})
        for kind in kinds:
            preds = predict_delays(fit_model(kind, train), test)
            rows.append({"fold": str(cut), "model": kind,
                         **_scores(test[TARGET], preds)})
    return pd.DataFrame(rows)


def train(
    data_path: str,
    model_dir: str = "models",
    model_kind: str = "auto",
    test_months: int = 12,
    cv_folds: int = 3,
    min_trains: int = 10,
    context_dir: str | Path | None = None,
) -> dict:
    context = Context.load(context_dir) if context_dir else Context()
    clean_panel = clean(load_raw(data_path), min_trains=min_trains)
    engineered = engineer_features(to_panel(clean_panel))
    usable = engineered[engineered[TARGET].notna() & engineered[ANCHOR].notna()]
    usable = usable.reset_index(drop=True)

    months = np.sort(usable[PERIOD].unique())
    if len(months) < test_months + 24:
        raise ValueError(
            f"need at least {test_months + 24} months of history, got {len(months)}"
        )
    split = months[-test_months]
    # fitted on the weather that happened, scored on the weather one could
    # have expected: the climatology of the training years only
    observed = add_context(usable, context, "observed")
    forecast = add_context(usable, context, "normal", climate_until=split)
    train_df = observed[observed[PERIOD] < split]
    test_df = forecast[forecast[PERIOD] >= split]
    test_observed = observed[observed[PERIOD] >= split]

    # --- model selection on the training period only
    candidates = list(MODELS) if model_kind == "auto" else [model_kind]
    cv = rolling_origin_cv(train_df, candidates, folds=cv_folds, horizon=6,
                           scored=forecast[forecast[PERIOD] < split])
    cv_mae = cv.groupby("model")["mae_minutes"].mean().sort_values()
    chosen = (
        cv_mae[[m in MODELS for m in cv_mae.index]].index[0]
        if model_kind == "auto" else model_kind
    )

    # --- evaluation model: everything before the test period
    t0 = time.time()
    eval_model = fit_model(chosen, train_df)
    fit_seconds = time.time() - t0
    test_scores = _scores(test_df[TARGET], predict_delays(eval_model, test_df))
    observed_scores = _scores(
        test_observed[TARGET], predict_delays(eval_model, test_observed)
    )
    ablation = fit_model(chosen, without_context(train_df))
    ablation_scores = _scores(
        test_df[TARGET], predict_delays(ablation, without_context(test_df))
    )
    baseline_scores = {
        name: _scores(test_df[TARGET], values)
        for name, values in baselines(train_df, test_df).items()
    }
    best_baseline = min(baseline_scores, key=lambda n: baseline_scores[n]["mae_minutes"])
    skill = 1 - test_scores["mae_minutes"] / baseline_scores[best_baseline]["mae_minutes"]

    # --- served model: refitted on every month, test period included
    model = fit_model(chosen, observed)

    metrics = {
        "model": chosen,
        "target": TARGET,
        "test": {"from": str(split), "to": str(months[-1]), "n_rows": len(test_df),
                 "weather": "normal (climatology of the training years)",
                 **test_scores},
        "test_observed_weather": observed_scores,
        "test_without_context": ablation_scores,
        "train": {"from": str(months[0]), "to": str(months[-test_months - 1]),
                  "n_rows": len(train_df), "fit_seconds": round(fit_seconds, 2)},
        "final_fit": {"from": str(months[0]), "to": str(months[-1]),
                      "n_rows": len(observed)},
        "baselines": baseline_scores,
        "skill_vs_best_baseline": {
            "baseline": best_baseline, "mae_reduction": round(float(skill), 3)
        },
        "cv_mean_mae": {k: round(float(v), 3) for k, v in cv_mae.items()},
        "n_features": len(FEATURE_COLUMNS),
        "n_od_pairs": int(usable[OD].nunique()),
        "context": {
            "weather_station_months": 0 if context.weather is None else len(context.weather),
            "school_calendar": context.school is not None,
        },
    }

    # --- artifacts: models + feature contract + the history and context the API needs
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    joblib.dump(eval_model, out / "model_eval.joblib", compress=3)
    context.save(out)
    with open(out / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"features": FEATURE_COLUMNS, "anchor": ANCHOR, "target": TARGET,
             "predicts": "deviation from anchor"},
            fh, indent=2,
        )
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    # Served by the API, in the same format as the raw export so that it goes
    # back through `load_raw` rather than through a second, divergent loader.
    clean_panel.drop(columns=[PERIOD, OD]).to_csv(
        out / "history.csv.gz", index=False, sep=RAW_SEP, compression="gzip"
    )

    _report(metrics, cv, eval_model, test_df)
    return metrics


def _report(metrics: dict, cv: pd.DataFrame, model, test_df: pd.DataFrame) -> None:
    print("=== Rolling-origin validation (mean MAE over folds, training period) ===")
    for name, mae in metrics["cv_mean_mae"].items():
        print(f"{name:>25}: {mae:.3f} min")

    t = metrics["test"]
    print(f"\n=== Held-out test: {t['from']} -> {t['to']} ({t['n_rows']} rows) ===")
    print(f"{'model (' + metrics['model'] + ')':>25}: "
          f"MAE {t['mae_minutes']:.3f}  RMSE {t['rmse_minutes']:.3f}  "
          f"R2 {t['r2']:+.3f}  bias {t['bias_minutes']:+.3f}")
    rows = {"with observed weather": metrics["test_observed_weather"],
            "no weather/calendar": metrics["test_without_context"],
            **{f"baseline: {k}": v for k, v in metrics["baselines"].items()}}
    for name, s in rows.items():
        print(f"{name:>25}: MAE {s['mae_minutes']:.3f}  "
              f"RMSE {s['rmse_minutes']:.3f}  R2 {s['r2']:+.3f}  "
              f"bias {s['bias_minutes']:+.3f}")
    skill = metrics["skill_vs_best_baseline"]
    print(f"-> {skill['mae_reduction']:+.1%} MAE vs best baseline "
          f"({skill['baseline']})")
    final = metrics["final_fit"]
    print(f"-> served model refitted on {final['from']} -> {final['to']} "
          f"({final['n_rows']} rows)")

    X, y, anchor = build_feature_matrix(test_df)
    with warnings.catch_warnings():
        # sklearn/joblib noise about config propagation, irrelevant here
        warnings.simplefilter("ignore", UserWarning)
        imp = permutation_importance(
            model, X, y - anchor, n_repeats=5, random_state=42,
            scoring="neg_mean_absolute_error",
        )
    print("\n=== Permutation importance on the test set (MAE increase, min) ===")
    for i in np.argsort(-imp.importances_mean)[:15]:
        print(f"{X.columns[i]:>32}: {imp.importances_mean[i]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",
                        default=str(ROOT / "data" / "regularite-mensuelle-tgv-aqst.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--model", default="auto", choices=["auto", *MODELS],
                        help="'auto' picks the best candidate by rolling-origin CV")
    parser.add_argument("--test-months", type=int, default=12,
                        help="number of final months held out for testing")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="number of 6-month walk-forward folds")
    parser.add_argument("--min-trains", type=int, default=10,
                        help="drop OD-months with fewer planned trains (noisy averages)")
    parser.add_argument("--context-dir", default=str(ROOT / "data"),
                        help="where fetch_context.py wrote weather and school holidays "
                             "('' to train without them)")
    args = parser.parse_args()
    train(args.data, args.model_dir, args.model, args.test_months,
          args.cv_folds, args.min_trains, args.context_dir or None)
