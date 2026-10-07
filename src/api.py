"""
FastAPI service exposing the monthly delay forecaster.

A caller cannot reasonably supply 35 lagged features, so the service ships
with the cleaned history panel written by `train.py` and builds the feature
row itself, through the exact same pipeline functions used for training --
that is what rules out train/serve skew. The client only provides what it
actually knows: an OD pair, a month, optionally next month's timetable and
a weather scenario (see `context.py`).

The coming month is forecast by the model refitted on every month
(`model.joblib`); past months are answered by the evaluation model
(`model_eval.joblib`, fitted before the test period) under the `normal`
weather scenario, so that the history chart shows genuine forecasts.

Run locally:
    uvicorn api:app --app-dir src --reload
Then open http://127.0.0.1:8000 (web UI) or http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from context import CONTEXT_COLUMNS, Context
from data_pipeline import (
    ANCHOR,
    KNOWN_IN_ADVANCE,
    OD,
    OD_KEY,
    PERIOD,
    TARGET,
    build_features,
    load_raw,
    predict_delays,
    to_panel,
)

# What past months are scored with: no hindsight on the weather.
BACKTEST_WEATHER = "normal"

# Overridable so a deployment (or a test) can point at another artifact set.
MODEL_DIR = Path(
    os.environ.get("SNCF_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="SNCF Delay Predictor",
    description=(
        "Forecasts the average arrival delay (minutes, all trains included) "
        "of a TGV origin-destination pair for a given month, from SNCF's "
        "public monthly regularity data."
    ),
    version="1.1.0",
)


class Artifacts:
    """Models, feature contract, history panel and context, loaded once."""

    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "model.joblib")
        eval_path = model_dir / "model_eval.joblib"
        self.eval_model = joblib.load(eval_path) if eval_path.exists() else self.model
        self.context = Context.load(model_dir)
        with open(model_dir / "feature_columns.json", encoding="utf-8") as fh:
            self.contract = json.load(fh)
        try:
            self.metrics = json.loads(
                (model_dir / "metrics.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            self.metrics = {}
        self.history = load_raw(str(model_dir / "history.csv.gz"))
        self.last_month = self.history[PERIOD].max()
        self._backtest: pd.DataFrame | None = None

    @property
    def backtest(self) -> pd.DataFrame:
        """
        One-step-ahead prediction for every observed OD-month. Features are
        causal, so a single pass over the full history gives each month the
        exact number `/predict` returns when asked for it as a backtest.
        """
        if self._backtest is None:
            engineered = build_features(
                to_panel(self.history), self.context, BACKTEST_WEATHER
            )
            observed = engineered[
                engineered[TARGET].notna() & engineered[ANCHOR].notna()
            ].copy()
            observed["predicted"] = predict_delays(self.eval_model, observed)
            self._backtest = observed[[*OD_KEY, PERIOD, TARGET, "predicted"]]
        return self._backtest


@lru_cache(maxsize=1)
def artifacts() -> Artifacts:
    try:
        return Artifacts(MODEL_DIR)
    except FileNotFoundError as exc:  # pragma: no cover - deployment mistake
        raise RuntimeError(
            f"missing artifact in {MODEL_DIR}: {exc}. Run `python src/train.py` first."
        ) from exc


class PredictionRequest(BaseModel):
    gare_depart: str = Field(examples=["PARIS MONTPARNASSE"])
    gare_arrivee: str = Field(examples=["BORDEAUX ST JEAN"])
    month: str | None = Field(
        default=None,
        description="Month to forecast, YYYY-MM. Defaults to the month after "
                    "the last one in the data. A month already present in the "
                    "history is answered as a backtest.",
        examples=["2026-07"],
    )
    nb_train_prevu: int | None = Field(
        default=None, ge=1,
        description="Planned trains that month (from the timetable). "
                    "Defaults to the OD's recent average.",
    )
    duree_moyenne: float | None = Field(
        default=None, gt=0,
        description="Scheduled journey time in minutes. Defaults to the "
                    "OD's recent average.",
    )
    weather: Literal["forecast", "normal", "harsh", "observed"] | None = Field(
        default=None,
        description="Weather scenario for the month at both ends of the route. "
                    "`forecast`: days observed so far + 16-day forecast + normals "
                    "for the rest; `normal`: seasonal normals; `harsh`: 90th "
                    "percentile of the season; `observed`: what happened (past "
                    "months only). Defaults to `forecast` for the coming month "
                    "and `normal` for a backtest.",
    )


class PredictionResponse(BaseModel):
    gare_depart: str
    gare_arrivee: str
    month: str
    predicted_delay_minutes: float
    recent_average_minutes: float | None = Field(
        description="The OD's own 6-month average, i.e. the baseline the "
                    "model corrects."
    )
    is_backtest: bool = Field(
        description="True when the requested month is already in the data, so "
                    "the actual value is known and returned for comparison."
    )
    actual_delay_minutes: float | None = None
    history_up_to: str
    weather: str = Field(description="Weather scenario used.")
    model: Literal["final", "evaluation"] = Field(
        description="`final`: refitted on every month (coming month); "
                    "`evaluation`: fitted before the test period (backtests)."
    )
    context: dict[str, float | None] = Field(
        description="Weather at the two ends of the route (mean) and calendar "
                    "of the month, as fed to the model."
    )


class HistoryPoint(BaseModel):
    month: str
    actual_delay_minutes: float
    predicted_delay_minutes: float | None = Field(
        description="What the model forecast for that month with the data "
                    "available the month before."
    )


class RouteHistory(BaseModel):
    gare_depart: str
    gare_arrivee: str
    test_from: str | None = Field(
        description="First month held out from training: predictions from "
                    "there on are genuinely out of sample."
    )
    points: list[HistoryPoint]


class RouteInfo(BaseModel):
    gare_depart: str
    gare_arrivee: str
    months_observed: int
    last_month: str
    recent_average_minutes: float


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    art = artifacts()
    weather, outlook = art.context.weather, art.context.outlook
    return {
        "status": "ok",
        "model": art.metrics.get("model", "unknown"),
        "history_up_to": str(art.last_month),
        "n_routes": int(art.history[OD].nunique()),
        "weather_up_to": None if weather is None else str(weather[PERIOD].max()),
        "outlook_months": [] if outlook is None or weather is None else sorted(
            str(p) for p in outlook[PERIOD].unique() if p > weather[PERIOD].max()
        ),
        "school_calendar": art.context.school is not None,
    }


@app.get("/metrics")
def metrics() -> dict:
    """Evaluation report of the currently served model."""
    return artifacts().metrics


@app.get("/routes", response_model=list[RouteInfo])
def routes() -> list[RouteInfo]:
    """Origin-destination pairs the model can forecast."""
    history = artifacts().history.sort_values(PERIOD)
    out = []
    for (dep, arr), group in history.groupby(OD_KEY, sort=True):
        out.append(RouteInfo(
            gare_depart=dep,
            gare_arrivee=arr,
            months_observed=len(group),
            last_month=str(group[PERIOD].max()),
            recent_average_minutes=round(float(group[TARGET].tail(6).mean()), 2),
        ))
    return out


@app.get("/history", response_model=RouteHistory)
def history(
    gare_depart: str = Query(examples=["PARIS MONTPARNASSE"]),
    gare_arrivee: str = Query(examples=["BORDEAUX ST JEAN"]),
) -> RouteHistory:
    """Observed monthly delays of a route, next to the model's forecasts."""
    art = artifacts()
    dep, arr = _normalise(gare_depart), _normalise(gare_arrivee)
    route = _find_route(art.history, dep, arr)
    predicted = art.backtest.set_index([*OD_KEY, PERIOD])["predicted"]
    points = []
    for period, actual in route.set_index(PERIOD)[TARGET].sort_index().items():
        value = predicted.get((dep, arr, period))
        points.append(HistoryPoint(
            month=str(period),
            actual_delay_minutes=round(float(actual), 2),
            predicted_delay_minutes=None if value is None else round(float(value), 2),
        ))
    return RouteHistory(
        gare_depart=dep,
        gare_arrivee=arr,
        test_from=art.metrics.get("test", {}).get("from"),
        points=points,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    art = artifacts()
    history = art.history
    dep, arr = _normalise(req.gare_depart), _normalise(req.gare_arrivee)
    route = _find_route(history, dep, arr)

    target_period = _parse_month(req.month) if req.month else art.last_month + 1
    if target_period < route[PERIOD].min() + 1:
        raise HTTPException(
            status_code=422,
            detail=f"no history before {target_period}; this route starts at "
                   f"{route[PERIOD].min()}",
        )
    if target_period > art.last_month + 1:
        raise HTTPException(
            status_code=422,
            detail=f"the model forecasts one month ahead: with data up to "
                   f"{art.last_month}, the furthest month is "
                   f"{art.last_month + 1}",
        )

    # Same code path as training: dense panel -> causal features -> predict.
    # Rows after `target_period` are dropped so a backtest cannot see them.
    panel = to_panel(history[history[PERIOD] < target_period], until=target_period)
    row_mask = (panel[OD] == route[OD].iloc[0]) & (panel[PERIOD] == target_period)

    # The timetable is published before the month starts, so it is a feature,
    # not a leak: take it from the request, else from the record when the
    # month is already past, else from the OD's recent level.
    on_record = route[route[PERIOD] == target_period]
    overrides = {"nb_train_prevu": req.nb_train_prevu,
                 "duree_moyenne": req.duree_moyenne}
    for col in KNOWN_IN_ADVANCE:
        value = overrides[col]
        if value is None and not on_record.empty:
            value = on_record[col].iloc[0]
        if value is None:
            value = route[col].tail(6).mean()
        panel.loc[row_mask, col] = value

    is_past = target_period <= art.last_month
    weather = req.weather or (BACKTEST_WEATHER if is_past else "forecast")
    if weather == "observed" and not art.context.has_observed_weather(
        [dep, arr], target_period
    ):
        raise HTTPException(
            status_code=422,
            detail=f"no observed weather for {dep} / {arr} in {target_period}; "
                   "use the 'forecast', 'normal' or 'harsh' scenario",
        )

    features = build_features(panel, art.context, weather)
    row = features[
        (features[OD] == route[OD].iloc[0]) & (features[PERIOD] == target_period)
    ]
    model = art.eval_model if is_past else art.model
    prediction = float(predict_delays(model, row)[0])

    actual = route.loc[route[PERIOD] == target_period, TARGET]
    recent = row["roll6"].iloc[0]
    return PredictionResponse(
        gare_depart=dep,
        gare_arrivee=arr,
        month=str(target_period),
        predicted_delay_minutes=round(prediction, 1),
        recent_average_minutes=None if pd.isna(recent) else round(float(recent), 1),
        is_backtest=not actual.empty,
        actual_delay_minutes=None if actual.empty else round(float(actual.iloc[0]), 1),
        history_up_to=str(art.last_month),
        weather=weather,
        model="evaluation" if is_past else "final",
        context={
            col: None if pd.isna(v) else round(float(v), 1)
            for col, v in row[CONTEXT_COLUMNS].iloc[0].items()
        },
    )


def _normalise(station: str) -> str:
    return " ".join(station.split()).upper()


def _find_route(history: pd.DataFrame, dep: str, arr: str) -> pd.DataFrame:
    route = history[(history["gare_depart"] == dep) & (history["gare_arrivee"] == arr)]
    if route.empty:
        raise HTTPException(
            status_code=404,
            detail=f"unknown route {dep!r} -> {arr!r}; "
                   "see GET /routes for the available pairs",
        )
    return route


def _parse_month(month: str) -> pd.Period:
    try:
        return pd.Period(month, freq="M")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid month {month!r}, expected YYYY-MM"
        ) from exc
