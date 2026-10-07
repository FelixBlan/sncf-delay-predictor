"""
ETL + feature engineering for the SNCF open dataset
"Regularite mensuelle des TGV" (AQST) -- https://data.sncf.com

Shape of the data
-----------------
One row = one origin-destination (OD) pair, aggregated over one calendar
month: ~130 OD pairs x ~100 months. There is no per-train or per-hour
information, so the modelling problem is a *panel forecast*: given
everything known up to month M-1 (plus the timetable and the calendar of
month M, and a weather scenario for it), predict the average arrival delay
of month M. Weather and calendar come from outside the export, see
`context.py`.

Design notes
------------
* Every predictive feature is built with a `shift(>=1)` inside the OD group,
  so a row never sees its own month. `OUTCOME_COLS` lists the columns that
  are only knowable after the month happened; they may enter the model as
  lags but never as raw values. `tests/test_pipeline.py` enforces this.
* The panel is made dense (every OD x every month) before shifting, so
  `shift(k)` really means "k months ago" even when a month is missing from
  the export (e.g. April 2020, where traffic went to zero).
* The same functions are used at training and at serving time (see
  `api.py`), which is what keeps train/serve skew out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from context import CONTEXT_COLUMNS, Context, add_context

RAW_SEP = ";"
TARGET = "retard_moyen_tous_trains_arrivee"
OD_KEY = ["gare_depart", "gare_arrivee"]
OD = "od"
PERIOD = "period"

# --- raw columns we keep (the three free-text "commentaire*" columns are dropped)
COUNT_COLS = [
    "nb_train_prevu", "nb_annulation", "nb_train_depart_retard",
    "nb_train_retard_arrivee", "nb_train_retard_sup_15",
    "nb_train_retard_sup_30", "nb_train_retard_sup_60",
]
MEAN_COLS = [
    "duree_moyenne", "retard_moyen_depart", "retard_moyen_tous_trains_depart",
    "retard_moyen_arrivee", TARGET, "retard_moyen_trains_retard_sup15",
    "prct_cause_externe", "prct_cause_infra", "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant", "prct_cause_gestion_gare",
    "prct_cause_prise_en_charge_voyageurs",
]

# Published by the timetable, therefore known *before* the month starts.
KNOWN_IN_ADVANCE = ["duree_moyenne", "nb_train_prevu"]

# Only observable *after* the month happened -> lagged features only.
OUTCOME_COLS = [c for c in COUNT_COLS + MEAN_COLS if c not in KNOWN_IN_ADVANCE]

# Per-train rates derived from the monthly counts (comparable across ODs of
# very different size, unlike the raw counts).
RATE_COLS = ["cancel_rate", "late_arrival_rate", "late_15_rate", "late_30_rate"]

# OD "reliability profile" of the recent past: how often trains were late,
# how late they were when they were, and what caused it.
PROFILE_COLS = RATE_COLS + [
    "retard_moyen_arrivee", "retard_moyen_tous_trains_depart",
    "prct_cause_infra", "prct_cause_externe",
    "prct_cause_gestion_trafic", "prct_cause_materiel_roulant",
    "prct_cause_gestion_gare", "prct_cause_prise_en_charge_voyageurs",
]

FEATURE_COLUMNS = [
    # --- the OD's own delay history
    "lag1", "lag2", "lag3", "lag12", "roll3", "roll6", "roll12",
    # --- network-wide state (captures shocks: strikes, storms, timetable changes)
    "net_lag1", "net_roll3", "net_anomaly", "net_lag12", "net_cancel_lag1",
    "dep_lag1", "arr_lag1",
    # --- known in advance for the month being predicted
    "duree_moyenne", "nb_train_prevu", "traffic_ratio", "month",
    # --- recent reliability profile (last month + 6-month mean)
    *[f"{c}_lag1" for c in PROFILE_COLS],
    *[f"{c}_roll6" for c in PROFILE_COLS],
    # --- weather scenario + calendar of the month being predicted (context.py)
    *CONTEXT_COLUMNS,
]

# The model predicts the *deviation* from this anchor rather than the delay
# itself: it cannot be fooled by the slow drift of the overall delay level,
# and it degrades gracefully to a sane baseline when a feature is missing.
ANCHOR = "anchor"

# Trees split on it as "missing"; -1 is out of range for every feature here
# (delays and rates are non-negative).
FILL_VALUE = -1.0


# --------------------------------------------------------------------------- #
# load / clean
# --------------------------------------------------------------------------- #
def load_raw(path: str) -> pd.DataFrame:
    """Read the AQST export and normalise it into a unique OD x month panel."""
    # utf-8-sig: the portal's export starts with a BOM, a saved copy may not
    df = pd.read_csv(path, sep=RAW_SEP, encoding="utf-8-sig")
    missing = {"date", *OD_KEY, TARGET} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} does not look like the AQST regularity export "
            f"(missing columns: {sorted(missing)})"
        )

    df[PERIOD] = pd.PeriodIndex(df["date"], freq="M")
    for col in COUNT_COLS + MEAN_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in OD_KEY:
        df[col] = df[col].astype(str).str.strip()

    df = _merge_services(df)
    df[OD] = df["gare_depart"] + " > " + df["gare_arrivee"]
    return df.sort_values([OD, PERIOD]).reset_index(drop=True)


def _merge_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since 2025-07 some ODs are split into a `National` and an `International`
    row for the same month. Summing the counts and taking a traffic-weighted
    mean of the averages keeps one continuous monthly series per OD, which is
    what the lag features need.
    """
    keys = [PERIOD, *OD_KEY]
    duplicated = df.duplicated(keys, keep=False)
    if not duplicated.any():
        return df

    rows = []
    for key, group in df[duplicated].groupby(keys, sort=False):
        weights = group["nb_train_prevu"].clip(lower=0).fillna(0)
        row = dict(zip(keys, key, strict=True))
        for col in COUNT_COLS:
            row[col] = group[col].sum(min_count=1)
        for col in MEAN_COLS:
            values = group[col]
            usable = weights.where(values.notna(), 0)
            row[col] = (
                np.average(values.fillna(0), weights=usable)
                if usable.sum() > 0 else values.mean()
            )
        if "service" in group.columns:
            row["service"] = group["service"].iloc[int(np.argmax(weights.to_numpy()))]
        rows.append(row)

    merged = pd.concat([df[~duplicated], pd.DataFrame(rows)], ignore_index=True)
    merged["date"] = merged[PERIOD].astype(str)
    return merged


def clean(
    df: pd.DataFrame,
    min_trains: int = 10,
    max_delay: float = 120.0,
) -> pd.DataFrame:
    """
    Drop the rows that cannot carry signal:

    * months where the OD ran no train at all (73 rows, mostly April 2020) --
      `duree_moyenne` is 0 there as well;
    * months with fewer than `min_trains` planned trains: the monthly average
      over a handful of trains is mostly noise (std ~80 min below 10 trains,
      ~5 min above 100);
    * corrupt aggregates: the export contains a handful of impossible
      negative averages (down to -472 min, around the Dec-2019 strike).
      Rounding noise just below zero is clipped instead of dropped;
    * averages above `max_delay`, which are month-long incidents rather than
      something a monthly model can be expected to predict.
    """
    df = df.copy()
    df = df[df["nb_train_prevu"].fillna(0) >= min_trains]
    df = df[df["duree_moyenne"].fillna(0) > 0]
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > -1.0]
    df[TARGET] = df[TARGET].clip(lower=0.0)
    df = df[df[TARGET] <= max_delay]
    return df.reset_index(drop=True)


def to_panel(df: pd.DataFrame, until: pd.Period | None = None) -> pd.DataFrame:
    """
    Make the panel dense: one row per (OD, month) over the full observed
    range, so that `shift(k)` means exactly "k months earlier".

    `until` extends the panel into the future, which is how a prediction row
    is created at serving time (outcome columns left as NaN).
    """
    if df.empty:
        return df.copy()
    last = df[PERIOD].max()
    if until is not None:
        last = max(last, until)
    months = pd.period_range(df[PERIOD].min(), last, freq="M")

    index = pd.MultiIndex.from_product(
        [df[OD].unique(), months], names=[OD, PERIOD]
    )
    dense = df.set_index([OD, PERIOD]).reindex(index).reset_index()
    # OD-level attributes do not vary by month
    for col in OD_KEY:
        dense[col] = dense.groupby(OD)[col].transform(lambda s: s.ffill().bfill())
    dense["date"] = dense[PERIOD].astype(str)
    return dense.sort_values([OD, PERIOD]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
def _grouped_roll(series: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    """Rolling mean of an already-shifted series, never crossing OD borders."""
    return series.groupby(groups).transform(
        lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the causal feature set. Expects the dense panel from `to_panel`;
    rows whose target is NaN are prediction rows and get their features
    exactly like any other row.
    """
    df = df.sort_values([OD, PERIOD]).reset_index(drop=True).copy()
    g = df.groupby(OD)
    trains = df["nb_train_prevu"].replace(0, np.nan)

    # per-train rates (this month's outcome -- only ever used lagged, below)
    df["cancel_rate"] = df["nb_annulation"] / trains
    df["late_arrival_rate"] = df["nb_train_retard_arrivee"] / trains
    df["late_15_rate"] = df["nb_train_retard_sup_15"] / trains
    df["late_30_rate"] = df["nb_train_retard_sup_30"] / trains

    # the OD's own delay history
    for lag in (1, 2, 3, 12):
        df[f"lag{lag}"] = g[TARGET].shift(lag)
    shifted = g[TARGET].shift(1)
    for window in (3, 6, 12):
        df[f"roll{window}"] = _grouped_roll(shifted, df[OD], window)

    # network-wide state, on a gap-free monthly index
    months = pd.period_range(df[PERIOD].min(), df[PERIOD].max(), freq="M")
    net = df.groupby(PERIOD, observed=True)[TARGET].mean().reindex(months)
    df["net_lag1"] = df[PERIOD].map(net.shift(1))
    df["net_roll3"] = df[PERIOD].map(net.shift(1).rolling(3, min_periods=2).mean())
    net_roll12 = df[PERIOD].map(net.shift(1).rolling(12, min_periods=6).mean())
    df["net_anomaly"] = df["net_roll3"] - net_roll12
    df["net_lag12"] = df[PERIOD].map(net.shift(12))
    # network-wide cancellation rate last month: strikes show up here first
    cancel = (
        df.groupby(PERIOD, observed=True)[["nb_annulation", "nb_train_prevu"]].sum(min_count=1)
        .reindex(months)
    )
    net_cancel = cancel["nb_annulation"] / cancel["nb_train_prevu"].replace(0, np.nan)
    df["net_cancel_lag1"] = df[PERIOD].map(net_cancel.shift(1))

    # station state last month (a station's delays spill over all its ODs),
    # on a gap-free index per station so that shift(1) is one month
    for col, name in (("gare_depart", "dep_lag1"), ("gare_arrivee", "arr_lag1")):
        station = df.groupby([col, PERIOD], observed=True)[TARGET].mean()
        station = station.unstack(col).reindex(months).shift(1).stack(future_stack=True)
        df[name] = pd.MultiIndex.from_arrays([df[PERIOD], df[col]]).map(station)

    # recent reliability profile
    for col in PROFILE_COLS:
        lag1 = g[col].shift(1)
        df[f"{col}_lag1"] = lag1
        df[f"{col}_roll6"] = _grouped_roll(lag1, df[OD], 6)

    # timetable / calendar
    df["traffic_ratio"] = df["nb_train_prevu"] / g["nb_train_prevu"].shift(1)
    df["month"] = df[PERIOD].dt.month

    # the baseline the model corrects: the OD's recent level, else last month
    df[ANCHOR] = df["roll6"].fillna(df["lag1"])
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return `(X, y, anchor)` with the columns in the exact training order."""
    X = df.reindex(columns=FEATURE_COLUMNS).astype(float).fillna(FILL_VALUE)
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, df[ANCHOR]


def predict_delays(model, df: pd.DataFrame) -> np.ndarray:
    """
    Anchor + predicted deviation, floored at zero. Shared by training,
    evaluation and the API, so a prediction is computed the same way
    everywhere.
    """
    X, _, anchor = build_feature_matrix(df)
    return np.clip(anchor.to_numpy() + model.predict(X), 0.0, None)


def build_features(
    panel: pd.DataFrame,
    context: Context | None = None,
    weather_mode: str = "observed",
    climate_until: pd.Period | None = None,
) -> pd.DataFrame:
    """
    Dense panel -> full feature set: the causal history features plus the
    month's weather scenario and calendar. The one entry point used by
    training, evaluation and the API.
    """
    return add_context(engineer_features(panel), context, weather_mode, climate_until)


def run_pipeline(
    raw_csv_path: str,
    min_trains: int = 10,
    context: Context | None = None,
    weather_mode: str = "observed",
):
    """
    raw CSV -> `(features, engineered_panel, clean_panel)`.

    `features` holds only the rows usable for supervised learning: a known
    target and at least one month of history behind them.
    """
    clean_panel = clean(load_raw(raw_csv_path), min_trains=min_trains)
    engineered = build_features(to_panel(clean_panel), context, weather_mode)
    features = engineered[engineered[TARGET].notna() & engineered[ANCHOR].notna()]
    return features.reset_index(drop=True), engineered, clean_panel
