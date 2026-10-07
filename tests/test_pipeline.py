"""Unit tests of the ETL and feature engineering, on the synthetic export."""

import numpy as np
import pandas as pd
import pytest

import data_pipeline as dp
from generate_data import generate, write


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "raw.csv"
    write(str(path), n_months=72, seed=7)
    return str(path)


@pytest.fixture(scope="module")
def panel(raw_csv):
    return dp.clean(dp.load_raw(raw_csv))


@pytest.fixture(scope="module")
def engineered(panel):
    return dp.engineer_features(dp.to_panel(panel))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_load_raw_parses_the_real_schema(raw_csv):
    df = dp.load_raw(raw_csv)
    assert {"date", dp.PERIOD, dp.OD, dp.TARGET, *dp.OD_KEY} <= set(df.columns)
    assert df[dp.PERIOD].dtype == "period[M]"
    # multi-line, semicolon-bearing comments must not shift the columns
    assert pd.api.types.is_numeric_dtype(df[dp.TARGET])


def test_load_raw_rejects_an_unrelated_csv(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AQST"):
        dp.load_raw(str(path))


def test_load_raw_merges_duplicate_service_rows(raw_csv):
    df = dp.load_raw(raw_csv)
    duplicated = df.duplicated([dp.PERIOD, *dp.OD_KEY]).sum()
    assert duplicated == 0

    # a National + International split of the same OD-month is summed/averaged
    row = df.iloc[0].copy()
    split = pd.DataFrame([row, row])
    split["service"] = ["National", "International"]
    split["nb_train_prevu"] = [100, 300]
    split[dp.TARGET] = [2.0, 10.0]
    merged = dp._merge_services(split)
    assert len(merged) == 1
    assert merged["nb_train_prevu"].iloc[0] == 400
    assert merged[dp.TARGET].iloc[0] == pytest.approx(8.0)  # traffic-weighted
    assert merged["service"].iloc[0] == "International"  # the dominant one


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def test_clean_drops_months_without_traffic(panel):
    assert (panel["nb_train_prevu"] >= 10).all()
    assert (panel["duree_moyenne"] > 0).all()


def test_clean_drops_corrupt_negative_averages(panel):
    assert (panel[dp.TARGET] >= 0).all()
    assert (panel[dp.TARGET] <= 120).all()


def test_clean_clips_rounding_noise_but_drops_real_negatives(panel):
    dirty = panel.head(3).copy()
    dirty.loc[dirty.index[0], dp.TARGET] = -0.4      # rounding noise -> clipped
    dirty.loc[dirty.index[1], dp.TARGET] = -120.0    # corrupt -> dropped
    cleaned = dp.clean(dirty)
    assert len(cleaned) == 2
    assert cleaned[dp.TARGET].min() == 0.0


def test_clean_keeps_min_trains_configurable(panel):
    small = panel.copy()
    small.loc[small.index[:5], "nb_train_prevu"] = 12
    assert len(dp.clean(small, min_trains=20)) < len(dp.clean(small, min_trains=5))


# --------------------------------------------------------------------------- #
# panel / features
# --------------------------------------------------------------------------- #
def test_to_panel_is_dense_so_shifts_mean_months(panel):
    dense = dp.to_panel(panel)
    months = dense[dp.PERIOD].nunique()
    assert len(dense) == dense[dp.OD].nunique() * months
    per_od = dense.groupby(dp.OD)[dp.PERIOD].apply(
        lambda s: (s.astype("int64").diff().dropna() == 1).all()
    )
    assert per_od.all(), "gaps left in the panel would corrupt lag features"


def test_to_panel_can_extend_into_the_future(panel):
    last = panel[dp.PERIOD].max()
    dense = dp.to_panel(panel, until=last + 1)
    future = dense[dense[dp.PERIOD] == last + 1]
    assert len(future) == dense[dp.OD].nunique()
    assert future[dp.TARGET].isna().all()
    assert future[dp.OD_KEY].notna().all().all()


def test_lag_features_are_the_previous_months_target(engineered):
    od = engineered[dp.OD].iloc[0]
    series = engineered[engineered[dp.OD] == od].sort_values(dp.PERIOD)
    assert series["lag1"].iloc[1:].to_numpy() == pytest.approx(
        series[dp.TARGET].iloc[:-1].to_numpy(), nan_ok=True
    )
    assert series["lag12"].iloc[12:].to_numpy() == pytest.approx(
        series[dp.TARGET].iloc[:-12].to_numpy(), nan_ok=True
    )


def test_rolling_features_never_cross_route_boundaries(engineered):
    first_rows = engineered.sort_values([dp.OD, dp.PERIOD]).groupby(dp.OD).head(1)
    # with nothing behind them, the first month of every route has no history
    assert first_rows[["lag1", "roll3", "roll6", "roll12"]].isna().all().all()


def test_no_raw_outcome_column_is_used_as_a_feature():
    """Outcome columns are only knowable after the month -> lags only."""
    assert not set(dp.FEATURE_COLUMNS) & set(dp.OUTCOME_COLS)
    assert set(dp.KNOWN_IN_ADVANCE) <= set(dp.FEATURE_COLUMNS)


def test_features_do_not_leak_the_current_month(engineered):
    """
    Corrupting a month's outcomes must not move that month's features:
    if it does, the model is reading its own answer.
    """
    target_period = engineered[dp.PERIOD].max()
    tampered = engineered.copy()
    mask = tampered[dp.PERIOD] == target_period
    for col in dp.OUTCOME_COLS:
        tampered.loc[mask, col] = 999.0

    before, _, anchor_before = dp.build_feature_matrix(engineered[mask])
    after, _, anchor_after = dp.build_feature_matrix(
        dp.engineer_features(tampered)[lambda d: d[dp.PERIOD] == target_period]
    )
    pd.testing.assert_frame_equal(before.reset_index(drop=True),
                                  after.reset_index(drop=True))
    assert anchor_before.to_numpy() == pytest.approx(anchor_after.to_numpy())


def test_build_feature_matrix_is_ordered_and_finite(engineered):
    X, y, anchor = dp.build_feature_matrix(engineered)
    assert list(X.columns) == dp.FEATURE_COLUMNS
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) == len(y) == len(anchor)


def test_run_pipeline_keeps_only_usable_rows(raw_csv):
    features, engineered, clean_panel = dp.run_pipeline(raw_csv)
    assert features[dp.TARGET].notna().all()
    assert features[dp.ANCHOR].notna().all()
    assert len(features) < len(engineered)
    assert len(clean_panel) <= len(engineered)


def test_predict_delays_is_anchor_plus_deviation(engineered):
    class Constant:
        def predict(self, X):
            return np.full(len(X), 2.0)

    rows = engineered[engineered[dp.ANCHOR].notna()].head(20)
    preds = dp.predict_delays(Constant(), rows)
    assert preds == pytest.approx(
        np.clip(rows[dp.ANCHOR].to_numpy() + 2.0, 0, None)
    )
    assert (preds >= 0).all()


# --------------------------------------------------------------------------- #
# the synthetic fixture must keep looking like the real export
# --------------------------------------------------------------------------- #
def test_generator_matches_the_real_export_schema():
    df = generate(n_months=24, seed=1)
    assert list(df.columns)[:4] == ["date", "service", "gare_depart", "gare_arrivee"]
    assert {*dp.COUNT_COLS, *dp.MEAN_COLS} <= set(df.columns)
    assert df["date"].str.match(r"^\d{4}-\d{2}$").all()


def test_generator_injects_the_quality_issues_cleaning_handles():
    df = generate(n_months=48, seed=3)
    assert (df[dp.TARGET] < -1).any(), "no corrupt aggregate to clean"
    assert (df["nb_train_prevu"] == 0).any(), "no zero-traffic month to clean"
