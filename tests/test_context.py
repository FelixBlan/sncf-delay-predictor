"""Weather and calendar context: aggregation, scenarios, calendar, joining."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import context as cx
import data_pipeline as dp
from generate_data import generate_context

M = pd.Period


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("year, expected", [
    (2019, date(2019, 4, 21)), (2024, date(2024, 3, 31)), (2026, date(2026, 4, 5)),
])
def test_easter(year, expected):
    assert cx.easter(year) == expected


def test_french_holidays_2026():
    hols = cx.french_holidays(2026)
    assert len(hols) == 11
    assert date(2026, 4, 6) in hols      # Easter Monday
    assert date(2026, 5, 14) in hols     # Ascension
    assert date(2026, 5, 25) in hols     # Whit Monday


def test_calendar_counts_holidays_bridges_and_weekends():
    cal = cx.calendar_features([M("2026-05", "M"), M("2025-11", "M")])
    may = cal.loc[M("2026-05", "M")]
    # Fri 1, Fri 8, Thu 14 (Ascension), Mon 25 (Whit Monday)
    assert may["cal_holidays"] == 4
    assert may["cal_long_weekends"] == 3
    assert may["cal_bridges"] == 1
    assert may["cal_weekend_days"] == 10
    # 1 Nov 2025 is a Saturday: only 11 Nov (Tuesday) counts, as a "pont"
    nov = cal.loc[M("2025-11", "M")]
    assert (nov["cal_holidays"], nov["cal_bridges"]) == (1, 1)
    assert np.isnan(nov["cal_school_any"])   # no school calendar given


def test_school_holidays_any_and_all_zones():
    school = pd.DataFrame({
        "zone": ["A", "B", "C", "A", "B", "C"],
        "start": pd.to_datetime(["2025-12-20"] * 3 + ["2026-02-07", "2026-02-14", "2026-02-21"]),
        "end": pd.to_datetime(["2026-01-05"] * 3 + ["2026-02-23", "2026-03-02", "2026-03-09"]),
    })
    cal = cx.calendar_features([M("2026-02", "M")], school).iloc[0]
    assert cal["cal_school_any"] == 22     # 7 Feb -> 28 Feb
    assert cal["cal_school_all"] == 2      # 21-22 Feb, all three zones off


def test_school_calendar_outside_its_range_is_missing_not_zero():
    school = pd.DataFrame({"zone": ["A"], "start": pd.to_datetime(["2026-02-07"]),
                           "end": pd.to_datetime(["2026-02-23"])})
    cal = cx.calendar_features([M("2030-01", "M")], school).iloc[0]
    assert np.isnan(cal["cal_school_any"])


# --------------------------------------------------------------------------- #
# weather aggregation
# --------------------------------------------------------------------------- #
def _daily(days, **values):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    base = {"tmax": 5.0, "tmin": 1.0, "precip": 0.0, "snow": 0.0, "wind": 2.0}
    base.update(values)
    return pd.DataFrame({"station": "X", "date": dates, **base})


def test_monthly_weather_counts_threshold_days():
    daily = _daily(31)
    daily.loc[:4, "tmin"] = -3.0           # 5 frost days
    daily.loc[10, "precip"] = 25.0         # 1 heavy-rain day
    daily.loc[11, "wind"] = 12.0           # 1 windy day
    row = cx.monthly_weather(daily).iloc[0]
    assert row["frost_days"] == 5
    assert row["heavy_rain_days"] == 1
    assert row["rain_mm"] == pytest.approx(25.0)
    assert row["windy_days"] == 1
    assert row["hot_days"] == 0


def test_monthly_weather_drops_incomplete_months():
    assert cx.monthly_weather(_daily(20)).empty


def test_monthly_outlook_keeps_partial_months_with_their_coverage():
    out = cx.monthly_outlook(_daily(20, tmin=-1.0)).iloc[0]
    assert out["known_days"] == 20
    assert out["frost_days"] == 20


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ctx():
    return generate_context(n_months=48, seed=5)


def test_normal_is_the_station_climatology(ctx):
    w = ctx.weather
    station = w[cx.STATION].iloc[0]
    expected = w[(w[cx.STATION] == station) & (w[cx.PERIOD].dt.month == 1)]["frost_days"].mean()
    out = cx.weather_scenario(w, pd.PeriodIndex([M("2030-01", "M")]), "normal")
    got = out[out[cx.STATION] == station]["frost_days"].iloc[0]
    assert got == pytest.approx(expected)


def test_harsh_is_at_least_normal(ctx):
    periods = pd.period_range("2030-01", "2030-12", freq="M")
    normal = cx.weather_scenario(ctx.weather, periods, "normal")
    harsh = cx.weather_scenario(ctx.weather, periods, "harsh")
    assert (harsh[cx.WEATHER_COLS].to_numpy() >= normal[cx.WEATHER_COLS].to_numpy() - 1e-9).all()


def test_observed_uses_what_happened_and_falls_back_to_normal(ctx):
    w = ctx.weather
    past, future = w[cx.PERIOD].max(), w[cx.PERIOD].max() + 1
    out = cx.weather_scenario(w, pd.PeriodIndex([past, future]), "observed")
    seen = w[w[cx.PERIOD] == past].set_index(cx.STATION)["rain_mm"]
    got = out[out[cx.PERIOD] == past].set_index(cx.STATION)["rain_mm"]
    pd.testing.assert_series_equal(got.sort_index(), seen.sort_index(), check_names=False)
    assert out[out[cx.PERIOD] == future]["rain_mm"].notna().all()


def test_normals_can_exclude_the_test_period(ctx):
    """Tampering with the weather after `climate_until` must not move the normals."""
    w = ctx.weather
    cut = w[cx.PERIOD].max() - 11
    tampered = w.copy()
    tampered.loc[tampered[cx.PERIOD] >= cut, cx.WEATHER_COLS] = 999.0
    periods = pd.PeriodIndex([cut])
    a = cx.weather_scenario(w, periods, "normal", climate_until=cut)
    b = cx.weather_scenario(tampered, periods, "normal", climate_until=cut)
    pd.testing.assert_frame_equal(a, b)


def test_forecast_blends_known_days_with_normals(ctx):
    w = ctx.weather
    month = w[cx.PERIOD].max() + 1
    station = w[cx.STATION].iloc[0]
    outlook = pd.DataFrame([{cx.STATION: station, cx.PERIOD: month, "frost_days": 10,
                             "hot_days": 0, "rain_mm": 40.0, "heavy_rain_days": 1,
                             "snow_days": 0, "windy_days": 0,
                             "known_days": month.days_in_month - 10}])
    periods = pd.PeriodIndex([month])
    normal = cx.weather_scenario(w, periods, "normal").set_index(cx.STATION)
    blend = cx.weather_scenario(w, periods, "forecast", outlook=outlook).set_index(cx.STATION)
    share = 10 / month.days_in_month
    assert blend.loc[station, "frost_days"] == pytest.approx(
        10 + normal.loc[station, "frost_days"] * share
    )
    # stations without an outlook keep their normals
    other = blend.index[blend.index != station][0]
    assert blend.loc[other, "rain_mm"] == pytest.approx(normal.loc[other, "rain_mm"])


def test_forecast_never_rewrites_a_month_already_on_record(ctx):
    w = ctx.weather
    past = w[cx.PERIOD].max()
    station = w[cx.STATION].iloc[0]
    outlook = pd.DataFrame([{cx.STATION: station, cx.PERIOD: past, **dict.fromkeys(
        cx.WEATHER_COLS, 999.0), "known_days": 5}])
    periods = pd.PeriodIndex([past])
    normal = cx.weather_scenario(w, periods, "normal")
    forecast = cx.weather_scenario(w, periods, "forecast", outlook=outlook)
    pd.testing.assert_frame_equal(normal, forecast)


def test_unknown_mode_is_rejected(ctx):
    with pytest.raises(ValueError, match="weather mode"):
        cx.weather_scenario(ctx.weather, pd.PeriodIndex([M("2030-01", "M")]), "sunny")


# --------------------------------------------------------------------------- #
# joining onto the panel, persistence
# --------------------------------------------------------------------------- #
def test_add_context_averages_both_ends_of_the_route(ctx):
    w = ctx.weather
    dep, arr = sorted(w[cx.STATION].unique())[:2]
    month = w[cx.PERIOD].max()
    panel = pd.DataFrame({"gare_depart": [dep], "gare_arrivee": [arr], "period": [month]})
    out = cx.add_context(panel, ctx, "observed").iloc[0]
    at = w[w[cx.PERIOD] == month].set_index(cx.STATION)["rain_mm"]
    assert out["wx_rain_mm"] == pytest.approx((at[dep] + at[arr]) / 2)


def test_add_context_without_context_leaves_missing_values():
    panel = pd.DataFrame({"gare_depart": ["A"], "gare_arrivee": ["B"],
                          "period": [M("2026-05", "M")]})
    out = cx.add_context(panel, None).iloc[0]
    assert np.isnan(out["wx_frost_days"])
    assert out["cal_holidays"] == 4      # the calendar needs no download


def test_context_round_trips_through_disk(ctx, tmp_path):
    ctx.save(tmp_path)
    back = cx.Context.load(tmp_path)
    pd.testing.assert_frame_equal(
        back.weather.reset_index(drop=True), ctx.weather.reset_index(drop=True),
        check_dtype=False, check_like=True,
    )
    assert len(back.school) == len(ctx.school)


def test_context_columns_are_model_features():
    assert set(cx.CONTEXT_COLUMNS) <= set(dp.FEATURE_COLUMNS)
    assert not set(cx.CONTEXT_COLUMNS) & set(dp.OUTCOME_COLS)
