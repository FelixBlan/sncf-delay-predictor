"""
Exogenous context of a month: weather along the route and the calendar.

The AQST export says nothing about *why* a month went badly beyond its own
cause shares. Two outside sources fill part of that gap:

* **Weather** (NASA POWER daily reanalysis, one point per station, aggregated
  per month by `fetch_context.py`): frost, heat, heavy rain, snow and wind
  days. Frozen points, buckled rails, flooded cuttings and trees on the
  catenary are classic delay causes.
* **Calendar**: French public holidays (computed), long weekends and
  "ponts", weekend days, and school holidays for zones A/B/C (published by
  the Ministry of Education years in advance). Holiday peaks load stations
  and trains.

Timing matters. The calendar of month M is known before M starts, so it is a
legitimate forecast input. The weather of month M is not: training uses what
was observed, but a forecast has to plug in a *scenario* for it:

* `forecast` -- for the month in progress: the days already observed plus
  the 16-day forecast (Open-Meteo), the remaining days at their normal rate.
  Months not covered by such an outlook fall back to `normal`;
* `normal`   -- the station's climatology for that calendar month (mean);
* `harsh`    -- the station's 90th percentile for that calendar month;
* `observed` -- what actually happened (only for past months: backtests,
  "what if we had known the weather").

The held-out score in `metrics.json` is computed with `normal`, i.e. as a
genuine one-month-ahead forecast without any weather forecast (past weather
forecasts are not archived here, so `forecast` cannot be backtested); the
`observed` score is reported next to it as the upper bound a perfect weather
forecast would reach. The served `forecast` scenario sits between the two.

Only the weather at the two ends of the route is used. A network-wide mean
was tried and dropped: with a single value per month, the trees used it to
memorise months and it hurt both walk-forward validation and the test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PERIOD = "period"
STATION = "station"

# --- weather, aggregated per station x month by `monthly_weather`
WEATHER_COLS = [
    "frost_days", "hot_days", "rain_mm", "heavy_rain_days", "snow_days", "windy_days",
]
# Thresholds on the daily values. Wind is the daily max of the hourly mean
# wind at 10 m (not gusts), hence the modest threshold.
FROST_MAX_C = 0.0
HOT_MIN_C = 32.0
HEAVY_RAIN_MM = 20.0
SNOW_MM = 1.0
WINDY_MS = 10.0
# Fewer valid days than this and the month is left missing rather than undercounted.
MIN_COVERAGE = 0.9

ROUTE_WEATHER = [f"wx_{c}" for c in WEATHER_COLS]      # mean of the two ends
CALENDAR_COLS = [
    "cal_holidays", "cal_long_weekends", "cal_bridges", "cal_weekend_days",
    "cal_school_any", "cal_school_all",
]
CONTEXT_COLUMNS = ROUTE_WEATHER + CALENDAR_COLS

WEATHER_MODES = ("forecast", "normal", "harsh", "observed")
SCHOOL_ZONES = ("A", "B", "C")

WEATHER_FILE = "weather_monthly.csv"
OUTLOOK_FILE = "weather_outlook.csv"
SCHOOL_FILE = "school_holidays.csv"


@dataclass
class Context:
    """Monthly weather per station, the current outlook and the school calendar."""

    weather: pd.DataFrame | None = None   # station, period, *WEATHER_COLS
    school: pd.DataFrame | None = None    # zone, start, end (end exclusive)
    # partial months: sums over the days known so far, and how many there are
    outlook: pd.DataFrame | None = None   # station, period, *WEATHER_COLS, known_days

    @classmethod
    def load(cls, directory: str | Path) -> Context:
        directory = Path(directory)
        frames: dict[str, pd.DataFrame | None] = {}
        for name, file in (("weather", WEATHER_FILE), ("outlook", OUTLOOK_FILE)):
            frames[name] = None
            if (directory / file).exists():
                df = pd.read_csv(directory / file)
                df[PERIOD] = pd.PeriodIndex(df.pop("date"), freq="M")
                frames[name] = df
        school = None
        if (directory / SCHOOL_FILE).exists():
            school = pd.read_csv(directory / SCHOOL_FILE, parse_dates=["start", "end"])
        return cls(frames["weather"], school, frames["outlook"])

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for df, file in ((self.weather, WEATHER_FILE), (self.outlook, OUTLOOK_FILE)):
            if df is not None:
                out = df.copy()
                out.insert(1, "date", out.pop(PERIOD).astype(str))
                out.to_csv(directory / file, index=False, float_format="%.3f")
        if self.school is not None:
            out = self.school.copy()
            for col in ("start", "end"):
                out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
            out.to_csv(directory / SCHOOL_FILE, index=False)

    def has_observed_weather(self, stations: list[str], period: pd.Period) -> bool:
        if self.weather is None:
            return False
        rows = self.weather[
            (self.weather[PERIOD] == period) & self.weather[STATION].isin(stations)
        ]
        return len(rows.dropna(subset=WEATHER_COLS)) == len(set(stations))


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #
def _daily_flags(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Daily values -> per-day contributions to `WEATHER_COLS` (NaN when the
    day is missing). `daily` columns: station, date, tmax, tmin (degC),
    precip, snow (mm of water), wind (m/s).
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d[PERIOD] = d["date"].dt.to_period("M")
    valid = d[["tmax", "tmin", "precip"]].notna().all(axis=1)
    d["frost_days"] = (d["tmin"] < FROST_MAX_C).where(valid)
    d["hot_days"] = (d["tmax"] >= HOT_MIN_C).where(valid)
    d["rain_mm"] = d["precip"].where(valid)
    d["heavy_rain_days"] = (d["precip"] >= HEAVY_RAIN_MM).where(valid)
    d["snow_days"] = (d["snow"] >= SNOW_MM).where(d["snow"].notna())
    d["windy_days"] = (d["wind"] >= WINDY_MS).where(d["wind"].notna())
    d["valid"] = valid
    return d


def monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily values -> one row per station x complete month."""
    g = _daily_flags(daily).groupby([STATION, PERIOD])
    out = g[WEATHER_COLS].sum(min_count=1)
    coverage = g["valid"].sum() / g[PERIOD].first().dt.days_in_month
    out.loc[coverage < MIN_COVERAGE, WEATHER_COLS] = np.nan
    return out.dropna(how="all").reset_index()


def monthly_outlook(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Daily observations + forecast -> per station x month, the sums over the
    days covered so far and their number (`known_days`). The rest of the
    month is filled with normals at use time, see `weather_scenario`.
    """
    d = _daily_flags(daily)
    d = d[d["valid"]]
    g = d.groupby([STATION, PERIOD])
    out = g[WEATHER_COLS].sum()
    out["known_days"] = g.size()
    return out.reset_index()


def weather_scenario(
    weather: pd.DataFrame,
    periods: pd.PeriodIndex,
    mode: str = "observed",
    climate_until: pd.Period | None = None,
    outlook: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Station x period weather for `periods` under a scenario (see module doc).

    Climatology is computed per station and calendar month over the months
    strictly before `climate_until` (all months when None), so an evaluation
    can keep the test period out of the normals. `observed` falls back to
    `normal` wherever nothing was observed (e.g. future months); so does
    `forecast` outside the months the outlook covers.
    """
    if mode not in WEATHER_MODES:
        raise ValueError(f"unknown weather mode {mode!r}, expected one of {WEATHER_MODES}")
    base = weather if climate_until is None else weather[weather[PERIOD] < climate_until]
    if base.empty:
        base = weather
    month = base[PERIOD].dt.month.rename("moy")
    grouped = base.groupby([base[STATION], month])[WEATHER_COLS]
    climate = grouped.quantile(0.9) if mode == "harsh" else grouped.mean()

    stations = weather[STATION].unique()
    grid = pd.MultiIndex.from_product([stations, periods], names=[STATION, PERIOD])
    grid = grid.to_frame(index=False)
    grid["moy"] = grid[PERIOD].dt.month
    out = grid.join(climate, on=[STATION, "moy"]).drop(columns="moy")
    keys = pd.MultiIndex.from_frame(out[[STATION, PERIOD]])

    if mode == "observed":
        observed = weather.set_index([STATION, PERIOD])[WEATHER_COLS]
        seen = observed.reindex(keys).to_numpy()
        out[WEATHER_COLS] = np.where(np.isnan(seen), out[WEATHER_COLS].to_numpy(), seen)

    if mode == "forecast" and outlook is not None and not outlook.empty:
        # only months the complete record does not cover yet: an outlook
        # written later must never leak into the backtest of a past month
        recent = outlook[outlook[PERIOD] > weather[PERIOD].max()]
        partial = recent.set_index([STATION, PERIOD]).reindex(keys)
        known = partial["known_days"].to_numpy()
        days = out[PERIOD].dt.days_in_month.to_numpy()
        unknown = np.clip(days - known, 0, None) / days
        normal = out[WEATHER_COLS].to_numpy()
        blended = partial[WEATHER_COLS].to_numpy() + normal * unknown[:, None]
        out[WEATHER_COLS] = np.where(np.isnan(known)[:, None], normal, blended)
    return out


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def french_holidays(year: int) -> list[date]:
    """The 11 public holidays of metropolitan France."""
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    e = easter(year)
    movable = [e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)]
    return sorted([date(year, m, d) for m, d in fixed] + movable)


def calendar_features(periods, school: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per month: weekday public holidays, long weekends (holiday on a
    Monday or Friday), "ponts" (holiday on a Tuesday or Thursday), weekend
    days, and days on school holiday in at least one / all three zones.
    School columns are NaN for months the school calendar does not cover.
    """
    periods = pd.PeriodIndex(sorted(set(periods)), freq="M")
    rows = []
    holidays: dict[int, list[date]] = {}
    for p in periods:
        days = pd.date_range(p.start_time, p.end_time.normalize(), freq="D")
        hols = holidays.setdefault(p.year, french_holidays(p.year))
        weekdays = [h.weekday() for h in hols if h.month == p.month]
        rows.append({
            PERIOD: p,
            "cal_holidays": sum(w < 5 for w in weekdays),
            "cal_long_weekends": sum(w in (0, 4) for w in weekdays),
            "cal_bridges": sum(w in (1, 3) for w in weekdays),
            "cal_weekend_days": int((days.dayofweek >= 5).sum()),
            **_school_days(school, days),
        })
    return pd.DataFrame(rows).set_index(PERIOD)


def _school_days(school: pd.DataFrame | None, days: pd.DatetimeIndex) -> dict:
    empty = {"cal_school_any": np.nan, "cal_school_all": np.nan}
    if school is None or school.empty:
        return empty
    # outside the published calendar, "no holiday" would be a lie
    if days[0] < school["start"].min() or days[-1] >= school["end"].max():
        return empty
    on = np.zeros((len(days), len(SCHOOL_ZONES)), dtype=bool)
    for z, zone in enumerate(SCHOOL_ZONES):
        for row in school[school["zone"] == zone].itertuples():
            on[:, z] |= (days >= row.start) & (days < row.end)
    zones_on = on.sum(axis=1)
    return {"cal_school_any": int((zones_on >= 1).sum()),
            "cal_school_all": int((zones_on == len(SCHOOL_ZONES)).sum())}


# --------------------------------------------------------------------------- #
# joining the context onto the feature panel
# --------------------------------------------------------------------------- #
def add_context(
    df: pd.DataFrame,
    context: Context | None,
    weather_mode: str = "observed",
    climate_until: pd.Period | None = None,
) -> pd.DataFrame:
    """
    Add `CONTEXT_COLUMNS` to a panel with `gare_depart`, `gare_arrivee` and
    `period`. Route weather is the mean of the departure and arrival
    stations. Anything the context does not cover is left NaN (the model
    sees it as missing).
    """
    df = df.copy()
    context = context or Context()
    periods = pd.PeriodIndex(df[PERIOD].unique(), freq="M")

    cal = calendar_features(periods, context.school)
    df[CALENDAR_COLS] = cal.reindex(df[PERIOD]).to_numpy(dtype=float)

    if context.weather is None or context.weather.empty:
        df[ROUTE_WEATHER] = np.nan
        return df

    wx = weather_scenario(context.weather, periods, weather_mode, climate_until,
                          context.outlook)
    by_station = wx.set_index([STATION, PERIOD])[WEATHER_COLS]
    ends = [
        by_station.reindex(pd.MultiIndex.from_arrays([df[col], df[PERIOD]])).to_numpy()
        for col in ("gare_depart", "gare_arrivee")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # both ends unknown -> NaN
        df[ROUTE_WEATHER] = np.nanmean(np.stack(ends), axis=0)
    return df
