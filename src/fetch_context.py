"""
Downloads the exogenous context used by the model (see `context.py`):

* daily weather at every station from NASA POWER (MERRA-2 / GEOS reanalysis,
  free, no API key, one request per point for the whole period), aggregated
  per station x month;
* the outlook of the month in progress from Open-Meteo (past 31 days +
  16-day forecast, free, no API key), used by the `forecast` scenario;
* the official school-holiday calendar (zones A/B/C) from the Ministry of
  Education's open-data portal.

All are written to data/ (`weather_monthly.csv`, `weather_outlook.csv`,
`school_holidays.csv`); `train.py` picks them up from there and ships a copy
with the model. Re-run it right before serving a new month to refresh the
outlook.

Usage:
    python src/fetch_context.py
    python src/fetch_context.py --start 2017-12-01 --out-dir data
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from context import STATION, Context, monthly_outlook, monthly_weather

ROOT = Path(__file__).resolve().parent.parent

# Station -> (lat, lon). City-level precision is plenty: the reanalysis grid
# is ~50 km. Paris termini share one point. "ITALIE" is the Paris - Turin -
# Milan service; Turin, right after the Alpine crossing, stands for it.
PARIS = (48.857, 2.352)
STATIONS: dict[str, tuple[float, float]] = {
    "AIX EN PROVENCE TGV": (43.455, 5.317),
    "ANGERS SAINT LAUD": (47.464, -0.556),
    "ANGOULEME": (45.653, 0.165),
    "ANNECY": (45.902, 6.122),
    "ARRAS": (50.287, 2.781),
    "AVIGNON TGV": (43.922, 4.786),
    "BARCELONA": (41.379, 2.140),
    "BELLEGARDE (AIN)": (46.108, 5.826),
    "BESANCON FRANCHE COMTE TGV": (47.307, 5.954),
    "BORDEAUX ST JEAN": (44.826, -0.556),
    "BREST": (48.388, -4.479),
    "CHAMBERY CHALLES LES EAUX": (45.571, 5.920),
    "DIJON VILLE": (47.323, 5.027),
    "DOUAI": (50.371, 3.090),
    "DUNKERQUE": (51.031, 2.369),
    "FRANCFORT": (50.107, 8.663),
    "GENEVE": (46.210, 6.142),
    "GRENOBLE": (45.191, 5.715),
    "ITALIE": (45.062, 7.678),
    "LA ROCHELLE VILLE": (46.153, -1.145),
    "LAUSANNE": (46.517, 6.629),
    "LAVAL": (48.076, -0.762),
    "LE CREUSOT MONTCEAU MONTCHANIN": (46.765, 4.499),
    "LE MANS": (47.995, 0.192),
    "LILLE": (50.637, 3.071),
    "LYON PART DIEU": (45.760, 4.859),
    "MACON LOCHE": (46.282, 4.779),
    "MADRID": (40.407, -3.691),
    "MARNE LA VALLEE": (48.870, 2.783),
    "MARSEILLE ST CHARLES": (43.303, 5.380),
    "METZ": (49.110, 6.177),
    "MONTPELLIER": (43.605, 3.881),
    "MULHOUSE VILLE": (47.742, 7.343),
    "NANCY": (48.690, 6.174),
    "NANTES": (47.217, -1.542),
    "NICE VILLE": (43.704, 7.262),
    "NIMES": (43.833, 4.366),
    "PARIS EST": PARIS,
    "PARIS LYON": PARIS,
    "PARIS MONTPARNASSE": PARIS,
    "PARIS NORD": PARIS,
    "PARIS VAUGIRARD": PARIS,
    "PERPIGNAN": (42.696, 2.879),
    "POITIERS": (46.582, 0.333),
    "QUIMPER": (47.995, -4.093),
    "REIMS": (49.259, 4.024),
    "RENNES": (48.103, -1.672),
    "SAINT ETIENNE CHATEAUCREUX": (45.443, 4.399),
    "ST MALO": (48.646, -2.003),
    "ST PIERRE DES CORPS": (47.386, 0.723),
    "STRASBOURG": (48.585, 7.735),
    "STUTTGART": (48.784, 9.182),
    "TOULON": (43.128, 5.929),
    "TOULOUSE MATABIAU": (43.611, 1.453),
    "TOURCOING": (50.717, 3.161),
    "TOURS": (47.390, 0.694),
    "VALENCE ALIXAN TGV": (44.991, 4.978),
    "VANNES": (47.665, -2.752),
    "ZURICH": (47.378, 8.540),
}

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,PRECSNOLAND,WS10M_MAX"
    "&community=AG&latitude={lat}&longitude={lon}&start={start}&end={end}"
    "&format=CSV&header=false"
)
POWER_COLUMNS = {"T2M_MAX": "tmax", "T2M_MIN": "tmin", "PRECTOTCORR": "precip",
                 "PRECSNOLAND": "snow", "WS10M_MAX": "wind"}

OUTLOOK_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,"
    "wind_speed_10m_max&wind_speed_unit=ms&timezone=Europe%2FParis"
    "&past_days=31&forecast_days=16"
)
OUTLOOK_COLUMNS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
                   "precipitation_sum": "precip", "snowfall_sum": "snow",
                   "wind_speed_10m_max": "wind"}

SCHOOL_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/exports/json"
    "?select=description,population,start_date,end_date,zones"
    "&where=zones%20in%20(%22Zone%20A%22,%22Zone%20B%22,%22Zone%20C%22)"
)


def _get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sncf-delay-predictor"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_point_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    url = POWER_URL.format(lat=lat, lon=lon, start=f"{start:%Y%m%d}", end=f"{end:%Y%m%d}")
    raw = pd.read_csv(io.BytesIO(_get(url)), na_values=[-999, -999.0])
    raw["date"] = pd.to_datetime(raw["YEAR"].astype(str), format="%Y") + pd.to_timedelta(
        raw["DOY"] - 1, unit="D"
    )
    return raw.rename(columns=POWER_COLUMNS)[["date", *POWER_COLUMNS.values()]]


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    """Daily weather for every station, one request per distinct point."""
    by_point: dict[tuple[float, float], pd.DataFrame] = {}
    frames = []
    for i, (station, point) in enumerate(sorted(STATIONS.items()), 1):
        if point not in by_point:
            by_point[point] = fetch_point_weather(*point, start, end)
        print(f"  [{i:>2}/{len(STATIONS)}] {station}")
        frames.append(by_point[point].assign(**{STATION: station}))
    return pd.concat(frames, ignore_index=True)


def fetch_outlook() -> pd.DataFrame:
    """
    Last 31 days + 16-day forecast for every station, in one request (the
    API takes a list of points). Snowfall comes in cm of snow, about the
    same number of mm of water, which is what the threshold expects.
    """
    points = sorted(set(STATIONS.values()))
    url = OUTLOOK_URL.format(lat=",".join(str(p[0]) for p in points),
                             lon=",".join(str(p[1]) for p in points))
    body = json.loads(_get(url))
    body = body if isinstance(body, list) else [body]
    by_point = {}
    for point, item in zip(points, body, strict=True):
        daily = pd.DataFrame(item["daily"]).rename(columns={"time": "date", **OUTLOOK_COLUMNS})
        by_point[point] = daily[["date", *OUTLOOK_COLUMNS.values()]]
    return pd.concat(
        [by_point[point].assign(**{STATION: station}) for station, point in STATIONS.items()],
        ignore_index=True,
    )


def fetch_school_holidays() -> pd.DataFrame:
    """
    Holiday periods per zone, as half-open [start, end) local dates. The
    portal stores them as UTC instants of local midnight, e.g. a holiday
    starting on Saturday 21 Oct 2017 is "2017-10-20T22:00:00+00:00".
    """
    records = json.loads(_get(SCHOOL_URL))
    df = pd.DataFrame(records)
    df = df[df["population"].fillna("-").isin(["-", "Élèves"])]
    df = df.dropna(subset=["start_date", "end_date"])
    local = {
        col: pd.to_datetime(df[col], utc=True).dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None).dt.normalize()
        for col in ("start_date", "end_date")
    }
    out = pd.DataFrame({
        "zone": df["zones"].str.replace("Zone ", "", regex=False),
        "description": df["description"].str.strip(),
        "start": local["start_date"],
        "end": local["end_date"],
    })
    out = out[out["end"] > out["start"]]
    return out.drop_duplicates(["zone", "start", "end"]).sort_values(["start", "zone"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-12-01",
                        help="first day of weather (a month before the AQST data, for lags)")
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.today() - timedelta(days=1)
    print(f"Weather {start} -> {end}, {len(set(STATIONS.values()))} points (NASA POWER)")
    weather = monthly_weather(fetch_weather(start, end))
    print("Outlook: past 31 days + 16-day forecast (Open-Meteo)")
    outlook = monthly_outlook(fetch_outlook())
    print("School holidays (data.education.gouv.fr)")
    school = fetch_school_holidays()

    Context(weather, school, outlook).save(args.out_dir)
    current = outlook[outlook["period"] > weather["period"].max()]
    print(f"Wrote {len(weather)} station-months (up to {weather['period'].max()}), "
          f"an outlook of {current['known_days'].max():.0f} days for "
          f"{', '.join(sorted(current['period'].astype(str).unique()))}, and "
          f"{len(school)} holiday periods "
          f"({school['start'].min():%Y-%m} -> {school['end'].max():%Y-%m}) to {args.out_dir}")
    print("Next: python src/train.py")
