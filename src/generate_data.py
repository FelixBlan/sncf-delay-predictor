"""
Synthetic stand-in for the real SNCF export, used by the tests and by CI.

The real dataset is public but not redistributed in this repository (see
`fetch_data.py`), and it is too large to commit as a fixture. This generator
produces a monthly OD panel with the *same schema, separator and quirks* as
the real export, so the pipeline, the training script and the API can be
exercised end to end without any download:

* the same French column names and `;` separator;
* OD pairs with a persistent delay level, a seasonal component and a
  network-wide drift -- i.e. the structure the lag features exploit;
* the same data-quality traps: months with zero traffic, a handful of
  corrupt negative averages, and a free-text comment column containing
  embedded newlines and semicolons;
* a matching context (`generate_context`): monthly weather per station,
  which the synthetic delays depend on, and a school-holiday calendar.

It is *not* a substitute for the real data when reporting metrics: the
numbers in the README come from the real export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from context import PERIOD, STATION, WEATHER_COLS, Context

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "raw_regularity.csv"

COLUMNS = [
    "date", "service", "gare_depart", "gare_arrivee", "duree_moyenne",
    "nb_train_prevu", "nb_annulation", "commentaire_annulation",
    "nb_train_depart_retard", "retard_moyen_depart",
    "retard_moyen_tous_trains_depart", "commentaire_retards_depart",
    "nb_train_retard_arrivee", "retard_moyen_arrivee",
    "retard_moyen_tous_trains_arrivee", "commentaires_retard_arrivee",
    "nb_train_retard_sup_15", "retard_moyen_trains_retard_sup15",
    "nb_train_retard_sup_30", "nb_train_retard_sup_60",
    "prct_cause_externe", "prct_cause_infra", "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant", "prct_cause_gestion_gare",
    "prct_cause_prise_en_charge_voyageurs",
]

ROUTES = [
    ("PARIS MONTPARNASSE", "BORDEAUX ST JEAN", 141),
    ("PARIS LYON", "MARSEILLE ST CHARLES", 195),
    ("PARIS NORD", "LILLE", 63),
    ("PARIS EST", "STRASBOURG", 110),
    ("LYON PART DIEU", "MARSEILLE ST CHARLES", 105),
    ("PARIS MONTPARNASSE", "NANTES", 124),
    ("PARIS MONTPARNASSE", "RENNES", 88),
    ("PARIS LYON", "GENEVE", 189),
]

_INCIDENT_COMMENT = (
    'Ce mois-ci, l\'OD a ete touchee par les incidents suivants :\n'
    "Le 3 : Derangement d'une aiguille (35 TGV ; 407mn);\n"
    "Le 12 : Incident catenaire (26 TGV ; 2188mn)"
)


def _synthetic_weather(months: pd.PeriodIndex, seed: int) -> pd.DataFrame:
    """Seasonal monthly weather for every station of `ROUTES`."""
    rng = np.random.default_rng(seed + 1000)
    stations = sorted({s for dep, arr, _ in ROUTES for s in (dep, arr)})
    rows = []
    for station in stations:
        for month in months:
            winter, summer = month.month in (12, 1, 2), month.month in (6, 7, 8)
            frost = rng.poisson(8 if winter else 2 if month.month in (3, 11) else 0)
            rain = rng.gamma(4.0, 15.0)
            rows.append({
                STATION: station, PERIOD: month,
                "frost_days": frost,
                "hot_days": rng.poisson(5 if summer else 0),
                "rain_mm": round(rain, 1),
                "heavy_rain_days": rng.poisson(rain / 80),
                "snow_days": rng.poisson(frost / 4),
                "windy_days": rng.poisson(3 if winter else 1),
            })
    return pd.DataFrame(rows)[[STATION, PERIOD, *WEATHER_COLS]]


def _synthetic_school(months: pd.PeriodIndex) -> pd.DataFrame:
    """Summer for all zones, and winter holidays staggered by zone."""
    rows = []
    for year in sorted({m.year for m in months}):
        for z, zone in enumerate("ABC"):
            rows.append({"zone": zone, "start": pd.Timestamp(year, 7, 5),
                         "end": pd.Timestamp(year, 9, 1)})
            winter = pd.Timestamp(year, 2, 7) + pd.Timedelta(weeks=z)
            rows.append({"zone": zone, "start": winter,
                         "end": winter + pd.Timedelta(weeks=2)})
    return pd.DataFrame(rows)


def generate_context(n_months: int = 96, seed: int = 42, start: str = "2018-01") -> Context:
    """The weather and school calendar that go with `generate(...)`."""
    months = pd.period_range(start, periods=n_months, freq="M")
    return Context(_synthetic_weather(months, seed), _synthetic_school(months))


def generate(n_months: int = 96, seed: int = 42, start: str = "2018-01") -> pd.DataFrame:
    """Build a synthetic monthly panel over `n_months` for every route."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, periods=n_months, freq="M")
    weather = _synthetic_weather(months, seed).set_index([STATION, PERIOD])

    # per-route persistent delay level, and a slow network-wide drift
    levels = {route: rng.uniform(3.0, 9.0) for route in ROUTES}
    drift = np.cumsum(rng.normal(0.02, 0.15, n_months))

    rows = []
    for route in ROUTES:
        dep, arr, duration = route
        service = "International" if arr in {"GENEVE"} else "National"
        level = levels[route]
        previous = level
        for i, month in enumerate(months):
            # seasonality: worse in summer holidays and in December
            seasonal = 1.6 if month.month in (7, 8, 12) else 0.0
            # frost and heat at either end of the line cost minutes
            ends = weather.loc[[(dep, month), (arr, month)]].mean()
            weather_effect = 0.15 * ends["frost_days"] + 0.25 * ends["hot_days"]
            # autocorrelated: this month resembles the last one
            delay = max(
                0.2,
                0.55 * previous + 0.45 * level + seasonal + weather_effect + drift[i]
                + rng.normal(0, 1.2),
            )
            previous = delay

            planned = int(np.clip(rng.normal(250, 60), 40, 500))
            cancelled = int(rng.poisson(planned * 0.01))
            late_arrival = int(np.clip(rng.normal(planned * 0.22, 30), 0, planned))
            late_15 = int(late_arrival * rng.uniform(0.3, 0.6))
            late_30 = int(late_15 * rng.uniform(0.2, 0.5))
            late_60 = int(late_30 * rng.uniform(0.1, 0.4))
            causes = rng.dirichlet(np.ones(6)) * 100

            comment = _INCIDENT_COMMENT if rng.random() < 0.05 else ""
            rows.append([
                str(month), service, dep, arr, duration, planned, cancelled, "",
                int(late_arrival * 1.3), round(delay * 1.8, 6),
                round(delay * 0.7, 6), "",
                late_arrival, round(delay * 4, 6), round(delay, 6), comment,
                late_15, round(delay * 4, 6), late_30, late_60,
                *[round(c, 6) for c in causes],
            ])

    df = pd.DataFrame(rows, columns=COLUMNS)
    return _inject_quality_issues(df, rng)


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reproduce the defects the cleaning step has to survive."""
    target = "retard_moyen_tous_trains_arrivee"

    # a month with no service at all (April 2020 in the real data)
    months = df["date"].unique()
    blackout = df["date"] == months[min(27, len(months) - 1)]
    df.loc[blackout, ["nb_train_prevu", "duree_moyenne", "nb_annulation"]] = 0
    df.loc[blackout, target] = 0.0

    # corrupt aggregates, as found around the December 2019 strike
    corrupt = rng.choice(df.index, size=max(2, len(df) // 400), replace=False)
    df.loc[corrupt, target] = -rng.uniform(50, 400, len(corrupt))

    # a couple of ODs served by too few trains to average meaningfully
    tiny = rng.choice(df.index, size=max(2, len(df) // 500), replace=False)
    df.loc[tiny, "nb_train_prevu"] = rng.integers(1, 8, len(tiny))
    df.loc[tiny, target] = rng.uniform(40, 90, len(tiny))
    return df


def write(path: str | Path = DEFAULT_OUT, **kwargs) -> pd.DataFrame:
    df = generate(**kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--months", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-dir", default=None,
                        help="also write the matching weather / school calendar there")
    args = parser.parse_args()

    df = write(args.out, n_months=args.months, seed=args.seed)
    print(f"Wrote {len(df)} rows ({df['date'].nunique()} months, "
          f"{len(ROUTES)} routes) to {args.out}")
    print(df.head(3).to_string())
    if args.context_dir:
        generate_context(n_months=args.months, seed=args.seed).save(args.context_dir)
        print(f"Wrote the synthetic context to {args.context_dir}")
