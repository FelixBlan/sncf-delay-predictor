"""
Downloads the real dataset from SNCF's open-data portal.

Dataset: "Regularite mensuelle TGV par liaisons" (AQST), published under the
ODbL on https://data.sncf.com. It is fetched rather than committed:
it is ~3 MB, it gets a new row per OD pair every month, and the portal is
the authoritative copy.

Usage:
    python src/fetch_data.py                     # -> data/regularite-mensuelle-tgv-aqst.csv
    python src/fetch_data.py --out somewhere.csv
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

DATASET = "regularite-mensuelle-tgv-aqst"
EXPORT_URL = (
    "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/"
    f"{DATASET}/exports/csv?delimiter=%3B"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / f"{DATASET}.csv"


def fetch(url: str = EXPORT_URL, out: Path | str = DEFAULT_OUT, timeout: int = 120) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sncf-delay-predictor"})
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--url", default=EXPORT_URL)
    args = parser.parse_args()

    path = fetch(args.url, args.out)
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path} ({size_mb:.1f} MB)")
    print("Next: python src/train.py")
