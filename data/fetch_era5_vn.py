#!/usr/bin/env python3
"""Fetch real ERA5 hourly 2-m temperature for the three Mekong-delta stations
used in the RABS climate replay, from the Open-Meteo historical weather API.

Reproducible: given the same station coordinates and year, the API returns the
same ERA5 reanalysis series. Output CSVs (time, temp_c) are written under
data/era5_vn/ and consumed by code/run_rabs_era5_main.py and friends.

Usage:
    python3 data/fetch_era5_vn.py            # fetches 2024 for all 3 stations

Stdlib-only (urllib + csv); no API key required.
"""
from __future__ import annotations
import csv
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "era5_vn"
OUT.mkdir(parents=True, exist_ok=True)

# Mekong-delta stations (lat, lon) — same three zones as the manuscript.
STATIONS = {
    "can_tho":   (10.0452, 105.7469),
    "soc_trang": (9.6037, 105.9739),
    "ca_mau":    (9.1769, 105.1524),
}
YEAR = 2024
BASE = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float) -> list[tuple[str, float]]:
    url = (f"{BASE}?latitude={lat}&longitude={lon}"
           f"&start_date={YEAR}-01-01&end_date={YEAR}-12-31"
           f"&hourly=temperature_2m&timezone=Asia%2FBangkok")
    req = urllib.request.Request(url, headers={"User-Agent": "rabs-repro/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    h = data["hourly"]
    return list(zip(h["time"], h["temperature_2m"]))


def main() -> None:
    for name, (lat, lon) in STATIONS.items():
        rows = fetch(lat, lon)
        # forward-fill any missing values so the replay never sees gaps
        last = None
        clean = []
        for t, v in rows:
            if v is None:
                v = last
            else:
                last = v
            clean.append((t, v))
        path = OUT / f"{name}_{YEAR}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "temp_c"])
            w.writerows(clean)
        temps = [v for _, v in clean if v is not None]
        print(f"{name}: {len(clean)} hours -> {path}  "
              f"range[{min(temps):.1f},{max(temps):.1f}]C")
        time.sleep(1)  # be polite to the API


if __name__ == "__main__":
    main()
