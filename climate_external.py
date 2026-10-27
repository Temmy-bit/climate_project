"""External daily weather (ERA5-based, via Open-Meteo, no API key) -> lagged-window features.

Two steps:
    python climate_external.py fetch --data-dir DIR --daily daily_weather.csv
    python climate_external.py build --data-dir DIR --daily daily_weather.csv --out external_features.csv

`fetch` downloads daily weather for every unique (lat, lon) in Train/Test (about 55 calls) and caches
the result, so re-running resumes where it stopped. `build` turns it into one row of features per ID,
using only weather BEFORE the death date (window ends the day before).

Then:  python climate_blend.py --data-dir DIR --external external_features.csv

Needs internet access (run it in Colab or locally). Requires: pandas numpy requests
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
              "et0_fao_evapotranspiration", "shortwave_radiation_sum", "wind_speed_10m_max"]
START, END = "2006-06-01", "2022-12-31"


def unique_coords(data_dir):
    tr = pd.read_csv(f"{data_dir}/Train_1.csv")
    te = pd.read_csv(f"{data_dir}/Test_1.csv")
    c = pd.concat([tr[["latitude", "longitude"]], te[["latitude", "longitude"]]])
    c = c.round(3).drop_duplicates().reset_index(drop=True)
    return c


# ---------------------------------------------------------------- fetch
def _get(lat, lon, retries=6):
    import requests
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "daily": ",".join(DAILY_VARS), "timezone": "Africa/Kampala"}
    for attempt in range(retries):
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:  # rate limited: wait and retry
            wait = 65 * (attempt + 1)
            print(f"  429 rate limit, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        print(f"  HTTP {r.status_code}: {r.text[:200]}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"failed for {lat},{lon}")


def parse(js, lat, lon):
    d = js["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    for v in DAILY_VARS:
        df[v] = pd.to_numeric(pd.Series(d[v]), errors="coerce").values  # null -> NaN
    df.insert(0, "lon", lon)
    df.insert(0, "lat", lat)
    return df


def fetch(data_dir, daily_csv):
    coords = unique_coords(data_dir)
    done = set()
    if os.path.exists(daily_csv):
        prev = pd.read_csv(daily_csv, usecols=["lat", "lon"]).drop_duplicates()
        done = set(map(tuple, prev.round(3).values))
    print(f"{len(coords)} unique coordinates, {len(done)} already cached")
    first = not os.path.exists(daily_csv)
    for i, (lat, lon) in enumerate(coords.itertuples(index=False)):
        if (round(lat, 3), round(lon, 3)) in done:
            continue
        df = parse(_get(lat, lon), lat, lon)
        df.to_csv(daily_csv, mode="a", header=first, index=False)
        first = False
        print(f"[{i + 1}/{len(coords)}] {lat},{lon}: {len(df)} days", flush=True)
        time.sleep(1.0)
    print("fetch done ->", daily_csv)


# ---------------------------------------------------------------- features
def coord_features(g):
    """g: daily weather for ONE coordinate, sorted by date. Returns features indexed by date, where the
    value on date d summarises days d-1, d-2, ... (the death date itself is excluded)."""
    g = g.sort_values("date").set_index("date")
    tmax, tmin, rain = g.temperature_2m_max, g.temperature_2m_min, g.precipitation_sum
    tmean = (tmax + tmin) / 2
    F = {}
    for w in (7, 14, 30, 60, 90):
        F[f"x_tmean_{w}"] = tmean.rolling(w, min_periods=w).mean().shift(1)
        F[f"x_tmax_{w}"] = tmax.rolling(w, min_periods=w).mean().shift(1)
        F[f"x_rain_{w}"] = rain.rolling(w, min_periods=w).sum().shift(1)
        F[f"x_wet_{w}"] = (rain >= 1).astype(float).where(rain.notna()).rolling(w, min_periods=w).sum().shift(1)
    hot = (tmax >= tmax.quantile(0.9)).astype(float).where(tmax.notna())
    F["x_hot_14"] = hot.rolling(14, min_periods=14).sum().shift(1)
    F["x_hot_30"] = hot.rolling(30, min_periods=30).sum().shift(1)
    F["x_trange_30"] = (tmax - tmin).rolling(30, min_periods=30).mean().shift(1)
    F["x_et0_30"] = g.et0_fao_evapotranspiration.rolling(30, min_periods=30).sum().shift(1)
    F["x_rad_30"] = g.shortwave_radiation_sum.rolling(30, min_periods=30).mean().shift(1)
    F["x_wind_30"] = g.wind_speed_10m_max.rolling(30, min_periods=30).mean().shift(1)
    # rainfall lag blocks (transmission-style lags: 1-7, 8-14, 15-28, 29-56 days before)
    F["x_rain_lag_1_7"] = rain.rolling(7, min_periods=7).sum().shift(1)
    F["x_rain_lag_8_14"] = rain.rolling(7, min_periods=7).sum().shift(8)
    F["x_rain_lag_15_28"] = rain.rolling(14, min_periods=14).sum().shift(15)
    F["x_rain_lag_29_56"] = rain.rolling(28, min_periods=28).sum().shift(29)
    # days since last wet day (capped at 90)
    idx = np.arange(len(g), dtype=float)
    last_wet = pd.Series(np.where((rain >= 1).values, idx, np.nan), index=g.index).ffill()
    F["x_dry_days"] = (pd.Series(idx, index=g.index) - last_wet).clip(upper=90).shift(1)
    F["x_dry_ratio_30_90"] = F["x_rain_30"] / (F["x_rain_90"] + 1.0)
    F = pd.DataFrame(F, index=g.index)
    # anomalies vs this coordinate's own month climatology (z-scores)
    for c in ("x_rain_30", "x_rain_90", "x_tmean_30", "x_tmax_14"):
        grp = F[c].groupby(F.index.month)
        F[c.replace("x_", "xz_", 1)] = (F[c] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
    return F


def build(data_dir, daily_csv, out_csv):
    daily = pd.read_csv(daily_csv, parse_dates=["date"])
    daily["lat"], daily["lon"] = daily.lat.round(3), daily.lon.round(3)
    parts = []
    for (lat, lon), g in daily.groupby(["lat", "lon"]):
        F = coord_features(g)
        F.insert(0, "lon", lon)
        F.insert(0, "lat", lat)
        parts.append(F.reset_index())
    feats = pd.concat(parts, ignore_index=True)

    rows = []
    for name in ("Train_1.csv", "Test_1.csv"):
        d = pd.read_csv(f"{data_dir}/{name}", usecols=["ID", "latitude", "longitude", "deathdate"])
        rows.append(d)
    rows = pd.concat(rows, ignore_index=True)
    rows["lat"], rows["lon"] = rows.latitude.round(3), rows.longitude.round(3)
    rows["date"] = pd.to_datetime(rows.deathdate)
    out = rows[["ID", "lat", "lon", "date"]].merge(feats, on=["lat", "lon", "date"], how="left")
    assert len(out) == len(rows), "merge changed row count"
    out = out.drop(columns=["lat", "lon", "date"])
    cols = [c for c in out.columns if c != "ID"]
    out[cols] = out[cols].astype("float32")
    out.to_csv(out_csv, index=False)
    miss = out[cols].isna().mean()
    print(f"wrote {out_csv}: {len(out)} rows x {len(cols)} features; "
          f"rows with any NaN {out[cols].isna().any(axis=1).mean():.1%}; worst column NaN {miss.max():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "build"])
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--daily", default="daily_weather.csv")
    ap.add_argument("--out", default="external_features.csv")
    a = ap.parse_args()
    fetch(a.data_dir, a.daily) if a.step == "fetch" else build(a.data_dir, a.daily, a.out)
