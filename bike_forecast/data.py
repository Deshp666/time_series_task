"""Immutable raw data and strictly causal training preparation."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SOURCE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
WEATHER = ["temp", "atemp", "hum", "windspeed", "weathersit"]
CALENDAR = ["hour", "weekday", "month", "weekend", "holiday"]


def download() -> dict:
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "bike-sharing.zip"
    if not archive.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "bike-forecast-course/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            if z.testzip() is not None:
                raise ValueError("Damaged UCI archive")
        archive.write_bytes(content)
    content = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in ("hour.csv", "day.csv", "Readme.txt"):
            candidates = [n for n in z.namelist() if Path(n).name.lower() == name.lower()]
            if candidates:
                (raw / name).write_bytes(z.read(candidates[0]))
    metadata_path = raw / "source.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata = {
        "source": SOURCE, "download_url": URL,
        "downloaded_at": old.get("downloaded_at", datetime.now(timezone.utc).isoformat()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "license": "CC BY 4.0; attribute UCI and dataset authors",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "authors": "Hadi Fanaee-T, Joao Gama", "doi": "10.24432/C5W894",
        "teacher_approval": "not recorded",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    prepare()
    return metadata


def prepare() -> pd.DataFrame:
    path = ROOT / "data/raw/hour.csv"
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    if df.ds.duplicated().any():
        raise ValueError("Duplicate local timestamps: resolve explicitly, never silently aggregate")
    df = df.set_index("ds").sort_index()[["cnt", *WEATHER]].rename(columns={"cnt": "y"})
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="ds"))
    df["observed"] = df.y.notna()
    if (df.y.dropna() < 0).any():
        raise ValueError("Negative counts")
    df = df.reset_index()
    df.insert(0, "unique_id", "capital_bikeshare")
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hourly.csv", index=False)
    df[["ds", "observed"]].to_csv(out / "observation_mask.csv", index=False)
    return df


def load_data() -> pd.DataFrame:
    path = ROOT / "data/processed/hourly.csv"
    if not path.exists():
        return prepare()
    return pd.read_csv(path, parse_dates=["ds"])


def causal_fill(values: pd.Series, period: int = 168) -> pd.Series:
    arr = values.to_numpy(dtype=float, copy=True)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            if i >= period and np.isfinite(arr[i-period]):
                arr[i] = arr[i-period]
            elif i and np.isfinite(arr[i-1]):
                arr[i] = arr[i-1]
    return pd.Series(arr, index=values.index)


def training_frame(df: pd.DataFrame, origin: str, missing: float = 0, seed: int = 42) -> pd.DataFrame:
    train = df.loc[df.ds < pd.Timestamp(origin)].copy()
    if missing:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(train), int(len(train)*missing), replace=False)
        train.iloc[positions, train.columns.get_loc("y")] = np.nan
    train["y"] = causal_fill(train.y)
    train = train.dropna(subset=["y"]).reset_index(drop=True)
    if train.empty or train.ds.max() >= pd.Timestamp(origin):
        raise ValueError("Invalid training cutoff")
    if not (train.ds.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Training series is not regular")
    return train


def calendar(dates: pd.Series) -> pd.DataFrame:
    ds = pd.DatetimeIndex(dates)
    # Known calendar only, not future observed fields in hour.csv.
    h = holidays.US(subdiv="DC", years=sorted(set(ds.year)))
    return pd.DataFrame({"hour": ds.hour, "weekday": ds.dayofweek,
                         "month": ds.month, "weekend": (ds.dayofweek >= 5).astype(int),
                         "holiday": [int(d.date() in h) for d in ds]})


def design_frames(train: pd.DataFrame, horizon: int, use_calendar=True, weather=False):
    """Future covariates depend only on the training frame and known calendar.

    For ML, a 168-hour weather lag is available across both forecast horizons.
    The same covariate is supplied to DL to keep this ablation comparable.
    """
    dates = pd.date_range(train.ds.iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    x = train[["unique_id", "ds", "y"]].reset_index(drop=True).copy()
    future = pd.DataFrame({"unique_id": train.unique_id.iloc[0], "ds": dates})
    features = []
    if use_calendar:
        features += CALENDAR
        x[CALENDAR] = calendar(x.ds).to_numpy()
        future[CALENDAR] = calendar(future.ds).to_numpy()
    if weather:
        if horizon > 168:
            raise ValueError("168-hour historical weather lag cannot support a longer horizon")
        for col in WEATHER:
            name = col + "_lag168"
            past = train[col].reset_index(drop=True).ffill().fillna(0)
            x[name] = past.shift(168).fillna(0)
            future[name] = past.iloc[-168: -168+horizon if horizon < 168 else None].to_numpy()
            features.append(name)
    return x, future, features
