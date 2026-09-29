"""Hourly multivariate time series for one-step-ahead forecasting.

    load(dataset, seq_len, train_frac, n_cap) -> Splits

Task: given the last `seq_len` rows of every channel (the target's own history
included), predict the next row's target. MAE is in standardized units, so it
is comparable across settings.

No dataset is downloaded by hand: each is fetched on first use from a static
URL (no key, no login) into data/ (git-ignored).

    air_quality   UCI Air Quality, target CO(GT), ~9.3k hourly rows
    beijing_pm25  UCI Beijing PM2.5, target pm2.5, ~43.8k hourly rows
    etth1         ETTh1 (electricity transformer), target OT, ~17.4k hourly rows

Splits are chronological by row: the first 70% is train, the next 10% is
validation, the last 20% is test. Windows are cut inside each segment, so no
row is shared by two segments' windows: there is no train/test leakage. Missing
values are forward-filled (causal; leading rows without a value are dropped),
and features and target are z-scored with statistics of the full train segment.
`train_frac` keeps only the most recent part of the train segment (the
data-size axis), while validation and test stay fixed, so every data size is
scored on the same targets in the same units.
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SPLITS = (0.7, 0.1, 0.2)   # train / val / test, by row

UCI = "https://archive.ics.uci.edu/static/public/{id}/{slug}.zip"
ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def _fetch(url, name, member=None):
    """Download `url` (unzipping `member` if given) into data/<name>, once."""
    path = DATA_DIR / name
    if path.exists():
        return path
    DATA_DIR.mkdir(exist_ok=True)
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        blob = resp.read()
    if member:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open(member) as f:
            blob = f.read()
    path.write_bytes(blob)
    return path


def _air_quality():
    path = _fetch(UCI.format(id=360, slug="air+quality"), "AirQualityUCI.csv",
                  "AirQualityUCI.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])
    df = df.dropna(how="all")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    # NMHC(GT) is missing for ~90% of rows past the first two months
    df = df.drop(columns=["Date", "Time", "NMHC(GT)"]).set_index("timestamp").sort_index()
    return df.replace(-200.0, np.nan), "CO(GT)"


def _beijing_pm25():
    path = _fetch(UCI.format(id=381, slug="beijing+pm2+5+data"),
                  "PRSA_data_2010.1.1-2014.12.31.csv", "PRSA_data_2010.1.1-2014.12.31.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"]).set_index("timestamp")
    df = pd.concat([df.drop(columns="cbwd"),
                    pd.get_dummies(df["cbwd"], prefix="wind", dtype=float)], axis=1)
    return df.sort_index(), "pm2.5"


def _etth1():
    path = _fetch(ETT, "ETTh1.csv")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df, "OT"


LOADERS = {"air_quality": _air_quality, "beijing_pm25": _beijing_pm25, "etth1": _etth1}
DATASETS = tuple(LOADERS)


@lru_cache(maxsize=None)
def frame(dataset):
    """(features DataFrame with the target as one column, target name), causally
    cleaned: forward-filled, with leading rows that still have gaps dropped."""
    df, target = LOADERS[dataset]()
    df = df.ffill().dropna()
    return df, target


@dataclass
class Splits:
    train: tuple          # (x, y) float32 arrays: x (n, seq_len, n_features), y (n,)
    val: tuple
    test: tuple
    persistence: dict     # {"val": mae, "test": mae} of predicting the last observed target
    n_features: int
    n_train_rows: int     # rows in the (possibly truncated) train segment


def _windows(values, target_idx, seq_len):
    """x[i] = rows i..i+seq_len-1, y[i] = target of row i+seq_len."""
    n = len(values) - seq_len
    if n <= 0:
        raise ValueError(f"segment of {len(values)} rows is too short for seq_len={seq_len}")
    view = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)[:n]
    x = np.ascontiguousarray(view.transpose(0, 2, 1))
    return x, values[seq_len:seq_len + n, target_idx], values[seq_len - 1:seq_len - 1 + n, target_idx]


@lru_cache(maxsize=None)
def load(dataset, seq_len, train_frac=1.0, n_cap=None):
    df, target = frame(dataset)
    values = df.to_numpy(dtype=np.float64)
    target_idx = list(df.columns).index(target)
    n = len(values)
    a, b = round(SPLITS[0] * n), round((SPLITS[0] + SPLITS[1]) * n)
    train_seg, val_seg, test_seg = values[:a], values[a:b], values[b:]

    mean, std = train_seg.mean(axis=0), train_seg.std(axis=0) + 1e-8
    train_seg, val_seg, test_seg = ((s - mean) / std for s in (train_seg, val_seg, test_seg))

    keep = max(seq_len + 2, int(round(train_frac * len(train_seg))))
    train_seg = train_seg[-keep:]
    if n_cap is not None:   # tiny runs (smoke tests)
        train_seg, val_seg, test_seg = (s[-(n_cap + seq_len):] for s in (train_seg, val_seg, test_seg))

    out, persistence = {}, {}
    for name, seg in (("train", train_seg), ("val", val_seg), ("test", test_seg)):
        x, y, last = _windows(seg, target_idx, seq_len)
        out[name] = (x.astype(np.float32), y.astype(np.float32))
        persistence[name] = float(np.abs(y - last).mean())
    return Splits(out["train"], out["val"], out["test"], persistence, values.shape[1], len(train_seg))


if __name__ == "__main__":
    for name in DATASETS:
        df, target = frame(name)
        s = load(name, 14)
        print(f"{name:13s} {len(df):6d} rows, {df.shape[1]} channels, target {target}; "
              f"windows train/val/test {len(s.train[1])}/{len(s.val[1])}/{len(s.test[1])}; "
              f"persistence MAE val {s.persistence['val']:.3f} test {s.persistence['test']:.3f}")
