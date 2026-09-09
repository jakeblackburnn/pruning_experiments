"""The UCI Air Quality dataset — hourly gas-sensor and weather readings from
a polluted street in an Italian city, March 2004 - February 2005 (De Vito et
al. 2008). One of the four datasets in Vos et al. 2025 ("high complexity",
9446 records): 13 sensor/weather channels at 1-hour resolution, with a
seasonal + diurnal structure that gives a sequence model something real to
predict, unlike a nearly-i.i.d. series.

Downloaded straight from UCI's static archive (no key, no anti-bot
challenge — unlike the paper's other three sources, which sit behind
Kaggle logins or a JS proof-of-work wall on the download CDN we tried).

Task: given the last `seq_len` hours of all 12 usable channels, predict the
next hour's CO(GT) concentration (mg/m^3) — a standard framing for this
dataset. `n_rows` (tail-truncate before windowing) is the dataset-size knob
for the bitter-lesson grid.
"""

import functools
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "AirQualityUCI.csv"
URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

# NMHC(GT) is -200 (missing) for ~90% of rows past the first two months —
# unusable as a feature. Date/Time become the index. Everything else is a
# feature; CO(GT) doubles as the forecast target.
DROP_COLUMNS = ["NMHC(GT)"]
TARGET_COLUMN = "CO(GT)"
MISSING_SENTINEL = -200.0


def download():
    """Fetch and cache the raw CSV. No-op if already on disk."""
    if CSV_PATH.exists():
        return CSV_PATH
    DATA_DIR.mkdir(exist_ok=True)
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open("AirQualityUCI.csv") as f, open(CSV_PATH, "wb") as out:
            out.write(f.read())
    print(f"saved {CSV_PATH}")
    return CSV_PATH


@functools.lru_cache(maxsize=1)
def load_frame():
    """The cleaned, hourly-indexed DataFrame: missing sentinels interpolated,
    empty trailing columns and NMHC(GT) dropped, sorted by time. Cached —
    both experiment grids call this once per (seq_len, n_rows) combination."""
    path = download()
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])
    df = df.dropna(how="all")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                      format="%d/%m/%Y %H.%M.%S")
    df = df.drop(columns=["Date", "Time"] + DROP_COLUMNS).set_index("timestamp").sort_index()
    df = df.replace(MISSING_SENTINEL, np.nan)
    df = df.interpolate(method="linear", limit_direction="both")
    return df


def feature_columns(df):
    return [c for c in df.columns if c != TARGET_COLUMN]


def make_windows(df, seq_len, n_rows=None):
    """Sliding windows: X[i] = the seq_len rows before i (all features),
    y[i] = TARGET_COLUMN at row i. n_rows keeps only the most recent n_rows
    of the frame before windowing (the dataset-size knob)."""
    if n_rows is not None:
        df = df.tail(n_rows)
    cols = feature_columns(df)
    features = df[cols].to_numpy(dtype=np.float32)
    target = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    n = len(df) - seq_len
    if n <= 0:
        raise ValueError(f"n_rows={n_rows} too small for seq_len={seq_len}")
    x = np.stack([features[i:i + seq_len] for i in range(n)])
    y = target[seq_len:seq_len + n]
    return x, y


def split_train_test(x, y, test_frac=0.2):
    """Chronological split — the last test_frac of windows held out."""
    n_test = max(1, int(len(x) * test_frac))
    return (x[:-n_test], y[:-n_test]), (x[-n_test:], y[-n_test:])


class Standardizer:
    """Per-feature z-score, fit on train only. MAE is reported in these
    standardized units throughout (see README) so that numbers are
    comparable across dataset-size and sequence-length settings, where the
    raw CO(GT) scale doesn't change but the train-set statistics used to fit
    the standardizer do."""

    def __init__(self, x):
        axes = tuple(range(x.ndim - 1))  # everything but the feature axis
        self.mean = x.mean(axis=axes, keepdims=True)
        self.std = x.std(axis=axes, keepdims=True) + 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std


def load(seq_len, n_rows=None, test_frac=0.2):
    """(x_train, y_train, x_test, y_test) as float32 arrays, features and
    target both standardized on train statistics. Shapes: x is
    (n, seq_len, n_features), y is (n,)."""
    df = load_frame()
    x, y = make_windows(df, seq_len, n_rows=n_rows)
    (x_tr, y_tr), (x_te, y_te) = split_train_test(x, y, test_frac=test_frac)

    xs = Standardizer(x_tr)
    ys = Standardizer(y_tr[:, None])
    x_tr, x_te = xs.transform(x_tr), xs.transform(x_te)
    y_tr = ys.transform(y_tr[:, None])[:, 0]
    y_te = ys.transform(y_te[:, None])[:, 0]
    return x_tr, y_tr, x_te, y_te


def n_features():
    return len(feature_columns(load_frame()))


if __name__ == "__main__":
    df = load_frame()
    print(f"{len(df)} hourly rows, {df.index.min()} .. {df.index.max()}")
    print(f"features: {feature_columns(df)}")
    x_tr, y_tr, x_te, y_te = load(seq_len=30)
    print(f"seq_len=30: train {x_tr.shape}, test {x_te.shape}")
