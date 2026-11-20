"""Download, load, clean and split the UCI 'Default of Credit Card Clients' dataset.

Source: Yeh, I.-C. (2009). UCI Machine Learning Repository, dataset id 350.
License: CC BY 4.0 (attribution required -- see README).
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"

TARGET = "default"
ID_COL = "ID"
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]  # repayment status, 1 = Sep 2005 ... 6 = Apr 2005
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file (logged so the exact data version is traceable)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(raw_dir: str | Path) -> Path:
    """Download and extract the dataset into ``raw_dir`` if missing; return the .xls path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls = next(iter(raw_dir.rglob(XLS_NAME)), None)
    if xls is not None:
        return xls
    zip_path = raw_dir / "uci_credit_default.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_URL, zip_path)  # noqa: S310 - fixed trusted URL
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (raw_dir / member).resolve()
            if not str(target).startswith(str(raw_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member}")
        zf.extractall(raw_dir)
    xls = next(iter(raw_dir.rglob(XLS_NAME)), None)
    if xls is None:
        raise FileNotFoundError(f"{XLS_NAME} not found after extracting {zip_path}")
    return xls


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the .xls (row 0 holds X1..X23 labels, row 1 the real header) and normalise names."""
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    expected = {ID_COL, "LIMIT_BAL", TARGET, *PAY_STATUS, *BILL_AMT, *PAY_AMT, *PROTECTED}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into 'other' and enforce integer dtypes.

    EDUCATION documents 1-4; observed 0, 5, 6 are mapped to 4 (others).
    MARRIAGE documents 1-3; observed 0 is mapped to 3 (others).
    PAY_k values -2 and 0 are undocumented but kept: they carry signal
    (-2 is commonly read as "no consumption", 0 as "revolving credit").
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 4), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].between(1, 3), 3)
    int_cols = [c for c in df.columns if c != TARGET]
    df[int_cols] = df[int_cols].astype("int64")
    df[TARGET] = df[TARGET].astype("int64")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs found in raw data")
    return df


def split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. The test set is touched exactly once per model."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into features (ID and target removed) and target."""
    return df.drop(columns=[ID_COL, TARGET]), df[TARGET]


def load_dataset(raw_dir: str | Path) -> pd.DataFrame:
    """Convenience: ensure, load and clean in one call."""
    return clean(load_raw(ensure_data(raw_dir)))
