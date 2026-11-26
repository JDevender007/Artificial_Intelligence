"""Download, clean and split the UCI 'Default of Credit Card Clients' data (same cleaning as P1).

Source: Yeh, I.-C. (2009), UCI Machine Learning Repository id 350. License: CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credops.schema import BILL_AMT, ID_COL, PAY_AMT, PAY_STATUS, PROTECTED, TARGET

log = logging.getLogger(__name__)
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"


def sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, logged so the exact data version is traceable."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(raw_dir: str | Path) -> Path:
    """Download and extract the dataset into ``raw_dir`` unless the .xls is already there."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls = next(iter(raw_dir.rglob(XLS_NAME)), None)
    if xls is not None:
        return xls
    zip_path = raw_dir / "uci_credit_default.zip"
    if not zip_path.exists():
        log.info("downloading %s", UCI_URL)
        urllib.request.urlretrieve(UCI_URL, zip_path)  # noqa: S310 - fixed trusted URL
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (raw_dir / member).resolve().is_relative_to(raw_dir.resolve()):
                raise RuntimeError(f"unsafe path in zip: {member}")
        zf.extractall(raw_dir)
    xls = next(iter(raw_dir.rglob(XLS_NAME)), None)
    if xls is None:
        raise FileNotFoundError(f"{XLS_NAME} not found after extracting {zip_path}")
    return xls


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the .xls (row 0 holds X1..X23, row 1 the real header) and normalise column names."""
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    missing = {ID_COL, "LIMIT_BAL", TARGET, *PAY_STATUS, *BILL_AMT, *PAY_AMT, *PROTECTED} - set(df.columns)
    if missing:
        raise ValueError(f"unexpected schema, missing columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented EDUCATION/MARRIAGE codes into 'other', enforce int dtypes, reject duplicate IDs."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 4), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].between(1, 3), 3)
    df = df.astype("int64")
    if df[ID_COL].duplicated().any():
        raise ValueError("duplicate IDs in raw data")
    return df


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. Test is the 'production' stream for the drift simulation as well."""
    tr, te = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
