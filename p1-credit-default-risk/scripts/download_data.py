"""Download the UCI dataset and print basic, verifiable statistics."""

from __future__ import annotations

import argparse

from credit_default.data import TARGET, clean, ensure_data, load_raw, sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    xls = ensure_data(args.raw_dir)
    df = clean(load_raw(xls))
    print(f"file:          {xls}")
    print(f"sha256:        {sha256(xls)}")
    print(f"rows x cols:   {df.shape}")
    print(f"positive rate: {df[TARGET].mean():.4f} ({int(df[TARGET].sum())} defaults)")
    print(f"missing cells: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
