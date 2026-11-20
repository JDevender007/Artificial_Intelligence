"""Collect the CNN rows of the main run and every ablation into one mean ± std table.

    python scripts/compare_ablations.py --main outputs/cifar10 --out outputs/ablations.md
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from dlfund.evaluate import aggregate, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="outputs/cifar10")
    ap.add_argument("--glob", default="outputs/abl_*")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--out", default="outputs/ablations.md")
    args = ap.parse_args()

    frames = []
    for d in [Path(args.main), *map(Path, sorted(glob.glob(args.glob)))]:
        f = d / "results_per_seed.csv"
        if not f.exists():
            print(f"skip {d} (no results)")
            continue
        df = pd.read_csv(f)
        df = df[df["model"] == args.model].copy()
        df["model"] = "full config" if d == Path(args.main) else d.name.removeprefix("abl_")
        frames.append(df)
    if not frames:
        raise SystemExit("No results found")
    agg = aggregate(pd.concat(frames))
    md = f"# Ablations ({args.model})\n\n" + to_markdown(agg, "full config") + "\n"
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
