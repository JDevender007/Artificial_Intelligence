"""Data-drift checks: PSI and KS per feature (dependency-free), plus an optional Evidently HTML report.

The alert rule is ours, not Evidently's, so it is unit-tested and stable across Evidently API changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index on reference-quantile bins. < 0.1 stable, 0.1-0.25 moderate, > 0.25 shift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference column
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str], psi_alert: float = 0.25,
                ks_alpha: float = 0.01) -> pd.DataFrame:
    """One row per column with PSI, KS statistic and p-value, and whether the column drifted.

    A column drifts when PSI > ``psi_alert`` *and* KS rejects at ``ks_alpha``: KS alone flags tiny, harmless shifts
    on large samples, PSI alone is noisy on small ones.
    """
    rows = []
    for c in columns:
        ks = ks_2samp(ref[c], cur[c])
        p = psi(ref[c].to_numpy(), cur[c].to_numpy())
        rows.append({"column": c, "psi": p, "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
                     "drifted": bool(p > psi_alert and ks.pvalue < ks_alpha)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def drift_verdict(table: pd.DataFrame, share_alert: float = 0.2) -> dict[str, Any]:
    """Dataset-level verdict: alert when the share of drifted columns exceeds ``share_alert``."""
    share = float(table["drifted"].mean())
    return {"n_columns": len(table), "n_drifted": int(table["drifted"].sum()), "share_drifted": share,
            "alert": share > share_alert, "top": table.head(5)[["column", "psi", "ks_p"]].to_dict("records")}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, html_path: str | Path) -> dict[str, Any] | None:
    """Evidently ``DataDriftPreset`` report as HTML. Returns its summary dict, or None if Evidently is absent."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        log.warning("evidently not installed; skipping HTML report")
        return None
    snap = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(html_path))
    return snap.dict()
