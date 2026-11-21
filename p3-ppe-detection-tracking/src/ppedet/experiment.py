"""End-to-end experiment: split -> stats -> variants x seeds -> error analysis -> tracking.

Every stage writes its output to ``cfg.output_dir`` and is skipped on re-run if the
output already exists, so a Kaggle session that dies can just be run again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ppedet.config import ExperimentConfig
from ppedet.data import locate_dataset, read_split, split_stats, stats_markdown, write_split
from ppedet.evaluate import aggregate, compliance_agreement, error_analysis, error_markdown, mean_std_markdown
from ppedet.utils import get_logger, set_seed, write_json

log = get_logger(__name__)
METRICS = ["map50_95", "map50", "precision", "recall", "ms_per_img", "params_m"]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def prepare_data(cfg: ExperimentConfig) -> tuple[Path, dict[int, str]]:
    """Locate/download the dataset, write the fixed split and its computed statistics."""
    out = Path(cfg.output_dir)
    split_yaml = out / "split" / "ppe_split.yaml"
    if split_yaml.exists():
        names = yaml.safe_load(split_yaml.read_text(encoding="utf-8"))["names"]
        return split_yaml, {int(k): v for k, v in names.items()}
    info = locate_dataset(cfg.data.yaml)
    names = {int(k): v for k, v in info["names"].items()}
    split_yaml = write_split(info["train"], info["val"], names, out / "split",
                             cfg.data.val_fraction, cfg.data.split_seed)
    stats = {s: split_stats(read_split(out / "split", s), names) for s in ("train", "val", "test")}
    write_json(stats, out / "data_summary.json")
    (out / "data_summary.md").write_text(stats_markdown(stats), encoding="utf-8")
    return split_yaml, names


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    from ppedet.train import predict_records, run_name, test_one, train_one

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    split_yaml, names = prepare_data(cfg)

    rows_path = out / "results.jsonl"
    rows = _load_rows(rows_path)
    done = {(r["variant"], r["seed"]) for r in rows}
    for variant in cfg.variants:
        for seed in cfg.seeds:
            if (variant.name, seed) in done:
                log.info("skip %s seed %d (done)", variant.name, seed)
                continue
            set_seed(seed)
            log.info("=== %s seed %d ===", variant.name, seed)
            best = train_one(cfg, variant, seed, split_yaml)
            row = {"variant": variant.name, "seed": seed, "weights": str(best),
                   **test_one(cfg, best, split_yaml, run_name(variant, seed))}
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)

    agg = aggregate(rows, METRICS)
    agg.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(mean_std_markdown(agg, METRICS) + "\n", encoding="utf-8")
    _per_class_table(rows, out)

    main = next((r for r in rows if r["variant"] == cfg.eval.main_variant and r["seed"] == min(cfg.seeds)), None)
    summary: dict[str, Any] = {"results": agg.to_dict(orient="records")}
    if main is None:
        return summary
    test_images = read_split(out / "split", "test")
    rec_path = out / "predictions_test.json"
    if rec_path.exists():
        records = json.loads(rec_path.read_text(encoding="utf-8"))
    else:
        records = predict_records(main["weights"], test_images, cfg.eval.conf, cfg.train.imgsz)
        write_json(records, rec_path)
    ea = error_analysis(records, names, cfg.eval.iou)
    comp = compliance_agreement(records, names, cfg.eval.iou)
    write_json({"errors": ea, "compliance": comp}, out / "error_analysis.json")
    (out / "error_analysis.md").write_text(
        f"Variant `{main['variant']}` seed {main['seed']}, conf >= {cfg.eval.conf}, IoU >= {cfg.eval.iou}\n\n"
        + error_markdown(ea, comp), encoding="utf-8")
    summary.update(errors=ea, compliance=comp)

    if cfg.track.enabled and not (out / "tracking.json").exists():
        from ppedet.track import evaluate_trackers, tracking_markdown

        t = cfg.track
        trk = evaluate_trackers(main["weights"], test_images, t.trackers, names, t.n_sequences, t.n_frames,
                                t.min_objects, t.seed, cfg.eval.conf, cfg.train.imgsz)
        write_json(trk, out / "tracking.json")
        (out / "tracking.md").write_text(tracking_markdown(trk), encoding="utf-8")
        summary["tracking"] = trk
    return summary


def _per_class_table(rows: list[dict[str, Any]], out: Path) -> None:
    """Per-class test mAP50-95, mean over seeds, one column per variant."""
    import pandas as pd

    recs = [{"variant": r["variant"], "seed": r["seed"], "class": c, "map50_95": v}
            for r in rows for c, v in r.get("per_class_map50_95", {}).items()]
    if not recs:
        return
    piv = pd.DataFrame(recs).pivot_table(index="class", columns="variant", values="map50_95", aggfunc="mean",
                                         sort=False)
    head = ["class", *piv.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([str(c), *(f"{v:.3f}" for v in row)]) + " |" for c, row in piv.iterrows()]
    (out / "per_class.md").write_text("Test mAP50-95 per class (mean over seeds)\n\n" + "\n".join(lines) + "\n",
                                      encoding="utf-8")
