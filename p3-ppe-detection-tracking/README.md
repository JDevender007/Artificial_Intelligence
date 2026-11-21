# PPE detection, compliance and tracking on construction sites

![CI](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p3-ci.yml/badge.svg)

This project detects 12 kinds of personal protective equipment, and the matching "missing PPE" labels, in construction-site photos. It does three things with them:

- a **controlled comparison**: from-scratch vs COCO-pretrained YOLO26, plus three ablations, 3 seeds each
- a **per-person compliance rule** (helmet and vest) built on top of the detections
- a **tracker comparison** (ByteTrack vs BoT-SORT), scored with CLEAR-MOT metrics that I implemented myself

The detection metrics, the error analysis, the compliance check and the MOT metrics are written in plain numpy/scipy and unit-tested. Ultralytics is used only to train and run the models.

## Problem and success criteria

| | |
|---|---|
| Task | object detection, 12 classes: `helmet, gloves, vest, boots, goggles, Person, no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest` |
| Primary metric | test **mAP50-95** (COCO-style, averaged over IoU 0.50:0.95) |
| Also reported | mAP50, precision, recall, per-class mAP, GPU inference ms/img, parameters |
| Downstream | per-person compliance agreement (the rule on predictions vs the rule on ground truth); MOTA, IDSW, MOTP for tracking |
| Baseline | `scratch_n`: the same YOLO26n architecture, randomly initialised, with identical training settings |
| Protocol | 3 seeds per variant, mean ± std; one fixed train/val split; the test set is evaluated once per run with the best-on-val weights |

## Data

| | |
|---|---|
| Dataset | [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/), loaded through Ultralytics' built-in `construction-ppe.yaml` (downloads automatically) |
| Licence | **CC BY 4.0**: attribution required; commercial use allowed |
| Split | The official `val` split is used as our **test** set. Our val set is 15% of the official train split (`split_seed=42`), so early stopping and `best.pt` selection never see test images. All seeds and variants share the same split. |
| Size (computed) | train **680** images / 10,100 boxes, val **120** / 1,802, test **200** / 2,961. There are no empty images, about 15 boxes per image, and small objects are 17% of test boxes (`no_helmet` 39% small). |
| Statistics | `outputs/ppe/data_summary.md` is **computed** by `ppedet.data.split_stats`. It gives images, empty images, boxes per image, and per-class instances and images with small/medium/large counts. None of these numbers are typed in by hand. |

Privacy: the images show real workers, often with visible faces. They are used here only for training and evaluation. The repo never commits dataset images; `datasets/` is git-ignored.

## Approach

```mermaid
flowchart LR
    A[construction-ppe.yaml<br/>CC BY 4.0] --> B[fixed split<br/>train / val 15% / test = official val]
    B --> S[data_summary.md<br/>computed stats]
    B --> C{5 variants x 3 seeds}
    C --> C1[scratch_n<br/>baseline]
    C --> C2[pretrained_n<br/>main]
    C --> C3[pretrained_n_noaug]
    C --> C4[pretrained_n_frozen<br/>freeze=10]
    C --> C5[pretrained_s]
    C1 & C2 & C3 & C4 & C5 --> D[early stop on val<br/>best.pt]
    D --> E[test once<br/>mAP, P, R, ms/img]
    E --> F[results.md mean ± std<br/>per_class.md]
    D -->|main, seed 0| G[predictions on test<br/>conf 0.25]
    G --> H[numpy error analysis<br/>recall by class/size, FP types]
    G --> I[compliance rule<br/>GT vs predicted]
    D -->|main, seed 0| J[synthetic camera-motion<br/>sequences from test images]
    J --> K[ByteTrack vs BoT-SORT<br/>own CLEAR-MOT]
```

| Variant | Question it answers |
|---|---|
| `scratch_n` (baseline) | How far does the small training split get you with no pretraining? |
| `pretrained_n` (main) | What does COCO transfer learning add? |
| `pretrained_n_noaug` | How much do mosaic, mixup, HSV, flip, scale and translate augmentation help on a small dataset? |
| `pretrained_n_frozen` | Does freezing the first 10 layers (the backbone) cost accuracy, and does it speed training up? |
| `pretrained_s` | Is the larger model worth roughly 3x the parameters and the extra latency? |

**Compliance rule** ([compliance.py](src/ppedet/compliance.py)). For each `Person` box and each item (helmet, vest):

- **violation** if a `no_<item>` box lies at least 50% inside the person box
- **ok** if an `<item>` box lies at least 50% inside the person box and its centre is in the expected vertical band (helmet: top 35% of the person; vest: 15–75%)
- **unknown** otherwise

A violation outranks an ok. The rule outputs "unknown" when the evidence is missing; it does not guess. The rule is scored by applying it to ground-truth boxes and to predicted boxes and comparing the result for each GT person.

**Tracking.** There are no identity-labelled videos, so each sequence is a test image viewed through a crop window that pans and zooms smoothly (30 frames). Boxes that are less than 50% visible are dropped. Ground-truth ids come from the source labels, which makes exact CLEAR-MOT scoring possible. This setup tests camera motion, missed detections and re-association. It does **not** test people walking past each other; see Limitations. `scripts/track_video.py` runs on real footage and writes a per-track compliance CSV (majority vote over the track's frames), with no accuracy claim.

Trade-offs, one recommendation each:

- **YOLO26 over YOLO11/YOLOv8.** It is Ultralytics' current release and NMS-free end to end; Ultralytics reports faster CPU and edge inference at similar accuracy. The latency column checks this on a T4.
- **Nano as the main model.** The `pretrained_s` row checks whether the extra cost pays off. For a site camera on edge hardware, latency matters as much as accuracy.
- **Official val as test, our own val from train.** Selecting `best.pt` on the same images you report on inflates results.
- **Own matching and MOT code** instead of only reading Ultralytics' numbers. Error types and identity switches can't be read off a single mAP, and the code is unit-tested against hand-computed cases.
- **ByteTrack as the default tracker.** It needs no re-ID network and is fast. BoT-SORT adds camera-motion compensation (GMC), which should matter here, and the comparison measures whether it does.

Recent API notes: YOLO26 weights are `yolo26n.pt` / `yolo26s.pt`, and `model.train(...)` returns metrics, with the best checkpoint at `model.trainer.best`. `model.val(split="test")` needs a `test:` key in the data YAML, which `ppe_split.yaml` provides. `model.track(..., persist=True)` keeps tracker state between calls, so a fresh `YOLO` object is created per sequence to reset ids.

## Repository structure

```
p3-ppe-detection-tracking/
├── configs/            ppe.yaml (5 variants x 3 seeds), smoke.yaml (1 epoch pipeline check)
├── src/ppedet/
│   ├── config.py       dataclass config + --set overrides + validation
│   ├── data.py         dataset location, fixed split, computed statistics
│   ├── boxes.py        IoU/IoA, size buckets, greedy class-aware matching + FP typing
│   ├── evaluate.py     error analysis, compliance agreement, mean ± std tables
│   ├── compliance.py   per-person helmet/vest rule
│   ├── sequences.py    synthetic pan/zoom sequences with exact GT ids
│   ├── mot.py          CLEAR-MOT (MOTA, MOTP, IDSW, FP, FN)
│   ├── train.py        Ultralytics train / test / predict wrapper (lazy import)
│   ├── track.py        tracker comparison
│   ├── experiment.py   resumable end-to-end runner
│   └── app.py          Gradio demo
├── scripts/            run_experiment.py, track_video.py, make_notebook.py
├── tests/              36 CPU tests, no ultralytics or torch needed
├── notebooks/          kaggle_run.ipynb (generated, thin wrapper)
├── reports/            committed snapshot of the final run
└── docs/               INTERVIEW.md, DEMO_SCRIPT.md, CAREER.md
```

## Setup and usage

```bash
python -m venv .venv && .venv\Scripts\activate           # or source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps
python scripts/run_experiment.py --config configs/smoke.yaml   # ~5 min pipeline check, numbers not reportable
python scripts/run_experiment.py --config configs/ppe.yaml     # full run, ~4-5 h on a T4
python -m ppedet.app --weights outputs/ppe/runs/pretrained_n_s0/weights/best.pt
python scripts/track_video.py --weights <best.pt> --video site.mp4
pytest && ruff check .
```

Any config value can be overridden, for example `--set train.epochs=10 "seeds=[0]"`. Runs are **resumable**: finished (variant, seed) rows in `results.jsonl` are skipped, an interrupted run resumes from `last.pt`, and the error-analysis and tracking stages are skipped once their outputs exist.

**Kaggle** (how the reported numbers were produced): open `notebooks/kaggle_run.ipynb` with a T4 GPU and Internet on, then Save & Run All. It writes `env.txt` (the exact ultralytics/torch versions, used to pin `requirements.txt`) and `p3_results.zip` (text results plus the main run's plots). The main model's weights are a separate output file. Unzip the archive into `outputs/` and copy it into `reports/`. The notebook is generated by `scripts/make_notebook.py`.

Reproducibility: the seed is passed to Ultralytics with `deterministic=True`, and the split is fixed. GPU kernels are still not bit-exact, so expect small seed-to-seed spread. That spread is why the tables report std.

## Results

> Pending the Kaggle run. Numbers are copied from `outputs/ppe/{results,per_class,error_analysis,tracking}.md` into `reports/`. Nothing is filled in by hand.

| variant | test mAP50-95 | mAP50 | precision | recall | ms/img (T4) | params (M) |
|---|---|---|---|---|---|---|
| scratch_n (baseline) | _ | _ | _ | _ | _ | _ |
| pretrained_n (main) | _ | _ | _ | _ | _ | _ |
| pretrained_n_noaug | _ | _ | _ | _ | _ | _ |
| pretrained_n_frozen | _ | _ | _ | _ | _ | _ |
| pretrained_s | _ | _ | _ | _ | _ | _ |

| tracker | scope | MOTA | IDSW | MOTP |
|---|---|---|---|---|
| ByteTrack | Person | _ | _ | _ |
| BoT-SORT | Person | _ | _ | _ |

## Error analysis

`error_analysis.md` covers the main model (seed 0, conf ≥ 0.25, IoU ≥ 0.5):

- recall per class and per box size
- false positives split into background, localization, class-confusion and duplicate errors
- the most frequent class confusions
- the compliance GT-vs-predicted table

The written interpretation will be added after the run.

## Limitations

- The test set is small (the official val split), so per-class numbers for rare classes have high variance. Read the per-class `GT` column before trusting a per-class recall.
- The synthetic sequences have rigid scenes and no person-to-person occlusion. They measure robustness to camera motion and missed detections, not crowd tracking, and real video will be harder.
- The compliance rule is a geometric heuristic. A helmet held at chest height, or two overlapping workers, can fool it. "Unknown" is common when items are small or occluded.
- From-scratch training gets the same epoch budget as fine-tuning, so the baseline gap partly reflects under-training. The comparison is "equal budget", not "scratch at convergence".

## Licences and ethics

- **Code licence: AGPL-3.0.** Ultralytics is AGPL-3.0, and this project imports and fine-tunes its models, so the project is released under the same terms. A closed-source product would need an Ultralytics Enterprise licence. The YOLO26 COCO weights fall under the same terms.
- **Data: CC BY 4.0.** Credit the Construction-PPE dataset (Ultralytics) whenever the data or derived models are shared.
- **Surveillance risk.** PPE detection is easily turned into worker monitoring. Use it as a safety aid with a human in the loop: alerts go to a site safety officer, not into automated discipline. Workers should be told that cameras are in use, and the system should keep aggregate statistics rather than per-person records.
- **Privacy.** Video contains faces. A real deployment should process on the device, keep no raw footage (or blur faces before storing anything), and follow local law (e.g. GDPR, or India's DPDP Act 2023).
- **Failure cost.** A missed `no_helmet` is worse than a false alarm. Consider the operating threshold with that asymmetry in mind, and the per-class recall table shows where the model is weakest.
