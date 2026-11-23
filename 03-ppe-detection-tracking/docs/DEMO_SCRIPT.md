# 60-second demo script (P3)

**0–10 s: the hook.** "Construction sites need to know who is missing a helmet or vest. This detects 12 PPE classes, decides compliance per worker, and tracks workers across frames. I wrote the evaluation code myself."

**10–25 s: the demo.** Run `python -m ppedet.app --weights best_pretrained_n_s0.pt` (the weights are a separate output file of the Kaggle kernel; they are not in git) and drop in a site photo. Show the boxes, then move the confidence slider: "Lower confidence catches more `no_helmet` boxes but adds false alarms. On a safety system a miss costs more than a false alarm."

**25–40 s: the evidence.** Open `reports/ppe/results.md`: mAP50-95 mean ± std over 3 seeds for scratch vs pretrained, no-aug, frozen backbone and the larger `s` model, plus ms/img. Then `error_analysis.md`: "Recall is 0.95 on large boxes but 0.17 on small ones. And the top confusions are no_boots read as boots and no_helmet as helmet, the dangerous direction."

**40–55 s: compliance + tracking.** Scroll to the compliance table: "The same rule on ground truth and on predictions gives 0.696 agreement, and only 14 of 227 non-compliant workers are called compliant. Most errors become 'unknown', which goes to a human." Then `tracking.md`: both trackers get 0.983 MOTA on persons; on all classes BoT-SORT recovers more boxes but has twice the ID switches, "scored with my own CLEAR-MOT implementation, which is unit-tested."

**55–60 s: the close.** "36 CPU tests in CI, one command per experiment, AGPL-licensed like Ultralytics, and built as a safety aid, not a surveillance tool."
