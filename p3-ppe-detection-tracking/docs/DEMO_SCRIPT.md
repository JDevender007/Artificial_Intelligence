# 60-second demo script (P3)

**0–10 s: the hook.** "Construction sites need to know who is missing a helmet or vest. This detects 12 PPE classes, decides compliance per worker, and tracks workers across frames. I wrote the evaluation code myself."

**10–25 s: the demo.** Run `python -m ppedet.app --weights reports/best_pretrained_n_s0.pt` and drop in a site photo. Show the boxes, then move the confidence slider: "Lower confidence catches more `no_helmet` boxes but adds false alarms. On a safety system a miss costs more than a false alarm."

**25–40 s: the evidence.** Open `reports/ppe/results.md`: mAP50-95 mean ± std over 3 seeds for scratch vs pretrained, no-aug, frozen backbone and the larger `s` model, plus ms/img. Then `error_analysis.md`: "Recall by box size shows where the model fails: small objects. The confusion table shows which classes it mixes up."

**40–55 s: compliance + tracking.** Scroll to the compliance table: "The same rule on ground truth and on predictions gives agreement of **[x]**." Then `tracking.md`: ByteTrack vs BoT-SORT MOTA and ID switches on camera-motion sequences, "scored with my own CLEAR-MOT implementation, which is unit-tested."

**55–60 s: the close.** "36 CPU tests in CI, one command per experiment, AGPL-licensed like Ultralytics, and built as a safety aid, not a surveillance tool."

_Before recording, replace [x] and check each spoken claim (small objects, top confusions, which tracker wins) against the real `reports/` files. Edit the line if the run says otherwise._
