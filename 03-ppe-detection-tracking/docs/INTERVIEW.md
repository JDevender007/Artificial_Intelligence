# Interview Q&A (P3: PPE detection + tracking)

**1. Why is the official val split your test set, and why does it matter?**
Ultralytics picks `best.pt` by val fitness, and early stopping also watches val. If I reported on that same split, the number would be optimistic, because the checkpoint was selected on those images. So I carve my own val (15% of train, `split_seed=42`) for selection and touch the official val only once per run, as test. Every variant and seed shares the split, so the differences come from the model, not the data.

**2. mAP50-95 dropped but mAP50 stayed flat. What does that tell you?**
The model still finds the objects but puts the boxes in slightly the wrong place: localization got worse, not detection. My error analysis separates this out. A "localization" FP is the right class at IoU 0.1–0.5. If that count rises while background FPs stay flat, I'd look at box-regression capacity, input resolution (small helmets and goggles), or label noise in box edges.

**3. How does your compliance rule work, and how did you evaluate it without compliance labels?**
For each Person box: a `no_helmet` / `no_vest` box at least 50% inside the person is a violation. An item box at least 50% inside, with its centre in the right vertical band (helmet: top 35%; vest: 15–75%), is ok. Anything else is "unknown". I run the same rule on ground-truth boxes and on predicted boxes and compare per GT person, with "missed" when no predicted person matches. That gives an agreement rate and a confusion table. It measures the end-to-end effect of detection errors on the decision that actually matters, rather than mAP alone.

**4. How did you measure tracking without labelled video?**
I made sequences from test images by sliding a smooth pan/zoom crop window across them over 30 frames. The GT identities come for free (box *i* in the source image is identity *i*), and boxes less than 50% visible are dropped. I then wrote CLEAR-MOT myself: keep previous matches while IoU ≥ 0.5, Hungarian for the rest, count an IDSW when a GT id changes track id, MOTA = 1 − (FN + FP + IDSW)/GT. The limitation is that this tests camera motion and dropouts but not people crossing each other. That's why BoT-SORT's camera-motion compensation is the interesting comparison, and why I don't claim crowd-tracking performance.

**5. Ultralytics is AGPL-3.0. What does that mean for shipping this?**
AGPL is copyleft that also covers network use. If I serve a modified version to users, even as a web service, I have to offer them the source under AGPL. So this repo is AGPL-3.0 too. A company embedding it in a closed product would need an Ultralytics Enterprise licence, or would switch to a permissively licensed detector (for example RT-DETR implementations under Apache-2.0) and retrain. The dataset is CC BY 4.0, so it only needs attribution.
