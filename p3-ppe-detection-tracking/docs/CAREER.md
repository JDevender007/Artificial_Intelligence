# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/p3-ppe-detection-tracking` (monorepo). Standalone name if split out: `ppe-compliance-yolo26-tracking`

## Resume bullets (fill in from `reports/ppe/*.md` once the run finishes)

- Fine-tuned YOLO26 for 12-class construction PPE detection, reaching **[mAP50-95 ± s]** test mAP50-95 vs **[scratch]** for a from-scratch baseline (3 seeds, 5 variants including augmentation and frozen-backbone ablations) at **[ms]** ms/img on a T4.
- Built a per-worker helmet/vest compliance rule and a numpy error analysis (recall by size, FP typing); evaluated ByteTrack vs BoT-SORT with my own unit-tested CLEAR-MOT implementation (MOTA **[a]** vs **[b]**). 36 pytest tests in GitHub Actions.

## LinkedIn post draft

> Project 3/11 of my AI-engineering portfolio: **PPE compliance from site cameras**.
>
> I fine-tuned YOLO26 on Construction-PPE (12 classes, CC BY 4.0) and ran a controlled comparison: 5 variants × 3 seeds.
> - from scratch: **[ ]** mAP50-95
> - COCO-pretrained: **[ ]**
> - without augmentation / frozen backbone / larger model: **[ ] / [ ] / [ ]**
>
> mAP alone doesn't say whether a worker is wearing a helmet, so I added:
> - a per-person compliance rule, scored as GT-rule vs prediction-rule agreement (**[ ]**)
> - ByteTrack vs BoT-SORT on camera-motion sequences, with CLEAR-MOT metrics I implemented and unit-tested myself
>
> Biggest lesson: **[from error_analysis.md]**.
>
> Ethics note: this should be a safety aid with a human in the loop, not worker surveillance.
>
> Code: https://github.com/JDevender007/Artificial_Intelligence/tree/main/p3-ppe-detection-tracking  #ComputerVision #YOLO #MLOps
