# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/03-ppe-detection-tracking` (monorepo). Standalone name if split out: `ppe-compliance-yolo26-tracking`

## Resume bullets (numbers from `reports/ppe/*.md`)

- Fine-tuned YOLO26 for 12-class construction PPE detection, reaching **0.529 Â± 0.003** test mAP50-95 vs **0.227 Â± 0.008** for a from-scratch baseline (3 seeds, 5 variants including augmentation and frozen-backbone ablations) at **4.0 ms/img** on a T4.
- Built a per-worker helmet/vest compliance rule and a numpy error analysis (recall by size, FP typing); evaluated ByteTrack vs BoT-SORT with my own unit-tested CLEAR-MOT implementation (all-class MOTA **0.742** vs **0.759**, ID switches 7 vs 15). 36 pytest tests in GitHub Actions.

## LinkedIn post draft

> Project 3/11 of my AI-engineering portfolio: **PPE compliance from site cameras**.
>
> I fine-tuned YOLO26 on Construction-PPE (12 classes, CC BY 4.0) and ran a controlled comparison: 5 variants Ã— 3 seeds.
> - from scratch: **0.227** mAP50-95
> - COCO-pretrained: **0.529** (Ã—2.3, same 50-epoch budget)
> - without augmentation / frozen backbone / larger model: **0.471 / 0.514 / 0.595**
>
> mAP alone doesn't say whether a worker is wearing a helmet, so I added:
> - a per-person compliance rule, scored as GT-rule vs prediction-rule agreement (**0.696** over 481 workers)
> - ByteTrack vs BoT-SORT on camera-motion sequences, with CLEAR-MOT metrics I implemented and unit-tested myself
>
> Biggest lesson: **mAP hides where the model fails. Recall is 0.95 on large boxes and 0.17 on small ones, and the top confusions are `no_boots`â†’`boots` and `no_helmet`â†’`helmet`, which turn violations into "compliant".**
>
> Ethics note: this should be a safety aid with a human in the loop, not worker surveillance.
>
> Code: https://github.com/JDevender007/Artificial_Intelligence/tree/main/03-ppe-detection-tracking  #ComputerVision #YOLO #MLOps
