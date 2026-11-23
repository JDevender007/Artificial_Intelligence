"""Track PPE in a real video and write an annotated mp4 + per-track compliance CSV.

No ground truth exists for arbitrary videos, so this reports no accuracy metric; the
CSV gives each person track's majority-vote helmet/vest status over its lifetime.

    python scripts/track_video.py --weights best.pt --video site.mp4 --out outputs/video
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ppedet.compliance import assess


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/video")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = {int(k): v for k, v in model.names.items()}
    person = next(k for k, v in names.items() if v == "Person")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    votes: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    frames = 0
    for r in model.track(a.video, stream=True, persist=True, tracker=a.tracker, conf=a.conf, verbose=False):
        frames += 1
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        b = r.boxes
        if b is None or b.id is None:
            continue
        boxes, cls, ids = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.id.cpu().numpy().astype(int)
        people = assess(boxes, cls, names)
        pids = ids[cls == person]
        for st, tid in zip(people, pids, strict=True):
            for item, s in st.items.items():
                votes[int(tid)][item][s] += 1
    cap.release()
    if writer is not None:
        writer.release()
    with (out / "tracks.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "frames_seen", "helmet", "vest"])
        for tid, v in sorted(votes.items()):
            seen = int(np.max([sum(c.values()) for c in v.values()]))
            wr.writerow([tid, seen, *(v[i].most_common(1)[0][0] if v[i] else "unknown" for i in ("helmet", "vest"))])
    print(f"{frames} frames, {len(votes)} person tracks -> {out}")


if __name__ == "__main__":
    main()
