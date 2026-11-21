"""Gradio demo: upload a site photo -> boxes + per-person helmet/vest compliance table.

Run: ``python -m ppedet.app --weights outputs/ppe/runs/pretrained_n_s0/weights/best.pt``
"""

from __future__ import annotations

import argparse

import numpy as np

from ppedet.compliance import assess

DISCLAIMER = ("Research demo. Detections can be wrong, and an 'unknown' person means the model could not tell. "
              "Do not use this to discipline workers; a human must review every alert.")


def build_demo(weights: str, conf: float = 0.25):
    import gradio as gr
    from ultralytics import YOLO

    model = YOLO(weights)
    names = {int(k): v for k, v in model.names.items()}

    def infer(image: np.ndarray, threshold: float):
        # Gradio gives RGB; Ultralytics treats numpy input as BGR (OpenCV convention)
        r = model.predict(np.ascontiguousarray(image[..., ::-1]), conf=threshold, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        people = assess(boxes, cls, names)
        table = [[p.person_index, p.items.get("helmet"), p.items.get("vest"), p.overall] for p in people]
        return r.plot()[..., ::-1], table  # plot() returns BGR

    return gr.Interface(
        fn=infer,
        inputs=[gr.Image(type="numpy", label="Site photo"), gr.Slider(0.05, 0.9, value=conf, label="confidence")],
        outputs=[gr.Image(label="Detections"),
                 gr.Dataframe(headers=["person", "helmet", "vest", "overall"], label="Compliance")],
        title="PPE detection and compliance (YOLO26n, Construction-PPE)",
        description=DISCLAIMER,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--share", action="store_true", help="public gradio.live link (72 h)")
    a = ap.parse_args()
    build_demo(a.weights, a.conf).launch(share=a.share)


if __name__ == "__main__":
    main()
