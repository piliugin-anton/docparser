#!/usr/bin/env python3
"""Dump intermediate tensors for Rust parity (optional dev harness)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"
FIXTURES = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"


def dump_layout() -> None:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    onnx_path = MODELS / "PP-DocLayoutV3" / "inference.onnx"
    image = Image.open(FIXTURES / "layout_demo.jpg").convert("RGB")
    import cv2

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    resized = cv2.resize(img, (800, 800))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "im_shape": np.array([[800.0, 800.0]], np.float32),
            "image": blob[np.newaxis].astype(np.float32),
            "scale_factor": np.array([[800 / h, 800 / w]], np.float32),
        },
    )
    dets = out[0]
    payload = {
        "detection_count": int((dets[:, 1] > 0.5).sum()),
        "first_det": dets[0].tolist() if len(dets) else [],
    }
    path = GOLDENS / "layout_onnx_dump.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", action="store_true")
    args = parser.parse_args()
    if args.layout:
        dump_layout()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
