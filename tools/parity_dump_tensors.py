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
    from transformers import AutoImageProcessor, PPDocLayoutV3ForObjectDetection
    from PIL import Image
    import torch

    model_dir = MODELS / "PP-DocLayoutV3"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir)
    model.eval()

    image = Image.open(FIXTURES / "layout_demo.jpg").convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]
    payload = {
        "detection_count": int(len(results["scores"])),
        "labels": [int(x) for x in results["labels"].tolist()],
        "first_score": float(results["scores"][0]) if len(results["scores"]) else 0.0,
    }
    path = GOLDENS / "layout_transformers_dump.json"
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
