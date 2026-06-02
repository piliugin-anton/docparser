#!/usr/bin/env python3
"""Regenerate parity golden JSON files using HuggingFace Transformers (dev-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"
FIXTURES = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def update_preprocess() -> None:
    try:
        from transformers import AutoProcessor
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("pip install transformers torch pillow") from exc

    model_dir = MODELS / "PaddleOCR-VL-1.6"
    image_path = FIXTURES / "ocr_demo2.jpg"
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    write_json(
        GOLDENS / "vlm_preprocess_ocr_demo2.json",
        {
            "input_ids_len": int(inputs["input_ids"].shape[-1]),
            "input_ids_head": inputs["input_ids"][0, :10].tolist(),
            "prompt": "OCR:",
            "pixel_values_corner_atol": 0.003,
        },
    )


def update_layout_postprocess() -> None:
    try:
        from transformers import AutoImageProcessor, PPDocLayoutV3ForObjectDetection
        from PIL import Image
        import torch
    except ImportError as exc:
        raise SystemExit("pip install transformers torch pillow") from exc

    model_dir = MODELS / "PP-DocLayoutV3"
    image_path = FIXTURES / "layout_demo.jpg"
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    write_json(
        GOLDENS / "layout_postprocess.json",
        {
            "detection_count": len(results["scores"]),
            "labels": [int(x) for x in results["labels"].tolist()],
            "score_atol": 0.02,
            "bbox_atol": 0.02,
            "first_score_min": float(results["scores"][0]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-goldens", action="store_true")
    args = parser.parse_args()
    if not args.update_goldens:
        parser.print_help()
        return
    update_preprocess()
    update_layout_postprocess()


if __name__ == "__main__":
    main()
