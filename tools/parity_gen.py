#!/usr/bin/env python3
"""Regenerate parity golden JSON files using HuggingFace Transformers (dev-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"
FIXTURES = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"

VLM_TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def pixel_corners(tensor) -> dict:
    """Corner samples from NCHW pixel_values."""
    t = tensor.detach().cpu().float()
    _, _, h, w = t.shape
    return {
        "shape": list(t.shape),
        "top_left": float(t[0, 0, 0, 0]),
        "top_right": float(t[0, 0, 0, w - 1]),
        "bottom_left": float(t[0, 0, h - 1, 0]),
        "ch0_mean": float(t[0, 0].mean()),
        "ch0_std": float(t[0, 0].std()),
    }


def update_vlm() -> None:
    try:
        from transformers import AutoProcessor
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("pip install transformers torch pillow") from exc

    model_dir = MODELS / "PaddleOCR-VL-1.6"
    image_path = FIXTURES / "ocr_demo2.jpg"
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")

    gen_cfg_path = model_dir / "generation_config.json"
    generation_config = {}
    if gen_cfg_path.is_file():
        generation_config = json.loads(gen_cfg_path.read_text(encoding="utf-8"))

    task_goldens = {}
    for key, prompt in VLM_TASKS.items():
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        task_goldens[key] = {
            "prompt": prompt,
            "input_ids_len": len(ids),
            "input_ids": ids,
            "input_ids_head": ids[:10],
            "input_ids_sha256": hashlib.sha256(
                ",".join(map(str, ids)).encode()
            ).hexdigest(),
        }
        if key == "ocr":
            grid_key = (
                "image_grid_thw"
                if "image_grid_thw" in inputs
                else "grid_thw" if "grid_thw" in inputs else None
            )
            grid_thw = None
            if grid_key is not None:
                grid_thw = inputs[grid_key].detach().cpu().tolist()
            payload = {
                **task_goldens[key],
                "pixel_values": pixel_corners(inputs["pixel_values"]),
                "grid_thw": grid_thw,
                "generation_config": generation_config,
                "pixel_values_corner_atol": 0.003,
            }
            write_json(GOLDENS / "vlm_preprocess_ocr_demo2.json", payload)

    write_json(GOLDENS / "vlm_preprocess_tasks.json", task_goldens)


def update_layout(fixture_name: str, golden_name: str) -> None:
    try:
        from transformers import AutoImageProcessor, PPDocLayoutV3ForObjectDetection
        from PIL import Image
        import torch
    except ImportError as exc:
        raise SystemExit("pip install transformers torch pillow") from exc

    model_dir = MODELS / "PP-DocLayoutV3"
    image_path = FIXTURES / fixture_name
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    detections = []
    for i in range(len(results["scores"])):
        box = results["boxes"][i].tolist()
        detections.append(
            {
                "label": int(results["labels"][i]),
                "score": float(results["scores"][i]),
                "bbox": [float(x) for x in box],
            }
        )

    write_json(
        GOLDENS / golden_name,
        {
            "fixture": fixture_name,
            "detection_count": len(detections),
            "detections": detections,
            "labels": [d["label"] for d in detections],
            "score_atol": 0.02,
            "bbox_atol": 0.02,
            "first_score_min": float(results["scores"][0]) if len(results["scores"]) else 0.0,
            "preprocess": {
                "pixel_values": pixel_corners(inputs["pixel_values"]),
            },
        },
    )


def update_layout_all() -> None:
    update_layout("layout_demo.jpg", "layout_postprocess.json")
    update_layout("ocr_demo2.jpg", "layout_postprocess_ocr_demo2.json")


def update_pipeline_e2e() -> None:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise SystemExit('pip install "paddleocr[doc-parser]" for e2e goldens') from exc

    image_path = FIXTURES / "layout_demo.jpg"
    pipeline = PaddleOCRVL(pipeline_version="v1.6")
    output = pipeline.predict(str(image_path))
    blocks = []
    markdown_parts = []
    for res in output:
        data = res.json if hasattr(res, "json") else {}
        if isinstance(data, dict):
            for item in data.get("parsing_res_list", data.get("layout_det_res", [])) or []:
                if isinstance(item, dict):
                    blocks.append(
                        {
                            "label": item.get("block_label") or item.get("label"),
                            "content_len": len(str(item.get("block_content", ""))),
                        }
                    )
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            markdown_parts.append(str(md.get("markdown_texts", ""))[:200])
        elif isinstance(md, str):
            markdown_parts.append(md[:200])

    write_json(
        GOLDENS / "pipeline/layout_demo_official.json",
        {
            "block_count": len(blocks),
            "blocks": blocks[:30],
            "markdown_preview": markdown_parts[0] if markdown_parts else "",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-goldens", action="store_true")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    args = parser.parse_args()
    if not args.update_goldens:
        parser.print_help()
        return
    run_layout = args.layout or not (args.layout or args.vlm or args.e2e)
    run_vlm = args.vlm or not (args.layout or args.vlm or args.e2e)
    if run_vlm:
        update_vlm()
    if run_layout:
        update_layout_all()
    if args.e2e:
        update_pipeline_e2e()


if __name__ == "__main__":
    main()
