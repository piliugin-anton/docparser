# PaddleOCR-VL v1.6 alignment defaults

Source of truth for `PipelineConfig::official_v16()` in Rust. Values mirror
[PaddleOCR-VL pipeline usage](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
and HuggingFace artifact configs where applicable.

| Parameter | Official v1.6 | Minimal (legacy docparser) | Notes |
|-----------|---------------|----------------------------|-------|
| `layout_threshold` | `0.5` | `0.5` | Matches HF `post_process_object_detection` and `inference.yml` `draw_threshold` |
| `layout_unclip_ratio` | `1.0` | `0.02` | PaddleX expands crop boxes by box size × ratio; verify with `PaddleOCRVL()` if upgrading |
| `layout_nms` | `false` | `false` | Optional; enable when golden confirms IoU NMS in your PaddleOCR build |
| `merge_layout_blocks` | `true` | `false` | Cross-column merge default on in PaddleOCR-VL |
| `layout_merge_bboxes_mode` | `large` | `union` | Keep outer box when overlapping |
| `layout_detection_threshold` | `0.5` | `0.5` | Same as `layout_threshold` |
| `use_chart_recognition` | `false` | `false` | Official default off |
| `use_seal_recognition` | `false` | `false` | Official default off |
| `use_ocr_for_image_block` | `false` | `false` | Official default off |
| `use_doc_orientation_classify` | `false` | `false` | Stub returns error if enabled |
| `use_doc_unwarping` | `false` | `false` | Stub returns error if enabled |
| `max_tokens` | `4096` | `4096` | Capped by `generation_config.json` when present |

## `markdown_ignore_labels` (official-style)

```text
number, footnote, header, header_image, footer, footer_image, aside_text, formula_number
```

## Regenerating parity goldens

```bash
pip install transformers torch pillow
# optional e2e: pip install "paddleocr[doc-parser]"
python tools/parity_gen.py --update-goldens --layout --vlm
python tools/parity_gen.py --update-goldens --e2e  # requires paddleocr + GPU
```
