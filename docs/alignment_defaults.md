# PaddleOCR-VL v1.6 alignment defaults

Source of truth for `PipelineConfig::official_v16()` in `crates/docparser-pipeline/src/lib.rs`. Runtime values mirror [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) and [PaddleOCR-VL usage](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md).

See also:

- [paddleocr_model_alignment.md](paddleocr_model_alignment.md) — model artifacts, threshold 0.3 vs HF 0.5, verification commands
- [layout_labels_and_models.md](layout_labels_and_models.md) — 25 layout labels and HF model map

| Parameter | Official v1.6 | Minimal (legacy docparser) | Notes |
|-----------|---------------|----------------------------|-------|
| `layout_threshold` | `0.3` | `0.5` | Official: PaddleX runtime; minimal/HF parity: `post_process_object_detection` at 0.5 |
| `layout_unclip_ratio` | `1.0` | `0.02` | Crop expansion = box size × ratio |
| `layout_nms` | `true` | `false` | IoU NMS after layout detection |
| `merge_layout_blocks` | `true` | `false` | |
| `layout_merge_bboxes_mode` | per-class (`official_v16_merge_mode_for_label`) | `union` | Official: `large` for chart/formula/titles; `union` otherwise |
| `layout_detection_threshold` | `0.3` | `0.5` | Same as `layout_threshold` |
| `use_chart_recognition` | `false` | `false` | |
| `use_seal_recognition` | `false` | `false` | |
| `use_ocr_for_image_block` | `false` | `false` | |
| `use_doc_orientation_classify` | `false` | `false` | Stub returns error if enabled |
| `use_doc_unwarping` | `false` | `false` | Stub returns error if enabled |
| `max_tokens` | `4096` | `4096` | Capped by `generation_config.json` when present |

## `markdown_ignore_labels`

**Official v1.6** (matches PaddleX YAML — 7 labels):

```text
number, footnote, header, header_image, footer, footer_image, aside_text
```

**Minimal** profile uses `default_markdown_ignore_labels()` which adds `formula_number`.

## Per-class merge (`large` vs `union`)

| Mode | Labels |
|------|--------|
| `large` | `chart`, `formula`, `display_formula`, `doc_title`, `inline_formula`, `paragraph_title` |
| `union` | all other classes |

## Regenerating parity goldens

```bash
pip install transformers torch pillow
# optional e2e: pip install "paddleocr[doc-parser]"
python tools/parity_gen.py --update-goldens --layout --vlm
python tools/parity_gen.py --update-goldens --e2e  # requires paddleocr + GPU
```

Layout goldens use threshold **0.5** (HF Transformers). Official API profile uses **0.3** (PaddleX).
