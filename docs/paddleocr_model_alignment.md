# PaddleOCR-VL-1.6 model and orchestration alignment

This document records how **docparser** maps to the official [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) **PaddleOCR-VL-1.6** pipeline ([usage guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)), as configured in [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml).

Rust source of truth for `PIPELINE_PROFILE=official_v16`: `PipelineConfig::official_v16()` in `crates/docparser-pipeline/src/lib.rs`.

## Downloaded artifacts

| Role | Hugging Face repo | On-disk path |
|------|-------------------|--------------|
| Layout | [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) | `models/PP-DocLayoutV3/` |
| VLM | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `models/PaddleOCR-VL-1.6/` |

PaddleX names the VLM `PaddleOCR-VL-1.6-0.9B`; the HF repo is the same 0.9B model family.

docparser uses the **PaddleOCR-VL** path (layout + single VLM), not **PP-StructureV3** (PP-OCRv5 + SLANeXt + PP-FormulaNet + …). See [layout_labels_and_models.md](layout_labels_and_models.md) for per-label HF models on that alternate pipeline.

## Pipeline shape (aligned)

```text
image → PP-DocLayoutV3 → (optional NMS) → merge_layout_blocks → crop → PaddleOCR-VL-1.6 per region → JSON/Markdown
```

## Orchestration parameters

| Parameter | PaddleX YAML | `official_v16()` | `minimal()` | Notes |
|-----------|--------------|------------------|-------------|-------|
| `layout_threshold` | `0.3` | `0.3` | `0.5` | Runtime / `paddleocr doc_parser`; HF parity tests use **0.5** (see below) |
| `layout_nms` | `true` | `true` | `false` | IoU NMS after detection |
| `layout_unclip_ratio` | `[1.0, 1.0]` | `1.0` | `0.02` | Crop padding = box size × ratio |
| `merge_layout_blocks` | `true` | `true` | `false` | |
| `layout_merge_bboxes_mode` | per-class map | per-class via `official_v16_merge_mode_for_label` | global `union` | See [alignment_defaults.md](alignment_defaults.md) |
| `markdown_ignore_labels` | 7 labels | 7 labels (`official_markdown_ignore_labels`) | 8 labels (+ `formula_number`) | |
| `use_chart_recognition` | `false` | `false` | `false` | |
| `use_seal_recognition` | `false` | `false` | `false` | |
| `use_ocr_for_image_block` | (default off) | `false` | `false` | |
| `use_doc_preprocessor` | `false` | stubs error if enabled | same | UVDoc / doc ori not implemented |

### Layout score threshold: 0.3 vs 0.5

| Context | Threshold | Why |
|---------|-----------|-----|
| `official_v16` API/runtime | **0.3** | [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) |
| `LayoutModel::from_dir()` default | **0.5** | Hugging Face `post_process_object_detection` |
| `tools/parity_gen.py` layout goldens | **0.5** | Transformers parity |
| `pp-doclayout-v3` postprocess parity tests | **0.5** | Weight/tensor parity vs HF |

Lower threshold → more regions → more VLM calls (expected for official behavior).

## Per-class merge modes (official v1.6)

From PaddleX YAML, using label names in our `config.json` (`formula` = Paddle `display_formula`):

| Mode | Labels |
|------|--------|
| `large` | `chart`, `formula`, `display_formula`, `doc_title`, `inline_formula`, `paragraph_title` |
| `union` | all other layout classes |

Implemented in `official_v16_merge_mode_for_label()` in `crates/docparser-pipeline/src/layout_merge.rs`.

## Not implemented (by design)

- PP-StructureV3 specialist models (table cell det, PP-FormulaNet, PP-Chart2Table, seal det/rec, etc.)
- Document orientation (`PP-LCNet_x1_0_doc_ori`) and unwarping (`UVDoc`) — enabled in PaddleX sub-pipeline YAML but `use_doc_preprocessor: false` at pipeline level
- Polygon masks from layout; Paddle `.pdiparams` inference
- `layout_shape_mode` / irregular polygon boxes

## Re-verification

```bash
# Fast Rust tests (merge, NMS, config)
cargo test -p docparser-pipeline

# HF layout/VLM tensor parity (needs models; often RUN_SLOW)
cargo test -p pp-doclayout-v3 --test postprocess_parity -- --ignored
cargo test -p paddleocr-vl --test preprocess_parity -- --ignored

# Regenerate goldens (Python)
pip install transformers torch pillow
python tools/parity_gen.py --update-goldens --layout --vlm
# optional e2e vs paddleocr:
# pip install "paddleocr[doc-parser]"
# python tools/parity_gen.py --update-goldens --e2e
```

## See also

- [alignment_defaults.md](alignment_defaults.md) — parameter table for profiles
- [layout_labels_and_models.md](layout_labels_and_models.md) — all 25 layout labels and HF model map
