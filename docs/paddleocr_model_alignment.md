# PaddleOCR-VL-1.6 model and orchestration alignment

This document records how **docparser** maps to the official [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) **PaddleOCR-VL-1.6** pipeline ([usage guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)), as configured in [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml).

Rust source of truth: `PipelineConfig::default()` in `crates/docparser-pipeline/src/lib.rs`.

## Downloaded artifacts

| Role | Hugging Face repo | On-disk path |
|------|-------------------|--------------|
| Layout | [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) | `models/PP-DocLayoutV3/` |
| VLM | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `models/PaddleOCR-VL-1.6/` |
| Doc orientation | [PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors) | `models/PP-LCNet_x1_0_doc_ori/` |
| Doc unwarping | [PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors) | `models/UVDoc/` |

PaddleX names the VLM `PaddleOCR-VL-1.6-0.9B`; the HF repo is the same 0.9B model family.

docparser uses the **PaddleOCR-VL** path (doc prep + layout + single VLM), not **PP-StructureV3** (PP-OCRv5 + SLANeXt + PP-FormulaNet + …). See [layout_labels_and_models.md](layout_labels_and_models.md) for per-label HF models on that alternate pipeline.

## Pipeline shape (aligned)

```text
image → PP-LCNet doc ori (rotate) → UVDoc (rectify) → PP-DocLayoutV3 → layout postprocess (NMS, merge bboxes, unclip) → filter_overlap_boxes → crop → merge_layout_blocks (text groups) → PaddleOCR-VL-1.6 per region → JSON/Markdown
```

**Note:** PaddleX YAML sets `use_doc_preprocessor: false` at the pipeline level. docparser runs orientation + unwarping **by default** (opt-out via env). See [alignment_defaults.md](alignment_defaults.md).

## Orchestration parameters

| Parameter | PaddleX YAML | docparser default | Notes |
|-----------|--------------|-------------------|-------|
| `layout_threshold` | `0.3` | `0.3` | HF parity tests use **0.5** (see below) |
| `layout_nms` | `true` | `true` | IoU NMS after detection |
| `layout_unclip_ratio` | `[1.0, 1.0]` | `1.0` | Box scale in layout postprocess (`1.0` = unchanged) |
| `merge_layout_blocks` | `true` | `true` | Text crop merge (`merge_blocks`), not bbox merge |
| `crop_padding_ratio` | — | `0.0` | Optional extra crop padding (local only) |
| `layout_merge_bboxes_mode` | per-class map | per-class via `merge_mode_for_label` | Containment merge in layout postprocess |
| `markdown_ignore_labels` | 7 labels | 7 labels (`official_markdown_ignore_labels`) | |
| `use_chart_recognition` | `false` | `false` | |
| `use_seal_recognition` | `false` | `false` | |
| `use_ocr_for_image_block` | (default off) | `false` | |
| `use_doc_orientation_classify` | `false` | **`true`** | Intentional default-on |
| `use_doc_unwarping` | `false` | **`true`** | Intentional default-on |

### Layout score threshold: 0.3 vs 0.5

| Context | Threshold | Why |
|---------|-----------|-----|
| docparser API/runtime | **0.3** | [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) |
| `LayoutModel::from_dir()` default | **0.5** | Hugging Face `post_process_object_detection` |
| `tools/parity_gen.py` layout goldens | **0.5** | Transformers parity |
| `pp-doclayout-v3` postprocess parity tests | **0.5** | Weight/tensor parity vs HF |

Lower threshold → more regions → more VLM calls (expected for official behavior).

## Per-class merge modes

From PaddleX YAML, using label names in our `config.json` (`formula` = Paddle `display_formula`):

| Mode | Labels |
|------|--------|
| `large` | `chart`, `formula`, `display_formula`, `doc_title`, `inline_formula`, `paragraph_title` |
| `union` | all other layout classes |

Implemented in `merge_mode_for_label()` / `apply_layout_merge_bboxes()` in `crates/docparser-pipeline/src/layout_merge.rs` (layout postprocess). `filter_overlap_boxes` in `layout_filter.rs`; text grouping in `block_merge.rs`.

## Not implemented (by design)

- PP-StructureV3 specialist models (table cell det, PP-FormulaNet, PP-Chart2Table, seal det/rec, etc.)
- Polygon masks from layout; Paddle `.pdiparams` inference
- `layout_shape_mode` / irregular polygon boxes

## Re-verification

```bash
# Fast Rust tests (merge, NMS, config)
cargo test -p docparser-pipeline

# Doc prep inference smoke (needs models; RUN_SLOW)
cargo test -p docparser-doc-prep --test load_inference -- --ignored

# HF layout/VLM tensor parity (needs models; often RUN_SLOW)
cargo test -p pp-doclayout-v3 --test postprocess_parity -- --ignored
cargo test -p paddleocr-vl --test preprocess_parity -- --ignored
cargo test -p paddleocr-vl --test generate_parity -- --ignored

# Regenerate goldens (Python)
pip install transformers torch pillow
python tools/parity_gen.py --update-goldens --layout --vlm
python tools/parity_gen.py --update-goldens --doc-prep
# optional e2e vs paddleocr:
# pip install "paddleocr[doc-parser]"
# python tools/parity_gen.py --update-goldens --e2e
```

## See also

- [alignment_defaults.md](alignment_defaults.md) — parameter table
- [layout_labels_and_models.md](layout_labels_and_models.md) — all 25 layout labels and HF model map
