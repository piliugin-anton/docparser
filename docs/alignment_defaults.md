# PaddleOCR-VL v1.6 alignment defaults

Source of truth: `PipelineConfig::default()` in `crates/docparser-pipeline/src/lib.rs`. Runtime values mirror [PaddleX `PaddleOCR-VL-1.6.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) and [PaddleOCR-VL usage](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md).

**Intentional deviation:** PaddleX sets `use_doc_preprocessor: false`. docparser enables document orientation + unwarping **by default** for skewed/curved/rotated inputs. Opt out with `USE_DOC_ORIENTATION_CLASSIFY=false` and/or `USE_DOC_UNWARPING=false`.

See also:

- [paddleocr_model_alignment.md](paddleocr_model_alignment.md) — model artifacts, threshold 0.3 vs HF 0.5, verification commands
- [layout_labels_and_models.md](layout_labels_and_models.md) — 25 layout labels and HF model map

| Parameter | docparser default | PaddleX YAML | Notes |
|-----------|-------------------|--------------|-------|
| `layout_threshold` | `0.3` | `0.3` | HF parity tests use **0.5** |
| `layout_unclip_ratio` | `1.0` | `[1.0, 1.0]` | Crop expansion = box size × ratio |
| `layout_nms` | `true` | `true` | IoU NMS after layout detection |
| `merge_layout_blocks` | `true` | `true` | |
| `layout_merge_bboxes_mode` | per-class (`merge_mode_for_label`) | per-class map | `large` for chart/formula/titles; `union` otherwise |
| `use_doc_orientation_classify` | **`true`** | `false` | PP-LCNet_x1_0_doc_ori |
| `use_doc_unwarping` | **`true`** | `false` | UVDoc |
| `use_chart_recognition` | `false` | `false` | |
| `use_seal_recognition` | `false` | `false` | |
| `use_ocr_for_image_block` | `false` | `false` | |
| `max_tokens` | `4096` | — | Capped by `generation_config.json` when present |

## `markdown_ignore_labels`

Matches PaddleX YAML (7 labels):

```text
number, footnote, header, header_image, footer, footer_image, aside_text
```

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
python tools/parity_gen.py --update-goldens --doc-prep
python tools/parity_gen.py --update-goldens --e2e  # requires paddleocr + GPU
```

Layout goldens use threshold **0.5** (HF Transformers). Runtime default is **0.3** (PaddleX).
