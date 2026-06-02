# docparser

Pure-Rust document parser pipeline using [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) and [PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) HuggingFace artifacts.

Two-stage flow: layout detection → region crop → VLM recognition → JSON + Markdown.

## Prerequisites

- Rust 1.75+
- ~2.1 GB disk for model weights
- 4 GB+ RAM recommended (CPU inference)

## Setup

Download models and parity fixtures in parallel:

```bash
cargo run -p docparser-download -- --models-dir models --fixtures-dir tests/fixtures --jobs 8
```

Verify artifacts:

```bash
cargo run -p docparser-download -- --verify-only
```

Expected layout:

```
models/
├── PaddleOCR-VL-1.6/     # HF VLM weights + tokenizer (Candle safetensors)
└── PP-DocLayoutV3/       # HF safetensors layout weights (`model.safetensors`)

tests/fixtures/
├── ocr_demo2.jpg
└── layout_demo.jpg
```

Copy environment defaults:

```bash
cp .env.example .env
```

## Run API

```bash
cargo run -p docparser-api
```

Endpoints:

- `GET /health` — service status
- `POST /v1/parse` — multipart field `file` (jpg/jpeg/png)

Example:

```bash
curl -s -F "file=@tests/fixtures/ocr_demo2.jpg" http://localhost:8080/v1/parse | jq .
```

## Pipeline profiles

| Profile | How to enable | Behavior |
|---------|---------------|----------|
| `minimal` | default | Legacy docparser settings (`layout_unclip_ratio=0.02`, no merge) |
| `official_v16` | `PIPELINE_PROFILE=official_v16` | PaddleOCR-VL v1.6-style orchestration (see [docs/alignment_defaults.md](docs/alignment_defaults.md)) |

```bash
PIPELINE_PROFILE=official_v16 cargo run -p docparser-api
```

## Workspace crates

| Crate | Role |
|-------|------|
| `docparser-download` | Parallel HF + fixture downloader |
| `docparser-candle-utils` | Shared safetensors mmap / parity helpers |
| `paddleocr-vl` | In-tree PaddleOCR-VL Candle inference |
| `pp-doclayout-v3` | PP-DocLayoutV3 layout (Candle + safetensors) |
| `docparser-pipeline` | Two-stage orchestration |
| `docparser-api` | Axum HTTP server |
| `docparser-test-utils` | Parity test helpers |

## Tests

Fast checks (no weights):

```bash
cargo test --workspace
```

Slow parity tests (requires downloaded models):

```bash
RUN_SLOW=1 cargo test --workspace -- --ignored
```

Regenerate golden files (optional, Python dev harness):

```bash
pip install transformers torch pillow
python tools/parity_gen.py --update-goldens --layout --vlm
# optional end-to-end reference (paddleocr + GPU):
# pip install "paddleocr[doc-parser]"
# python tools/parity_gen.py --update-goldens --e2e
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BIND_ADDR` | `0.0.0.0:8080` | HTTP listen address |
| `MODELS_DIR` | `models` | Model artifacts root |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `MAX_TOKENS` | `4096` | VLM decode limit |
| `PIPELINE_PROFILE` | `minimal` | `official_v16` for aligned orchestration |
| `LAYOUT_UNCLIP_RATIO` | profile default | Crop expansion ratio |
| `LAYOUT_THRESHOLD` | `0.5` | Layout score threshold |
| `LAYOUT_NMS` | `false` | Enable layout NMS |
| `MERGE_LAYOUT_BLOCKS` | profile default | Merge overlapping layout boxes |
| `HF_TOKEN` | — | Optional HuggingFace auth |
| `RUN_SLOW` | — | Enable ignored parity tests |

## Alignment with official Paddle

- **Tensor parity:** HF Transformers + safetensors weights (`parity_gen.py` goldens under `tests/goldens/`).
- **Layout:** `preprocessor_config.json`-driven resize/normalize; labels from `config.json` `id2label`.
- **VLM:** Greedy decode; `generation_config.json` caps `max_new_tokens`; manual prompt layout matches HF `AutoProcessor` length (see slow tests).
- **Orchestration:** `PipelineConfig::official_v16()` — unclip, merge, markdown ignore labels per [docs/alignment_defaults.md](docs/alignment_defaults.md).
- **Not implemented:** doc orientation/unwarping (stubs error if enabled), polygon masks, Paddle `.pdiparams` runtime.

## Notes

- First request loads ~1.9 GB VLM weights (mmap) plus ~130 MB layout safetensors; allow 1–3 minutes on CPU before the first `/v1/parse` completes.
- Per-page latency depends on region count (each layout region runs a VLM decode).
- Optional: `cargo run -p docparser-download -- --include-reference` fetches HF `modeling_*.py` for porting.
