# docparser

Pure-Rust document parser pipeline using [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) and [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) HuggingFace artifacts.

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
└── PP-DocLayoutV3/       # HF safetensors + official Paddle ONNX (`inference.onnx`)

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

## Workspace crates

| Crate | Role |
|-------|------|
| `docparser-download` | Parallel HF + fixture downloader |
| `docparser-candle-utils` | Shared safetensors mmap / parity helpers |
| `paddleocr-vl` | In-tree PaddleOCR-VL Candle inference |
| `pp-doclayout-v3` | PP-DocLayoutV3 layout (ONNX via ORT) |
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
python tools/parity_gen.py --update-goldens
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BIND_ADDR` | `0.0.0.0:8080` | HTTP listen address |
| `MODELS_DIR` | `models` | Model artifacts root |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `MAX_TOKENS` | `4096` | VLM decode limit |
| `HF_TOKEN` | — | Optional HuggingFace auth |
| `RUN_SLOW` | — | Enable ignored parity tests |

## Notes

- First request loads ~1.9 GB VLM weights (mmap) plus layout ONNX; allow 1–3 minutes on CPU before the first `/v1/parse` completes.
- Per-page latency depends on region count (each layout region runs a VLM decode).
- Layout inference uses [PaddlePaddle/PP-DocLayoutV3_onnx](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx) (`inference.onnx`) with HF-matching preprocess (800×800, `/255` only).
- VLM uses vendored Candle modules under `paddleocr-vl/src/paddleocr_vl/` (from `candle-transformers` 0.10).
- Optional: `cargo run -p docparser-download -- --include-reference` fetches HF `modeling_*.py` for porting.
