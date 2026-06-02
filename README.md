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
├── PaddleOCR-VL-1.6/     # HF VLM weights + tokenizer
└── PP-DocLayoutV3/       # HF layout safetensors + config

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
| `paddleocr-vl` | VLM Candle port (HF safetensors) |
| `pp-doclayout-v3` | Layout Candle port (HF safetensors) |
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

- First inference loads ~1.9 GB VLM weights; expect 30–120s startup on CPU once the Candle port is complete.
- Per-page latency depends on region count (each region runs VLM decode).
- Layout and VLM inference modules are scaffolded; `model.rs` in each crate is where the Candle ports land.
