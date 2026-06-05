# docparser

Pure-Rust document parser pipeline using [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), [PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors), [PP-LCNet_x1_0_doc_ori_safetensors](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors), and [UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors) HuggingFace artifacts.

Flow: document orientation → geometric unwarping → layout detection → region crop → VLM recognition → JSON + Markdown.

## Prerequisites

- Rust 1.75+
- ~2.15 GB disk for model weights
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
├── PaddleOCR-VL-1.6/        # HF VLM weights + tokenizer (Candle safetensors)
├── PP-DocLayoutV3/          # HF safetensors layout weights
├── PP-LCNet_x1_0_doc_ori/   # Document orientation classifier (~7 MB)
└── UVDoc/                   # Document unwarping (~32 MB)

tests/fixtures/
├── ocr_demo2.jpg
├── layout_demo.jpg
├── doc_ori_demo.png
└── uvdoc_demo.jpeg
```

Copy environment defaults:

```bash
cp .env.example .env
```

## Run API

```bash
cargo run -p docparser-api --release
```

For production CPU inference, use a release build (workspace `profile.release`: thin LTO, `codegen-units = 1`). `.cargo/config.toml` enables `target-cpu=native` for the build host. Faster test links: `cargo test --profile release-fast --release`. Stronger LTO: `--profile release-lto`.

### Optional: Intel MKL (faster CPU matmul)

Candle can use [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html) for BLAS-heavy ops. Use a **system** MKL install (`MKLROOT`); the bundled MKL 2020.1 crate is too old and fails to link (`undefined reference to hgemm_`).

**1. Install oneAPI MKL** (Ubuntu/Debian example):

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install intel-oneapi-mkl-devel intel-oneapi-openmp
```

**2. Load oneAPI env before every build/run** (MKL alone does not add `libiomp5.so`):

```bash
source scripts/mkl-env.sh
# or manually:
# source /opt/intel/oneapi/mkl/latest/env/vars.sh
# source /opt/intel/oneapi/compiler/latest/env/vars.sh
# optional:
# export MKL_THREADING_LAYER=GNU
# export OMP_NUM_THREADS=8
```

**3. Build with the workspace `mkl` feature** (`MKLROOT` must be set so the linker uses system MKL, not the bundled 2020.1 libs):

```bash
source scripts/mkl-env.sh
cargo build -p docparser-api --release --features mkl
```

The API binary embeds rpath entries for `$MKLROOT/lib` and the oneAPI compiler lib when `MKLROOT` is set at **build** time, so you can usually run `./target/release/docparser-api` without sourcing env again. If you see `libiomp5.so: cannot open shared object file`, either rebuild after `source scripts/mkl-env.sh`, or source that script before launching the binary.

```bash
source scripts/mkl-env.sh
./target/release/docparser-api
# or:
source scripts/mkl-env.sh && cargo run -p docparser-api --release --features mkl
```

The feature is forwarded through `docparser-pipeline` → `paddleocr-vl`, `pp-doclayout-v3`, `docparser-doc-prep`, and `docparser-candle-utils`. Without `--features mkl`, inference uses Candle’s default CPU backend (no MKL required).

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

## Pipeline

Single **PaddleOCR-VL v1.6** orchestration (`PipelineConfig::default()`): layout threshold 0.3, NMS, per-class merge, PaddleX markdown ignores, and **document preprocessing enabled by default** (orientation then unwarping). PaddleX YAML sets `use_doc_preprocessor: false`; docparser turns it on for skewed/curved/rotated inputs — see [docs/alignment_defaults.md](docs/alignment_defaults.md).

Opt out of doc preprocessing:

```bash
USE_DOC_ORIENTATION_CLASSIFY=false USE_DOC_UNWARPING=false cargo run -p docparser-api
```

Each parse adds ~40 MB mmap for doc-prep weights and modest per-page latency on top of layout + VLM.

## Workspace crates

| Crate | Role |
|-------|------|
| `docparser-download` | Parallel HF + fixture downloader |
| `docparser-candle-utils` | Shared safetensors mmap / parity helpers |
| `paddleocr-vl` | In-tree PaddleOCR-VL Candle inference |
| `pp-doclayout-v3` | PP-DocLayoutV3 layout (Candle + safetensors) |
| `docparser-doc-prep` | Document orientation + UVDoc unwarping (`orientation` / `unwarp` modules) |
| `docparser-pipeline` | Full orchestration (doc prep + layout + VLM) |
| `docparser-api` | Axum HTTP server |
| `docparser-test-utils` | Parity test helpers |

## Tests

Fast checks (no weights):

```bash
cargo nextest run --workspace
```

Slow parity tests (requires downloaded models; VLM generate ~5 min on CPU):

```bash
# VLM only (avoid loading the full pipeline):
RUN_SLOW=1 cargo nextest run -p paddleocr-vl --test preprocess_parity --test generate_parity --run-ignored all
# All slow tests:
RUN_SLOW=1 cargo nextest run --workspace --run-ignored all
```

Regenerate golden files (optional, Python dev harness):

```bash
pip install 'transformers==4.55.0' torch pillow einops sentencepiece
python tools/parity_gen.py --update-goldens --layout --vlm --doc-prep
# If HF VLM generate fails, fall back to: cargo run -p paddleocr-vl --bin vlm_write_golden --release
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
| `USE_DOC_ORIENTATION_CLASSIFY` | `true` | Rotate page via PP-LCNet doc ori |
| `USE_DOC_UNWARPING` | `true` | Rectify curved pages via UVDoc |
| `LAYOUT_UNCLIP_RATIO` | `1.0` | Scale layout boxes in postprocess (`1.0` = unchanged) |
| `CROP_PADDING_RATIO` | `0.0` | Extra symmetric crop padding (debug; not PaddleX) |
| `LAYOUT_THRESHOLD` | `0.3` | Layout score threshold |
| `LAYOUT_NMS` | `true` | Enable layout NMS |
| `MERGE_LAYOUT_BLOCKS` | `true` | Merge adjacent text crops for VLM (`merge_blocks`) |
| `HF_TOKEN` | — | Optional HuggingFace auth |
| `RUN_SLOW` | — | Enable ignored parity tests |
| `OMP_NUM_THREADS` | — | MKL/OpenMP thread count when built with `--features mkl` |
| `MKL_THREADING_LAYER` | — | Set to `GNU` if MKL reports threading errors on Linux |

## Alignment with official Paddle

Uses the **PaddleOCR-VL-1.6** stack (`PP-DocLayoutV3` + `PaddleOCR-VL-1.6`), not the separate **PP-StructureV3** multi-model pipeline.

- **Tensor parity:** HF Transformers + safetensors weights (`parity_gen.py` goldens under `tests/goldens/`).
- **Layout:** `preprocessor_config.json`-driven resize/normalize; labels from `config.json` `id2label`.
- **VLM:** Greedy decode; `generation_config.json` caps `max_new_tokens`; manual prompt layout matches HF `AutoProcessor` length (see slow tests).
- **Orchestration:** `PipelineConfig::default()` — threshold 0.3, NMS, per-class merge, PaddleX markdown ignores, doc prep on — [docs/alignment_defaults.md](docs/alignment_defaults.md), [docs/paddleocr_model_alignment.md](docs/paddleocr_model_alignment.md).
- **Layout labels & HF models:** [docs/layout_labels_and_models.md](docs/layout_labels_and_models.md).
- **Not implemented:** polygon masks, Paddle `.pdiparams` runtime, PP-StructureV3 specialist models.

## Notes

- First request loads ~1.9 GB VLM weights (mmap) plus ~130 MB layout and ~40 MB doc-prep safetensors; allow 1–3 minutes on CPU before the first `/v1/parse` completes.
- Per-page latency depends on region count (each layout region runs a VLM decode).
- Optional: `cargo run -p docparser-download -- --include-reference` fetches HF `modeling_*.py` for porting.
