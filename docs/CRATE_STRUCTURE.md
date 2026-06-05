# Crate layout

The workspace splits model code into separate crates (`paddleocr-vl`, `pp-doclayout-v3`,
`docparser-doc-prep`) so Cargo can compile them in parallel. `docparser-doc-prep` combines
document orientation (`orientation`) and unwarping (`unwarp`) modules that share the same
load/preprocess patterns. Application boundaries (`docparser-api`, `docparser-download`) and
shared utilities stay separate.

Consolidating the large VLM/layout crates into a single library would reduce workspace
complexity but increase incremental rebuild cost. Revisit only if link or full-workspace build
times dominate development (profile with `cargo build --timings`).
