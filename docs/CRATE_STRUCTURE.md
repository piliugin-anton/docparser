# Crate layout

The workspace splits model code into separate crates (`paddleocr-vl`, `pp-doclayout-v3`,
`pp-lcnet-doc-ori`, `uvdoc`) so Cargo can compile them in parallel. Application boundaries
(`docparser-api`, `docparser-download`) and shared utilities stay separate.

Consolidating model crates into a single library would reduce workspace complexity but
increase incremental rebuild cost for any change that touches shared inference code. Revisit
only if link or full-workspace build times dominate development (profile with
`cargo build --timings`).
