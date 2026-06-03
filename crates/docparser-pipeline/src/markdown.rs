//! Markdown assembly from parsed blocks.

use crate::Block;

pub fn blocks_to_markdown(blocks: &[Block], ignore_labels: &[String]) -> String {
    let mut out = String::new();
    for block in blocks {
        if ignore_labels.iter().any(|l| l == &block.label) {
            continue;
        }
        if !block.content.is_empty() {
            out.push_str(&block.content);
            if !block.content.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
    }
    out
}

/// PaddleOCR-VL-1.6 / PaddleX default (`markdown_ignore_labels` in pipeline YAML).
pub fn official_markdown_ignore_labels() -> Vec<String> {
    [
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}
