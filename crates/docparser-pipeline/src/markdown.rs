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

#[cfg(test)]
mod tests {
    use super::*;

    fn block(label: &str, content: &str) -> Block {
        Block {
            id: 0,
            order: Some(0),
            label: label.into(),
            bbox: [0.0, 0.0, 1.0, 1.0],
            score: 1.0,
            content: content.into(),
            group_id: None,
        }
    }

    #[test]
    fn skips_ignored_labels() {
        let blocks = vec![
            block("text", "hello"),
            block("number", "99"),
            block("text", "world"),
        ];
        let ignore = vec!["number".into()];
        let md = blocks_to_markdown(&blocks, &ignore);
        assert!(md.contains("hello"));
        assert!(md.contains("world"));
        assert!(!md.contains("99"));
    }

    #[test]
    fn adds_trailing_newline_between_blocks() {
        let blocks = vec![block("text", "line")];
        let md = blocks_to_markdown(&blocks, &[]);
        assert_eq!(md, "line\n\n");
    }
}
