/// Build HuggingFace resolve URLs for model files.
pub fn hf_resolve_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_format() {
        let url = hf_resolve_url("PaddlePaddle/PaddleOCR-VL-1.6", "config.json");
        assert_eq!(
            url,
            "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/resolve/main/config.json"
        );
    }
}
