//! File manifests for HuggingFace model repos and parity test fixtures.

pub const VLM_REPO: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
pub const LAYOUT_REPO: &str = "PaddlePaddle/PP-DocLayoutV3_safetensors";
pub const DOC_ORI_REPO: &str = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors";
pub const UVDOC_REPO: &str = "PaddlePaddle/UVDoc_safetensors";

pub const VLM_DIR_NAME: &str = "PaddleOCR-VL-1.6";
pub const LAYOUT_DIR_NAME: &str = "PP-DocLayoutV3";
pub const DOC_ORI_DIR_NAME: &str = "PP-LCNet_x1_0_doc_ori";
pub const UVDOC_DIR_NAME: &str = "UVDoc";

/// Required for VLM inference (from HF PaddleOCR-VL-1.6 tree).
pub const VLM_REQUIRED: &[&str] = &[
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
];

/// Optional reference Python shipped on HF (porting aid).
pub const VLM_REFERENCE: &[&str] = &[
    "modeling_paddleocr_vl.py",
    "configuration_paddleocr_vl.py",
    "image_processing_paddleocr_vl.py",
    "processing_paddleocr_vl.py",
];

/// Required for layout inference (from PP-DocLayoutV3_safetensors).
pub const LAYOUT_REQUIRED: &[&str] = &[
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "inference.yml",
];

/// Expected file sizes from HuggingFace API (bytes). Used for skip-if-present checks.
pub const VLM_SIZES: &[(&str, u64)] = &[
    ("model.safetensors", 1_917_255_968),
    ("config.json", 2_059),
    ("preprocessor_config.json", 641),
    ("processor_config.json", 137),
    ("generation_config.json", 133),
    ("tokenizer.json", 11_189_060),
    ("tokenizer.model", 1_614_363),
    ("tokenizer_config.json", 186_947),
    ("special_tokens_map.json", 1_151),
    ("added_tokens.json", 25_381),
    ("chat_template.jinja", 1_474),
];

pub const LAYOUT_SIZES: &[(&str, u64)] = &[
    ("model.safetensors", 133_270_468),
    ("config.json", 2_460),
    ("preprocessor_config.json", 575),
    ("inference.yml", 1_482),
];

/// Required for document orientation classification.
pub const DOC_ORI_REQUIRED: &[&str] = &[
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
];

/// Required for UVDoc geometric rectification.
pub const UVDOC_REQUIRED: &[&str] = &[
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
];

pub const DOC_ORI_SIZES: &[(&str, u64)] = &[
    ("model.safetensors", 6_769_728),
    ("config.json", 229),
    ("preprocessor_config.json", 749),
];

pub const UVDOC_SIZES: &[(&str, u64)] = &[
    ("model.safetensors", 31_667_380),
    ("config.json", 1_421),
    ("preprocessor_config.json", 240),
];

pub struct FixtureDownload {
    pub filename: &'static str,
    pub url: &'static str,
}

pub const FIXTURES: &[FixtureDownload] = &[
    FixtureDownload {
        filename: "ocr_demo2.jpg",
        url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg",
    },
    FixtureDownload {
        filename: "layout_demo.jpg",
        url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg",
    },
    FixtureDownload {
        filename: "doc_ori_demo.png",
        url: "https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/4ifXaBJmFByG_mAnF86Vv.png",
    },
    FixtureDownload {
        filename: "uvdoc_demo.jpeg",
        url: "https://cdn-uploads.huggingface.co/production/uploads/63d7b8ee07cd1aa3c49a2026/SfMVKd0xnMII5KBDV6Mfz.jpeg",
    },
];

pub fn expected_size(repo_files: &[(&str, u64)], name: &str) -> Option<u64> {
    repo_files
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
}
