//! Document preprocessor stubs (orientation / unwarping).

use anyhow::{bail, Result};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocPreprocessorConfig {
    pub use_orientation: bool,
    pub use_unwarping: bool,
}

pub fn preprocess_document(
    image: DynamicImage,
    cfg: &DocPreprocessorConfig,
) -> Result<DynamicImage> {
    if cfg.use_orientation {
        bail!(
            "document orientation classification is not implemented; \
             set use_doc_orientation_classify=false or use PaddleOCR-VL"
        );
    }
    if cfg.use_unwarping {
        bail!(
            "document unwarping is not implemented; \
             set use_doc_unwarping=false or use PaddleOCR-VL"
        );
    }
    Ok(image)
}
