//! Document preprocessor: orientation classification then geometric unwarping.

use anyhow::Result;
use image::DynamicImage;
use pp_lcnet_doc_ori::DocOrientationModel;
use serde::{Deserialize, Serialize};
use uvdoc::UvdocModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPreprocessorConfig {
    pub use_orientation: bool,
    pub use_unwarping: bool,
}

impl Default for DocPreprocessorConfig {
    fn default() -> Self {
        Self {
            use_orientation: true,
            use_unwarping: true,
        }
    }
}

pub struct DocPreprocessor {
    orientation: Option<DocOrientationModel>,
    unwarping: Option<UvdocModel>,
}

impl DocPreprocessor {
    pub fn from_model_dirs(
        doc_ori_dir: Option<&std::path::Path>,
        uvdoc_dir: Option<&std::path::Path>,
        cfg: &DocPreprocessorConfig,
    ) -> Result<Self> {
        let orientation = if cfg.use_orientation {
            Some(DocOrientationModel::from_dir(
                doc_ori_dir.ok_or_else(|| {
                    anyhow::anyhow!("doc orientation model path required when use_orientation=true")
                })?,
            )?)
        } else {
            None
        };
        let unwarping = if cfg.use_unwarping {
            Some(UvdocModel::from_dir(
                uvdoc_dir.ok_or_else(|| {
                    anyhow::anyhow!("UVDoc model path required when use_unwarping=true")
                })?,
            )?)
        } else {
            None
        };
        Ok(Self {
            orientation,
            unwarping,
        })
    }

    pub fn preprocess_document(
        &self,
        image: DynamicImage,
        cfg: &DocPreprocessorConfig,
    ) -> Result<(DynamicImage, Vec<&'static str>)> {
        let mut stages = Vec::new();
        let mut current = image;

        if cfg.use_orientation {
            if let Some(model) = &self.orientation {
                let (rotated, _angle) = model.predict_and_rotate(current)?;
                current = rotated;
                stages.push("doc_orientation");
            }
        }

        if cfg.use_unwarping {
            if let Some(model) = &self.unwarping {
                let rgb = current.to_rgb8();
                let rectified = model.rectify(&rgb)?;
                current = DynamicImage::ImageRgb8(rectified);
                stages.push("doc_unwarping");
            }
        }

        Ok((current, stages))
    }
}

pub fn preprocess_document(
    image: DynamicImage,
    prep: &DocPreprocessor,
    cfg: &DocPreprocessorConfig,
) -> Result<DynamicImage> {
    prep.preprocess_document(image, cfg).map(|(img, _)| img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_enables_both_stages() {
        let cfg = DocPreprocessorConfig::default();
        assert!(cfg.use_orientation);
        assert!(cfg.use_unwarping);
    }
}
