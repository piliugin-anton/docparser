use image::{DynamicImage, RgbImage};

/// Rotate a document image by a Paddle orientation label (0, 90, 180, 270 degrees CW).
pub fn rotate_by_angle(image: DynamicImage, angle: u32) -> DynamicImage {
    match angle {
        90 => image.rotate90(),
        180 => image.rotate180(),
        270 => image.rotate270(),
        _ => image,
    }
}

pub fn rotate_rgb(image: RgbImage, angle: u32) -> RgbImage {
    match angle {
        0 => image,
        _ => rotate_by_angle(DynamicImage::ImageRgb8(image), angle).to_rgb8(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_90_swaps_dimensions() {
        let img = RgbImage::new(400, 200);
        let out = rotate_rgb(img, 90);
        assert_eq!(out.dimensions(), (200, 400));
    }

    #[test]
    fn rotate_0_unchanged() {
        let img = RgbImage::new(100, 50);
        let out = rotate_rgb(img, 0);
        assert_eq!(out.dimensions(), (100, 50));
    }
}
