//! Layout detection postprocess: NMS, bbox merge, unclip (PaddleX order).

use pp_doclayout_v3::LayoutElement;

use crate::layout_merge::apply_layout_merge_bboxes;
use crate::layout_nms::layout_nms;

/// Scale box width/height around center. Ratio `1.0` leaves the box unchanged.
pub fn unclip_bbox(bbox: [f32; 4], ratio: (f32, f32)) -> [f32; 4] {
    let [x1, y1, x2, y2] = bbox;
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let cx = x1 + w / 2.0;
    let cy = y1 + h / 2.0;
    let new_w = w * ratio.0;
    let new_h = h * ratio.1;
    [
        cx - new_w / 2.0,
        cy - new_h / 2.0,
        cx + new_w / 2.0,
        cy + new_h / 2.0,
    ]
}

pub fn clamp_bbox_to_image(bbox: [f32; 4], img_w: u32, img_h: u32) -> [f32; 4] {
    let w = img_w as f32;
    let h = img_h as f32;
    [
        bbox[0].max(0.0).min(w),
        bbox[1].max(0.0).min(h),
        bbox[2].max(0.0).min(w),
        bbox[3].max(0.0).min(h),
    ]
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutPostprocessConfig {
    pub layout_nms: bool,
    pub layout_nms_iou: f32,
    pub layout_unclip_ratio: f32,
}

pub fn apply_layout_postprocess(
    mut elements: Vec<LayoutElement>,
    img_w: u32,
    img_h: u32,
    cfg: LayoutPostprocessConfig,
) -> Vec<LayoutElement> {
    if cfg.layout_nms {
        elements = layout_nms(elements, cfg.layout_nms_iou);
    }
    elements = apply_layout_merge_bboxes(elements);
    let ratio = (cfg.layout_unclip_ratio, cfg.layout_unclip_ratio);
    if ratio.0 != 1.0 || ratio.1 != 1.0 {
        for el in &mut elements {
            el.bbox = clamp_bbox_to_image(unclip_bbox(el.bbox, ratio), img_w, img_h);
        }
    }
    for (idx, el) in elements.iter_mut().enumerate() {
        el.id = idx;
    }
    elements
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prop_assert;

    #[test]
    fn unclip_identity_at_one() {
        let b = [10.0, 20.0, 30.0, 40.0];
        let out = unclip_bbox(b, (1.0, 1.0));
        for i in 0..4 {
            assert!((out[i] - b[i]).abs() < 1e-5, "got {:?}", out);
        }
    }

    #[test]
    fn unclip_expands_twenty_percent() {
        let b = [0.0, 0.0, 100.0, 100.0];
        let out = unclip_bbox(b, (1.2, 1.2));
        assert!((out[0] - (-10.0)).abs() < 1e-4);
        assert!((out[2] - 110.0).abs() < 1e-4);
    }

    proptest::proptest! {
        #[test]
        fn clamp_bbox_stays_inside_image(
            x1 in 0.0f32..200.0,
            y1 in 0.0f32..200.0,
            x2 in 1.0f32..400.0,
            y2 in 1.0f32..400.0,
            img_w in 1u32..300,
            img_h in 1u32..300,
        ) {
            let bbox = [x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)];
            let out = clamp_bbox_to_image(bbox, img_w, img_h);
            let w = img_w as f32;
            let h = img_h as f32;
            prop_assert!(out[0] >= 0.0 && out[1] >= 0.0);
            prop_assert!(out[2] <= w && out[3] <= h);
            prop_assert!(out[0] <= out[2] && out[1] <= out[3]);
        }

        #[test]
        fn unclip_ratio_one_is_identity(
            x1 in 0.0f32..100.0,
            y1 in 0.0f32..100.0,
            w in 1.0f32..100.0,
            h in 1.0f32..100.0,
        ) {
            let bbox = [x1, y1, x1 + w, y1 + h];
            let out = unclip_bbox(bbox, (1.0, 1.0));
            for i in 0..4 {
                prop_assert!((out[i] - bbox[i]).abs() < 1e-4, "out={out:?} bbox={bbox:?}");
            }
        }
    }
}
