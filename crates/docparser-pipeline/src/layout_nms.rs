//! IoU-based NMS on layout detection boxes (optional pipeline step).

use pp_doclayout_v3::LayoutElement;

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Greedy NMS sorted by descending score.
pub fn layout_nms(mut elements: Vec<LayoutElement>, iou_threshold: f32) -> Vec<LayoutElement> {
    elements.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = Vec::new();
    while !elements.is_empty() {
        let cur = elements.remove(0);
        elements.retain(|e| iou(cur.bbox, e.bbox) < iou_threshold);
        kept.push(cur);
    }
    for (i, el) in kept.iter_mut().enumerate() {
        el.id = i;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prop_assert;

    fn element(id: usize, score: f32, bbox: [f32; 4]) -> LayoutElement {
        LayoutElement {
            id,
            order: Some(id),
            label: "text".into(),
            score,
            bbox,
            text: None,
        }
    }

    #[test]
    fn suppresses_high_iou_lower_score_box() {
        let input = vec![
            element(0, 0.9, [0.0, 0.0, 10.0, 10.0]),
            element(1, 0.5, [1.0, 1.0, 11.0, 11.0]),
            element(2, 0.8, [50.0, 50.0, 60.0, 60.0]),
        ];
        let out = layout_nms(input, 0.5);
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|e| (e.score - 0.9).abs() < f32::EPSILON));
        assert!(out.iter().any(|e| (e.score - 0.8).abs() < f32::EPSILON));
        assert_eq!(out[0].id, 0);
        assert_eq!(out[1].id, 1);
    }

    #[test]
    fn keeps_disjoint_boxes() {
        let input = vec![
            element(0, 0.7, [0.0, 0.0, 10.0, 10.0]),
            element(1, 0.6, [20.0, 20.0, 30.0, 30.0]),
        ];
        let out = layout_nms(input, 0.5);
        assert_eq!(out.len(), 2);
    }

    proptest::proptest! {
        #[test]
        fn nms_never_increases_count(
            x1 in 0.0f32..50.0,
            y1 in 0.0f32..50.0,
            w in 1.0f32..30.0,
            h in 1.0f32..30.0,
            iou_thresh in 0.0f32..1.0,
        ) {
            let bbox = [x1, y1, x1 + w, y1 + h];
            let input = vec![
                element(0, 0.9, bbox),
                element(1, 0.5, [bbox[0] + 1.0, bbox[1] + 1.0, bbox[2] + 1.0, bbox[3] + 1.0]),
            ];
            let out = layout_nms(input, iou_thresh);
            prop_assert!(out.len() <= 2);
            for pair in out.windows(2) {
                prop_assert!(iou(pair[0].bbox, pair[1].bbox) < iou_thresh);
            }
        }

        #[test]
        fn iou_is_symmetric_and_bounded(
            ax1 in 0.0f32..100.0,
            ay1 in 0.0f32..100.0,
            ax2 in 1.0f32..100.0,
            ay2 in 1.0f32..100.0,
            bx1 in 0.0f32..100.0,
            by1 in 0.0f32..100.0,
            bx2 in 1.0f32..100.0,
            by2 in 1.0f32..100.0,
        ) {
            let a = [ax1.min(ax2), ay1.min(ay2), ax1.max(ax2), ay1.max(ay2)];
            let b = [bx1.min(bx2), by1.min(by2), bx1.max(bx2), by1.max(by2)];
            let ab = iou(a, b);
            let ba = iou(b, a);
            prop_assert!((ab - ba).abs() < 1e-6);
            prop_assert!(ab >= 0.0 && ab <= 1.0);
        }
    }
}
