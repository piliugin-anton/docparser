//! PaddleX `filter_overlap_boxes` for layout detections.

use pp_doclayout_v3::LayoutElement;

const SMALL_BOX_THRESHOLD: f32 = 6.0;
const INLINE_FORMULA_OVERLAP: f32 = 0.5;
const OVERLAP_DROP_THRESHOLD: f32 = 0.7;

fn bbox_area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn pairwise_overlap_small(coords: &[[f32; 4]]) -> Vec<Vec<f32>> {
    let n = coords.len();
    let mut overlap = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let [x1, y1, x2, y2] = coords[i];
            let [x1b, y1b, x2b, y2b] = coords[j];
            let inter_x1 = x1.max(x1b);
            let inter_y1 = y1.max(y1b);
            let inter_x2 = x2.min(x2b);
            let inter_y2 = y2.min(y2b);
            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;
            let small_area = bbox_area(coords[i]).min(bbox_area(coords[j]));
            overlap[i][j] = if small_area > 0.0 {
                inter_area / small_area
            } else {
                0.0
            };
        }
    }
    overlap
}

fn exception_labels() -> [&'static str; 4] {
    ["image", "table", "seal", "chart"]
}

fn should_skip_overlap_drop(label_a: &str, label_b: &str) -> bool {
    let mut labels = std::collections::HashSet::new();
    labels.insert(label_a);
    labels.insert(label_b);
    let exc: std::collections::HashSet<_> = exception_labels().into_iter().collect();
    if labels.intersection(&exc).count() <= 1 {
        return false;
    }
    if !labels.contains("table") {
        return true;
    }
    labels.is_subset(&exc)
}

/// Remove overlapping layout boxes (PaddleX paddleocr_vl `filter_overlap_boxes`).
pub fn filter_overlap_boxes(elements: Vec<LayoutElement>) -> Vec<LayoutElement> {
    let boxes: Vec<LayoutElement> = elements
        .into_iter()
        .filter(|e| e.label != "reference")
        .collect();
    if boxes.is_empty() {
        return boxes;
    }

    let coords: Vec<[f32; 4]> = boxes.iter().map(|b| b.bbox).collect();
    let widths: Vec<f32> = coords.iter().map(|c| c[2] - c[0]).collect();
    let heights: Vec<f32> = coords.iter().map(|c| c[3] - c[1]).collect();
    let areas: Vec<f32> = coords.iter().map(|c| bbox_area(*c)).collect();
    let overlap_matrix = pairwise_overlap_small(&coords);

    let mut dropped = std::collections::HashSet::new();
    let n = boxes.len();

    for i in 0..n {
        if widths[i] < SMALL_BOX_THRESHOLD || heights[i] < SMALL_BOX_THRESHOLD {
            dropped.insert(i);
        }
        for j in (i + 1)..n {
            if dropped.contains(&i) || dropped.contains(&j) {
                continue;
            }
            let overlap_ratio = overlap_matrix[i][j];
            let li = boxes[i].label.as_str();
            let lj = boxes[j].label.as_str();

            if li == "inline_formula" || lj == "inline_formula" {
                if overlap_ratio > INLINE_FORMULA_OVERLAP {
                    if li == "inline_formula" {
                        dropped.insert(i);
                    }
                    if lj == "inline_formula" {
                        dropped.insert(j);
                    }
                }
                continue;
            }

            if overlap_ratio > OVERLAP_DROP_THRESHOLD {
                if should_skip_overlap_drop(li, lj) {
                    continue;
                }
                if areas[i] >= areas[j] {
                    dropped.insert(j);
                } else {
                    dropped.insert(i);
                }
            }
        }
    }

    boxes
        .into_iter()
        .enumerate()
        .filter_map(|(idx, el)| (!dropped.contains(&idx)).then_some(el))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pp_doclayout_v3::LayoutElement;
    use proptest::prop_assert;

    fn el(id: usize, label: &str, bbox: [f32; 4]) -> LayoutElement {
        LayoutElement {
            id,
            order: Some(id),
            label: label.into(),
            score: 0.9,
            bbox,
            text: None,
        }
    }

    #[test]
    fn drops_tiny_boxes() {
        let elements = vec![el(0, "text", [0.0, 0.0, 5.0, 5.0])];
        let out = filter_overlap_boxes(elements);
        assert!(out.is_empty());
    }

    #[test]
    fn removes_reference_labels() {
        let elements = vec![
            el(0, "reference", [0.0, 0.0, 50.0, 50.0]),
            el(1, "text", [10.0, 10.0, 40.0, 40.0]),
        ];
        let out = filter_overlap_boxes(elements);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "text");
    }

    #[test]
    fn inline_formula_overlap_drops_formula() {
        let elements = vec![
            el(0, "text", [0.0, 0.0, 100.0, 100.0]),
            el(1, "inline_formula", [10.0, 10.0, 90.0, 90.0]),
        ];
        let out = filter_overlap_boxes(elements);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "text");
    }

    #[test]
    fn image_table_overlap_not_dropped() {
        let elements = vec![
            el(0, "image", [0.0, 0.0, 100.0, 100.0]),
            el(1, "table", [10.0, 10.0, 90.0, 90.0]),
        ];
        let out = filter_overlap_boxes(elements);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn high_overlap_drops_smaller() {
        let elements = vec![
            el(0, "text", [0.0, 0.0, 100.0, 100.0]),
            el(1, "text", [10.0, 10.0, 30.0, 30.0]),
        ];
        let out = filter_overlap_boxes(elements);
        assert_eq!(out.len(), 1);
        assert!((out[0].bbox[2] - 100.0).abs() < 1e-3);
    }

    proptest::proptest! {
        #[test]
        fn filter_never_increases_count(
            x1 in 0.0f32..50.0,
            y1 in 0.0f32..50.0,
            w in 1.0f32..40.0,
            h in 1.0f32..40.0,
        ) {
            let bbox = [x1, y1, x1 + w, y1 + h];
            let elements = vec![
                el(0, "text", bbox),
                el(1, "reference", bbox),
            ];
            let out = filter_overlap_boxes(elements);
            prop_assert!(out.len() <= 1);
            prop_assert!(out.iter().all(|e| e.label != "reference"));
        }
    }
}
