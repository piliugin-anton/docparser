//! Decode ONNX detection rows into [`LayoutElement`].

use crate::{label_name, LayoutElement};

const SCORE_THRESH: f32 = 0.5;

/// ONNX row: `[class_id, score, xmin, ymin, xmax, ymax, read_order]`.
pub fn decode_detections(rows: &[[f32; 7]]) -> Vec<LayoutElement> {
    let mut elements: Vec<LayoutElement> = rows
        .iter()
        .filter(|r| r[1] > SCORE_THRESH)
        .map(|r| LayoutElement {
            id: 0,
            order: Some(r[6].round() as usize),
            label: label_name(r[0] as i64).to_string(),
            score: r[1],
            bbox: [r[2], r[3], r[4], r[5]],
            text: None,
        })
        .collect();

    elements.sort_by_key(|e| e.order.unwrap_or(usize::MAX));
    for (i, el) in elements.iter_mut().enumerate() {
        el.id = i;
    }
    elements
}
