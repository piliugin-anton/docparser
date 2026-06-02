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
    while let Some(cur) = elements.first().cloned() {
        kept.push(cur.clone());
        elements.remove(0);
        elements.retain(|e| iou(cur.bbox, e.bbox) < iou_threshold);
    }
    for (i, el) in kept.iter_mut().enumerate() {
        el.id = i;
    }
    kept
}
