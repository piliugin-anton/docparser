use docparser_pipeline::{layout_nms, merge_layout_blocks, MergeBboxesMode};
use pp_doclayout_v3::LayoutElement;

fn el(id: usize, score: f32, bbox: [f32; 4]) -> LayoutElement {
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
fn layout_nms_suppresses_overlap() {
    let elements = vec![
        el(0, 0.9, [0.0, 0.0, 10.0, 10.0]),
        el(1, 0.8, [1.0, 1.0, 9.0, 9.0]),
        el(2, 0.7, [50.0, 50.0, 60.0, 60.0]),
    ];
    let out = layout_nms(elements, 0.5);
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|e| (e.bbox[0] - 50.0).abs() < 1e-3));
}

#[test]
fn merge_large_keeps_outer_box() {
    let elements = vec![
        el(0, 0.9, [0.0, 0.0, 100.0, 100.0]),
        el(1, 0.85, [10.0, 10.0, 20.0, 20.0]),
    ];
    let out = merge_layout_blocks(elements, MergeBboxesMode::Large);
    assert_eq!(out.len(), 1);
    assert!((out[0].bbox[2] - 100.0).abs() < 1e-3);
}
