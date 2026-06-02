use docparser_pipeline::{
    layout_nms, merge_layout_blocks, merge_layout_blocks_with_mode_fn,
    official_v16_merge_mode_for_label, MergeBboxesMode,
};
use pp_doclayout_v3::LayoutElement;

fn el(id: usize, label: &str, score: f32, bbox: [f32; 4]) -> LayoutElement {
    LayoutElement {
        id,
        order: Some(id),
        label: label.into(),
        score,
        bbox,
        text: None,
    }
}

#[test]
fn layout_nms_suppresses_overlap() {
    let elements = vec![
        el(0, "text", 0.9, [0.0, 0.0, 10.0, 10.0]),
        el(1, "text", 0.8, [1.0, 1.0, 9.0, 9.0]),
        el(2, "text", 0.7, [50.0, 50.0, 60.0, 60.0]),
    ];
    let out = layout_nms(elements, 0.5);
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|e| (e.bbox[0] - 50.0).abs() < 1e-3));
}

#[test]
fn merge_large_keeps_outer_box() {
    let elements = vec![
        el(0, "text", 0.9, [0.0, 0.0, 100.0, 100.0]),
        el(1, "text", 0.85, [10.0, 10.0, 20.0, 20.0]),
    ];
    let out = merge_layout_blocks(elements, MergeBboxesMode::Large);
    assert_eq!(out.len(), 1);
    assert!((out[0].bbox[2] - 100.0).abs() < 1e-3);
}

#[test]
fn official_v16_formula_large_keeps_outer() {
    // Union-mode `text` is kept first; incoming `formula` (large) drops the inner box.
    let elements = vec![
        el(0, "text", 0.85, [10.0, 10.0, 20.0, 20.0]),
        el(1, "formula", 0.9, [0.0, 0.0, 100.0, 100.0]),
    ];
    let out = merge_layout_blocks_with_mode_fn(elements, official_v16_merge_mode_for_label);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].label, "formula");
}

#[test]
fn official_v16_union_labels_keep_both_when_overlapping() {
    let elements = vec![
        el(0, "text", 0.9, [0.0, 0.0, 10.0, 10.0]),
        el(1, "table", 0.85, [1.0, 1.0, 9.0, 9.0]),
    ];
    let out = merge_layout_blocks_with_mode_fn(elements, official_v16_merge_mode_for_label);
    assert_eq!(out.len(), 2);
}

#[test]
fn official_v16_merge_mode_for_display_formula() {
    assert_eq!(
        official_v16_merge_mode_for_label("display_formula"),
        MergeBboxesMode::Large
    );
}
