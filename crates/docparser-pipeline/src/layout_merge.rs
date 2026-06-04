//! Layout bbox merge via containment (`layout_merge_bboxes_mode` in PaddleX).

use pp_doclayout_v3::LayoutElement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MergeBboxesMode {
    Large,
    Small,
    Union,
}

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// PaddleX `is_contained`: intersection / inner_area >= 0.9
fn is_contained(inner: [f32; 4], outer: [f32; 4]) -> bool {
    let inner_area = area(inner);
    if inner_area <= 0.0 {
        return false;
    }
    let xi1 = inner[0].max(outer[0]);
    let yi1 = inner[1].max(outer[1]);
    let xi2 = inner[2].min(outer[2]);
    let yi2 = inner[3].min(outer[3]);
    let inter_w = (xi2 - xi1).max(0.0);
    let inter_h = (yi2 - yi1).max(0.0);
    inter_w * inter_h / inner_area >= 0.9
}

fn check_containment(
    elements: &[LayoutElement],
    formula_label: &str,
    category_label: Option<&str>,
    mode: MergeBboxesMode,
) -> (Vec<u8>, Vec<u8>) {
    let n = elements.len();
    let mut contains_other = vec![0u8; n];
    let mut contained_by_other = vec![0u8; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if elements[i].label == formula_label && elements[j].label != formula_label {
                continue;
            }
            match (category_label, mode) {
                (Some(cat), MergeBboxesMode::Large) if elements[j].label == cat => {
                    if is_contained(elements[i].bbox, elements[j].bbox) {
                        contained_by_other[i] = 1;
                        contains_other[j] = 1;
                    }
                }
                (Some(cat), MergeBboxesMode::Small) if elements[i].label == cat => {
                    if is_contained(elements[i].bbox, elements[j].bbox) {
                        contained_by_other[i] = 1;
                        contains_other[j] = 1;
                    }
                }
                (None, _) | (Some(_), MergeBboxesMode::Union) => {
                    if is_contained(elements[i].bbox, elements[j].bbox) {
                        contained_by_other[i] = 1;
                        contains_other[j] = 1;
                    }
                }
                _ => {}
            }
        }
    }
    (contains_other, contained_by_other)
}

/// Per-class merge mode for PaddleOCR-VL-1.6 (PaddleX YAML).
pub fn merge_mode_for_label(label: &str) -> MergeBboxesMode {
    match label {
        "chart" | "formula" | "display_formula" | "doc_title" | "inline_formula"
        | "paragraph_title" => MergeBboxesMode::Large,
        _ => MergeBboxesMode::Union,
    }
}

/// Labels that use non-union merge in the official YAML (unique list for iteration).
const MERGE_CATEGORY_LABELS: &[&str] = &[
    "chart",
    "formula",
    "display_formula",
    "doc_title",
    "inline_formula",
    "paragraph_title",
];

/// Apply PaddleX `layout_merge_bboxes_mode` dict using containment.
pub fn apply_layout_merge_bboxes(elements: Vec<LayoutElement>) -> Vec<LayoutElement> {
    if elements.is_empty() {
        return elements;
    }
    let n = elements.len();
    let mut keep_mask = vec![true; n];
    let formula_label = "formula";

    for &category_label in MERGE_CATEGORY_LABELS {
        let mode = merge_mode_for_label(category_label);
        if mode == MergeBboxesMode::Union {
            continue;
        }
        let (contains_other, contained_by_other) =
            check_containment(&elements, formula_label, Some(category_label), mode);
        match mode {
            MergeBboxesMode::Large => {
                for i in 0..n {
                    if contained_by_other[i] != 0 {
                        keep_mask[i] = false;
                    }
                }
            }
            MergeBboxesMode::Small => {
                for i in 0..n {
                    if contains_other[i] != 0 && contained_by_other[i] == 0 {
                        keep_mask[i] = false;
                    }
                }
            }
            MergeBboxesMode::Union => {}
        }
        let _ = contains_other;
    }

    let out: Vec<LayoutElement> = elements
        .into_iter()
        .enumerate()
        .filter_map(|(i, el)| keep_mask[i].then_some(el))
        .collect();
    out
}

/// Legacy overlap-based merge (tests / backward compat exports).
pub fn merge_layout_blocks(
    elements: Vec<LayoutElement>,
    mode: MergeBboxesMode,
) -> Vec<LayoutElement> {
    merge_layout_blocks_with_mode_fn(elements, |_| mode)
}

pub fn merge_layout_blocks_with_mode_fn(
    elements: Vec<LayoutElement>,
    mode_for_label: impl Fn(&str) -> MergeBboxesMode,
) -> Vec<LayoutElement> {
    if elements.is_empty() {
        return elements;
    }
    let n = elements.len();
    let mut keep_mask = vec![true; n];
    let formula_label = "formula";

    let categories: Vec<&str> = elements
        .iter()
        .map(|e| e.label.as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for category_label in categories {
        let mode = mode_for_label(category_label);
        if mode == MergeBboxesMode::Union {
            continue;
        }
        let (contains_other, contained_by_other) =
            check_containment(&elements, formula_label, Some(category_label), mode);
        match mode {
            MergeBboxesMode::Large => {
                for i in 0..n {
                    if contained_by_other[i] != 0 {
                        keep_mask[i] = false;
                    }
                }
            }
            MergeBboxesMode::Small => {
                for i in 0..n {
                    if contains_other[i] != 0 && contained_by_other[i] == 0 {
                        keep_mask[i] = false;
                    }
                }
            }
            MergeBboxesMode::Union => {}
        }
    }

    elements
        .into_iter()
        .enumerate()
        .filter_map(|(i, el)| keep_mask[i].then_some(el))
        .collect()
}
