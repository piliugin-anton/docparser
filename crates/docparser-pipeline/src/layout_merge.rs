//! Merge overlapping layout boxes (PaddleOCR `merge_layout_blocks`).

use pp_doclayout_v3::LayoutElement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MergeBboxesMode {
    Large,
    Small,
    Union,
}

fn contains(outer: [f32; 4], inner: [f32; 4]) -> bool {
    outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]
}

fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn related(a: [f32; 4], b: [f32; 4]) -> bool {
    overlaps(a, b) || contains(a, b) || contains(b, a)
}

fn merge_one_pair(mode: MergeBboxesMode, el: &LayoutElement, o: &LayoutElement) -> MergePairAction {
    match mode {
        MergeBboxesMode::Union => MergePairAction::KeepBoth,
        MergeBboxesMode::Large => {
            if area(el.bbox) >= area(o.bbox) {
                MergePairAction::DropOther
            } else {
                MergePairAction::DropIncoming
            }
        }
        MergeBboxesMode::Small => {
            if area(el.bbox) <= area(o.bbox) {
                MergePairAction::DropOther
            } else {
                MergePairAction::DropIncoming
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergePairAction {
    KeepBoth,
    DropOther,
    DropIncoming,
}

/// Per-class merge mode for PaddleOCR-VL-1.6 (PaddleX YAML).
pub fn merge_mode_for_label(label: &str) -> MergeBboxesMode {
    match label {
        "chart" | "formula" | "display_formula" | "doc_title" | "inline_formula"
        | "paragraph_title" => MergeBboxesMode::Large,
        _ => MergeBboxesMode::Union,
    }
}

/// Merge overlapping / nested boxes using one mode for all elements.
pub fn merge_layout_blocks(
    elements: Vec<LayoutElement>,
    mode: MergeBboxesMode,
) -> Vec<LayoutElement> {
    merge_layout_blocks_with_mode_fn(elements, |_| mode)
}

/// Merge overlapping / nested boxes; mode is chosen from each incoming element's label.
pub fn merge_layout_blocks_with_mode_fn(
    elements: Vec<LayoutElement>,
    mode_for_label: impl Fn(&str) -> MergeBboxesMode,
) -> Vec<LayoutElement> {
    if elements.is_empty() {
        return elements;
    }

    let mut out: Vec<LayoutElement> = Vec::new();
    for el in elements {
        let mode = mode_for_label(&el.label);
        if mode == MergeBboxesMode::Union {
            out.push(el);
            continue;
        }

        let mut keep = true;
        let mut i = 0;
        while i < out.len() {
            let o = &out[i];
            if !related(el.bbox, o.bbox) {
                i += 1;
                continue;
            }
            match merge_one_pair(mode, &el, o) {
                MergePairAction::KeepBoth => {
                    i += 1;
                }
                MergePairAction::DropOther => {
                    out.remove(i);
                }
                MergePairAction::DropIncoming => {
                    keep = false;
                    break;
                }
            }
        }
        if keep {
            out.push(el);
        }
    }
    for (idx, el) in out.iter_mut().enumerate() {
        el.id = idx;
    }
    out
}
