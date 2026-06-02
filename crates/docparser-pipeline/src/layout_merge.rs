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

/// Merge overlapping / nested boxes per `layout_merge_bboxes_mode`.
pub fn merge_layout_blocks(
    elements: Vec<LayoutElement>,
    mode: MergeBboxesMode,
) -> Vec<LayoutElement> {
    if mode == MergeBboxesMode::Union || elements.is_empty() {
        return elements;
    }

    let mut out: Vec<LayoutElement> = Vec::new();
    for el in elements {
        let mut keep = true;
        let mut i = 0;
        while i < out.len() {
            let o = &out[i];
            if !related(el.bbox, o.bbox) {
                i += 1;
                continue;
            }
            match mode {
                MergeBboxesMode::Large => {
                    if area(el.bbox) >= area(o.bbox) {
                        out.remove(i);
                    } else {
                        keep = false;
                        break;
                    }
                }
                MergeBboxesMode::Small => {
                    if area(el.bbox) <= area(o.bbox) {
                        out.remove(i);
                    } else {
                        keep = false;
                        break;
                    }
                }
                MergeBboxesMode::Union => {
                    i += 1;
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
