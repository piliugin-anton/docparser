//! PaddleX `merge_blocks`: merge adjacent text crops into composite VLM images.

use image::{Rgb, RgbImage, imageops};

use pp_doclayout_v3::LayoutElement;

pub const IMAGE_LABELS: &[&str] = &["image", "header_image", "footer_image"];

pub fn non_merge_labels(
    use_chart_recognition: bool,
    use_seal_recognition: bool,
    use_ocr_for_image_block: bool,
) -> Vec<String> {
    let mut labels: Vec<String> = if use_ocr_for_image_block {
        vec![]
    } else {
        IMAGE_LABELS.iter().map(|s| s.to_string()).collect()
    };
    labels.push("table".into());
    if use_chart_recognition {
        labels.push("chart".into());
    }
    if use_seal_recognition {
        labels.push("seal".into());
    }
    labels
}

#[derive(Debug, Clone)]
pub struct CropBlock {
    pub element: LayoutElement,
    pub crop: Option<RgbImage>,
    pub group_id: Option<usize>,
}

pub fn calculate_projection_overlap_ratio(
    bbox1: [f32; 4],
    bbox2: [f32; 4],
    direction: &str,
) -> f32 {
    let (start_index, end_index) = if direction == "horizontal" {
        (0usize, 2usize)
    } else {
        (1, 3)
    };
    let intersection_start = bbox1[start_index].max(bbox2[start_index]);
    let intersection_end = bbox1[end_index].min(bbox2[end_index]);
    let overlap = intersection_end - intersection_start;
    if overlap <= 0.0 {
        return 0.0;
    }
    let ref_width =
        bbox1[end_index].max(bbox2[end_index]) - bbox1[start_index].min(bbox2[start_index]);
    if ref_width > 0.0 {
        overlap / ref_width
    } else {
        0.0
    }
}

pub fn calculate_overlap_ratio(bbox1: [f32; 4], bbox2: [f32; 4]) -> f32 {
    let x_min_inter = bbox1[0].max(bbox2[0]);
    let y_min_inter = bbox1[1].max(bbox2[1]);
    let x_max_inter = bbox1[2].min(bbox2[2]);
    let y_max_inter = bbox1[3].min(bbox2[3]);
    let inter_w = (x_max_inter - x_min_inter).max(0.0);
    let inter_h = (y_max_inter - y_min_inter).max(0.0);
    let inter_area = inter_w * inter_h;
    let a1 = (bbox1[2] - bbox1[0]).max(0.0) * (bbox1[3] - bbox1[1]).max(0.0);
    let a2 = (bbox2[2] - bbox2[0]).max(0.0) * (bbox2[3] - bbox2[1]).max(0.0);
    let union = a1 + a2 - inter_area;
    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

fn merge_images(images: Vec<RgbImage>, aligns: &[&str]) -> RgbImage {
    if images.is_empty() {
        return RgbImage::new(1, 1);
    }
    if images.len() == 1 {
        return images.into_iter().next().expect("single-image merge group");
    }
    let mut x_offsets = vec![0u32; images.len()];
    let mut merged_w = images[0].width();

    for i in 1..images.len() {
        let img2_w = images[i].width();
        let step_w = merged_w.max(img2_w);
        let align = aligns.get(i - 1).copied().unwrap_or("center");
        let (x1, x2) = match align {
            "center" => ((step_w - merged_w) / 2, (step_w - img2_w) / 2),
            "right" => (step_w - merged_w, step_w - img2_w),
            _ => (0, 0),
        };
        for k in 0..i {
            x_offsets[k] += x1;
        }
        x_offsets[i] = x2;
        merged_w = step_w;
    }

    let total_h: u32 = images.iter().map(|i| i.height()).sum();
    let mut canvas = RgbImage::from_pixel(merged_w, total_h, Rgb([255, 255, 255]));
    let mut y_offset = 0u32;
    for (i, img) in images.iter().enumerate() {
        imageops::overlay(
            &mut canvas,
            img,
            i64::from(x_offsets[i]),
            i64::from(y_offset),
        );
        y_offset += img.height();
    }
    canvas
}

fn is_aligned(a1: f32, a2: f32) -> bool {
    (a1 - a2).abs() <= 5.0
}

fn get_alignment(block_bbox: [f32; 4], prev_bbox: [f32; 4]) -> &'static str {
    if is_aligned(block_bbox[0], prev_bbox[0]) {
        "left"
    } else if is_aligned(block_bbox[2], prev_bbox[2]) {
        "right"
    } else {
        "center"
    }
}

fn overlap_with_other_box(
    block_idx: usize,
    prev_idx: usize,
    blocks: &[Option<CropBlock>],
    non_merge_set: &std::collections::HashSet<&str>,
) -> bool {
    let prev_bbox = blocks[prev_idx]
        .as_ref()
        .expect("block slot must be populated during planning")
        .element
        .bbox;
    let block_bbox = blocks[block_idx]
        .as_ref()
        .expect("block slot must be populated during planning")
        .element
        .bbox;
    let min_box = [
        prev_bbox[0].min(block_bbox[0]),
        prev_bbox[1].min(block_bbox[1]),
        prev_bbox[2].max(block_bbox[2]),
        prev_bbox[3].max(block_bbox[3]),
    ];
    for (idx, other) in blocks.iter().enumerate() {
        if idx == block_idx || idx == prev_idx {
            continue;
        }
        let Some(other) = other else {
            continue;
        };
        if non_merge_set.contains(other.element.label.as_str())
            && calculate_overlap_ratio(min_box, other.element.bbox) > 0.0
        {
            return true;
        }
    }
    false
}

/// Merge adjacent text blocks into composite crops (PaddleX `merge_blocks`).
pub fn merge_blocks(blocks: Vec<CropBlock>, non_merge_labels: &[String]) -> Vec<CropBlock> {
    if blocks.is_empty() {
        return blocks;
    }

    let mut blocks: Vec<Option<CropBlock>> = blocks.into_iter().map(Some).collect();

    let non_merge_set: std::collections::HashSet<&str> =
        non_merge_labels.iter().map(|s| s.as_str()).collect();

    let mut non_merge_indices = std::collections::HashSet::new();
    let mut blocks_to_merge: Vec<(usize, usize)> = Vec::new();

    for (idx, block) in blocks.iter().enumerate() {
        let block = block
            .as_ref()
            .expect("block slot must be populated during planning");
        if non_merge_set.contains(block.element.label.as_str()) {
            non_merge_indices.insert(idx);
        } else {
            blocks_to_merge.push((idx, idx));
        }
    }

    let mut merged_groups: Vec<(Vec<usize>, Vec<&'static str>)> = Vec::new();
    let mut current_group: Vec<usize> = Vec::new();
    let mut current_aligns: Vec<&'static str> = Vec::new();

    for i in 0..blocks_to_merge.len() {
        let (idx, _) = blocks_to_merge[i];
        if current_group.is_empty() {
            current_group.push(idx);
            continue;
        }

        let prev_idx = blocks_to_merge[i - 1].0;
        let prev_block = blocks[prev_idx]
            .as_ref()
            .expect("block slot must be populated during planning");
        let block = blocks[idx]
            .as_ref()
            .expect("block slot must be populated during planning");
        let prev_bbox = prev_block.element.bbox;
        let block_bbox = block.element.bbox;
        let block_label = block.element.label.as_str();
        let prev_label = prev_block.element.label.as_str();

        let iou_h = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal");
        let is_cross = iou_h == 0.0
            && block_label == "text"
            && prev_label == "text"
            && block_bbox[0] > prev_bbox[2]
            && block_bbox[1] < prev_bbox[3]
            && (block_bbox[0] - prev_bbox[2])
                < (prev_bbox[2] - prev_bbox[0]).max(block_bbox[2] - block_bbox[0]) * 0.3;

        let is_updown_align = iou_h > 0.0
            && block_label == "text"
            && prev_label == "text"
            && block_bbox[3] >= prev_bbox[1]
            && (block_bbox[1] - prev_bbox[3]).abs()
                < (prev_bbox[3] - prev_bbox[1]).max(block_bbox[3] - block_bbox[1]) * 0.5
            && (is_aligned(block_bbox[0], prev_bbox[0]) ^ is_aligned(block_bbox[2], prev_bbox[2]))
            && overlap_with_other_box(idx, prev_idx, &blocks, &non_merge_set);

        let align_mode = if is_cross {
            Some("center")
        } else if is_updown_align {
            Some(get_alignment(block_bbox, prev_bbox))
        } else {
            None
        };

        if align_mode.is_some() {
            current_group.push(idx);
            if let Some(a) = align_mode {
                current_aligns.push(a);
            }
        } else {
            merged_groups.push((
                std::mem::take(&mut current_group),
                std::mem::take(&mut current_aligns),
            ));
            current_group.push(idx);
        }
    }
    if !current_group.is_empty() {
        merged_groups.push((current_group, current_aligns));
    }

    let mut group_by_start: std::collections::HashMap<
        usize,
        (usize, usize, Vec<usize>, Vec<&'static str>),
    > = std::collections::HashMap::new();
    for (group_indices, aligns) in merged_groups {
        let (Some(&start), Some(&end)) = (group_indices.iter().min(), group_indices.iter().max())
        else {
            continue;
        };
        group_by_start.insert(start, (start, end, group_indices, aligns));
    }

    let mut result_blocks: Vec<CropBlock> = Vec::new();
    let mut used_indices = std::collections::HashSet::new();
    let mut idx = 0usize;

    while idx < blocks.len() {
        let mut group_found = false;
        if let Some((start, end, group_indices, aligns)) = group_by_start.get(&idx) {
            let (start, end, group_indices, aligns) = (*start, *end, group_indices, aligns);
            if group_indices.iter().all(|i| !used_indices.contains(i)) {
                let crop_dims: Vec<(u32, u32)> = group_indices
                    .iter()
                    .filter_map(|&gi| {
                        blocks[gi]
                            .as_ref()
                            .and_then(|b| b.crop.as_ref().map(RgbImage::dimensions))
                    })
                    .collect();
                let merged_w = crop_dims.iter().map(|(w, _)| *w).max().unwrap_or(0);
                let merged_h: u32 = crop_dims.iter().map(|(_, h)| h).sum();
                let aspect_ratio = if merged_w > 0 {
                    merged_h as f32 / merged_w as f32
                } else {
                    f32::INFINITY
                };

                if aspect_ratio >= 3.0 {
                    for &block_idx in group_indices {
                        let mut b = blocks[block_idx]
                            .take()
                            .expect("block slot must be populated during merge");
                        b.group_id = None;
                        result_blocks.push(b);
                        used_indices.insert(block_idx);
                    }
                } else if !crop_dims.is_empty() {
                    let imgs: Vec<RgbImage> = group_indices
                        .iter()
                        .filter_map(|&gi| blocks[gi].as_mut().and_then(|b| b.crop.take()))
                        .collect();
                    let mut merged_crop = Some(merge_images(imgs, aligns));
                    for (j, &block_idx) in group_indices.iter().enumerate() {
                        let mut b = blocks[block_idx]
                            .take()
                            .expect("block slot must be populated during merge");
                        if j == 0 {
                            b.crop = merged_crop.take();
                        } else {
                            b.crop = None;
                        }
                        b.group_id = Some(group_indices[0]);
                        result_blocks.push(b);
                        used_indices.insert(block_idx);
                    }
                } else {
                    for &block_idx in group_indices {
                        result_blocks.push(
                            blocks[block_idx]
                                .take()
                                .expect("block slot must be populated during merge"),
                        );
                        used_indices.insert(block_idx);
                    }
                }

                let mut insert_list = Vec::new();
                for n_idx in (start + 1)..=end {
                    if non_merge_indices.contains(&n_idx) {
                        insert_list.push(n_idx);
                    }
                }
                for n_idx in insert_list {
                    result_blocks.push(
                        blocks[n_idx]
                            .take()
                            .expect("block slot must be populated during merge"),
                    );
                    used_indices.insert(n_idx);
                }
                idx = end + 1;
                group_found = true;
            }
        }
        if group_found {
            continue;
        }
        if non_merge_indices.contains(&idx) {
            if !used_indices.contains(&idx) {
                result_blocks.push(
                    blocks[idx]
                        .take()
                        .expect("block slot must be populated during merge"),
                );
                used_indices.insert(idx);
            }
        } else if !used_indices.contains(&idx) {
            result_blocks.push(
                blocks[idx]
                    .take()
                    .expect("block slot must be populated during merge"),
            );
            used_indices.insert(idx);
        }
        idx += 1;
    }

    result_blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use pp_doclayout_v3::LayoutElement;
    use proptest::prelude::*;

    fn block(id: usize, label: &str, h: u32) -> CropBlock {
        CropBlock {
            element: LayoutElement {
                id,
                order: Some(id),
                label: label.into(),
                score: 0.9,
                bbox: [0.0, 0.0, 100.0, h as f32],
                text: None,
            },
            crop: Some(RgbImage::new(50, h)),
            group_id: None,
        }
    }

    #[test]
    fn tall_merge_splits_by_aspect_ratio() {
        let blocks = vec![
            block(0, "text", 200),
            block(1, "text", 200),
            block(2, "text", 200),
        ];
        let nm = non_merge_labels(false, false, false);
        let out = merge_blocks(blocks, &nm);
        assert!(out.iter().all(|b| b.group_id.is_none()));
    }

    proptest::proptest! {
        #[test]
        fn overlap_ratio_bounded(a1 in 0.0f32..100.0, a2 in 0.0f32..100.0, b1 in 0.0f32..100.0, b2 in 0.0f32..100.0) {
            let x1 = a1.min(a2);
            let x2 = a1.max(a2);
            let y1 = b1.min(b2);
            let y2 = b1.max(b2);
            let bbox1 = [x1, y1, x2, y2];
            let bbox2 = [x1 + 1.0, y1 + 1.0, x2 + 1.0, y2 + 1.0];
            let ratio = calculate_overlap_ratio(bbox1, bbox2);
            prop_assert!(ratio >= 0.0 && ratio <= 1.0);
            prop_assert_eq!(ratio, calculate_overlap_ratio(bbox2, bbox1));
        }
    }
}
