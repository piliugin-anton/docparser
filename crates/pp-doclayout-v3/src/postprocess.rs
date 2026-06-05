//! HF `post_process_object_detection` (boxes + reading order, no mask polygons).

use std::collections::HashMap;

use candle_core::{D, Result, Tensor};
use candle_nn::ops::sigmoid;

use crate::LayoutElement;
use crate::pp_doclayout_v3::ModelOutputs;
use crate::pp_doclayout_v3::ops::{
    class_and_query_index, gather_dim, get_order_seqs, topk_last_dim,
};

pub fn post_process_object_detection(
    outputs: &ModelOutputs,
    orig_height: u32,
    orig_width: u32,
    id2label: &HashMap<u32, String>,
    threshold: f32,
) -> Result<Vec<LayoutElement>> {
    let label_name = |id: i64| -> String {
        id2label
            .get(&(id as u32))
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    };

    let boxes = &outputs.pred_boxes;
    let logits = &outputs.logits;

    let box_centers = boxes.narrow(D::Minus1, 0, 2)?;
    let box_dims = boxes.narrow(D::Minus1, 2, 2)?;
    let half = Tensor::new(&[0.5f32], boxes.device())?;
    let top_left = (&box_centers - &box_dims.broadcast_mul(&half)?)?;
    let bottom_right = (&box_centers + &box_dims.broadcast_mul(&half)?)?;
    let mut boxes = Tensor::cat(&[&top_left, &bottom_right], D::Minus1)?;

    let img_h = orig_height as f32;
    let img_w = orig_width as f32;
    let scale = Tensor::new(&[img_w, img_h, img_w, img_h], boxes.device())?.reshape((1, 1, 4))?;
    boxes = boxes.broadcast_mul(&scale)?;

    let (_batch, num_top_queries, num_classes) = logits.dims3()?;
    let num_queries = num_top_queries;
    let scores = sigmoid(logits)?;
    let flat_scores = scores.flatten(1, 2)?;
    let (top_scores, index) = topk_last_dim(&flat_scores, num_top_queries)?;
    let (labels, query_index) = class_and_query_index(&index, num_classes, num_queries)?;

    let order_seqs = get_order_seqs(&outputs.order_logits)?;

    let boxes = gather_dim(&boxes, &query_index, 1)?;
    let order_seqs = gather_dim(&order_seqs, &query_index, 1)?;

    let scores_v = top_scores.to_vec2::<f32>()?;
    let labels_v = labels.to_vec2::<i64>()?;
    let boxes_v = boxes.to_vec3::<f32>()?;
    let order_v = order_seqs.to_vec2::<i64>()?;

    let mut elements = Vec::new();
    for i in 0..scores_v[0].len() {
        let score = scores_v[0][i];
        if score < threshold {
            continue;
        }
        let b = &boxes_v[0][i];
        elements.push(LayoutElement {
            id: 0,
            order: Some(order_v[0][i] as usize),
            label: label_name(labels_v[0][i]),
            score,
            bbox: [b[0], b[1], b[2], b[3]],
            text: None,
        });
    }

    elements.sort_by_key(|e| e.order.unwrap_or(usize::MAX));
    for (i, el) in elements.iter_mut().enumerate() {
        el.id = i;
    }
    Ok(elements)
}
