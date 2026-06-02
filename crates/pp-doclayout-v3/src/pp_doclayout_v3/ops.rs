//! Small tensor ops missing from Candle.

use candle_core::{Result, Tensor};

/// Top-k along last dimension; returns (values, indices as I64).
pub fn topk_last_dim(scores: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let dims = scores.dims();
    let last = *dims.last().unwrap();
    let batch = scores.elem_count() / last;
    let data = scores.flatten_all()?.to_vec1::<f32>()?;
    let mut values = Vec::with_capacity(batch * k);
    let mut indices = Vec::with_capacity(batch * k);
    for b in 0..batch {
        let base = b * last;
        let mut order: Vec<usize> = (0..last).collect();
        order.sort_by(|&a, &c| data[base + c].partial_cmp(&data[base + a]).unwrap());
        for i in 0..k.min(last) {
            let idx = order[i];
            values.push(data[base + idx]);
            indices.push(idx as i64);
        }
    }
    let mut out_dims = dims.to_vec();
    out_dims[dims.len() - 1] = k;
    let v = Tensor::from_vec(values, out_dims.as_slice(), scores.device())?;
    let idx = Tensor::from_vec(indices, out_dims.as_slice(), scores.device())?.to_dtype(candle_core::DType::I64)?;
    Ok((v, idx))
}

pub fn class_and_query_index(
    flat_index: &Tensor,
    num_classes: usize,
    num_queries: usize,
) -> Result<(Tensor, Tensor)> {
    let idx = flat_index.flatten_all()?.to_vec1::<i64>()?;
    let nc = num_classes as i64;
    let max_q = (num_queries.saturating_sub(1)) as i64;
    let mut labels = Vec::with_capacity(idx.len());
    let mut queries = Vec::with_capacity(idx.len());
    for &i in &idx {
        labels.push(i % nc);
        queries.push((i / nc).min(max_q));
    }
    let shape = flat_index.dims();
    let labels = Tensor::from_vec(labels, shape, flat_index.device())?;
    let queries = Tensor::from_vec(queries, shape, flat_index.device())?;
    Ok((labels, queries))
}

/// Gather along `dim` using int64 indices (PyTorch `gather` on dim 1 for batch tensors).
pub fn gather_dim(src: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
    let idx = match indices.dims() {
        [_n] => indices.clone(),
        [1, _n] => indices.squeeze(0)?,
        [b, n] if *b == src.dims()[0] => {
            // Per-batch gather: only batch size 1 is used in this model.
            if *b != 1 {
                candle_core::bail!("gather_dim: batched index not supported for b={b}");
            }
            indices.squeeze(0)?
        }
        other => candle_core::bail!("gather_dim: unexpected index shape {other:?}"),
    };
    src.index_select(&idx, dim)
}

pub fn get_order_seqs(order_logits: &Tensor) -> Result<Tensor> {
    let scores = candle_nn::ops::sigmoid(order_logits)?;
    let data = scores.to_vec3::<f32>()?;
    let b = data.len();
    let n = data[0].len();
    let mut out = vec![0i64; b * n];
    for bi in 0..b {
        let mut votes = vec![0f32; n];
        for i in 0..n {
            for j in 0..n {
                if j > i {
                    votes[i] += data[bi][i][j];
                }
                if j < i {
                    votes[i] += 1.0 - data[bi][j][i];
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &c| votes[c].partial_cmp(&votes[a]).unwrap());
        for (rank, &pos) in order.iter().enumerate() {
            out[bi * n + pos] = rank as i64;
        }
    }
    Tensor::from_vec(out, (b, n), order_logits.device())?.to_dtype(candle_core::DType::I64)
}
