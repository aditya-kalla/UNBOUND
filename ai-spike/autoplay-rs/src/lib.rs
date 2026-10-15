//! Autoplay & Dynamic Queue Generation Engine in Native Rust.
//! Uses Centroid Affinity + Skip Penalty + Maximal Marginal Relevance (MMR)
//! on 512-dimensional CLAP embeddings.

pub const EMBEDDING_DIM: usize = 512;

/// Fast dot product for 512-dimensional normalized vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    // Chunk in 8s for loop unrolling / SIMD autovectorization
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        sum += ca[0] * cb[0]
            + ca[1] * cb[1]
            + ca[2] * cb[2]
            + ca[3] * cb[3]
            + ca[4] * cb[4]
            + ca[5] * cb[5]
            + ca[6] * cb[6]
            + ca[7] * cb[7];
    }
    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }
    sum
}

/// Compute L2-normalized centroid from selected embedding indices.
pub fn compute_centroid(embeddings: &[f32], indices: &[usize]) -> Vec<f32> {
    let mut centroid = vec![0.0f32; EMBEDDING_DIM];
    if indices.is_empty() {
        return centroid;
    }

    for &idx in indices {
        let offset = idx * EMBEDDING_DIM;
        for i in 0..EMBEDDING_DIM {
            centroid[i] += embeddings[offset + i];
        }
    }

    let norm = centroid.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for x in centroid.iter_mut() {
            *x /= norm;
        }
    }
    centroid
}

/// Score candidates using Centroid Affinity minus Skip Penalty.
/// Score: s(x) = cos(x, c_liked) - beta * max(0, cos(x, c_skipped))
pub fn score_candidates(
    candidates: &[f32],
    n_candidates: usize,
    liked_centroid: &[f32],
    skipped_centroid: Option<&[f32]>,
    beta: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(n_candidates);

    for idx in 0..n_candidates {
        let offset = idx * EMBEDDING_DIM;
        let cand = &candidates[offset..offset + EMBEDDING_DIM];

        let affinity = dot_product(cand, liked_centroid);
        let penalty = if let Some(skip_c) = skipped_centroid {
            let skip_sim = dot_product(cand, skip_c);
            if skip_sim > 0.0 {
                beta * skip_sim
            } else {
                0.0
            }
        } else {
            0.0
        };

        scores.push(affinity - penalty);
    }

    scores
}

/// Generate Top-K queue using Maximal Marginal Relevance (MMR).
/// MMR objective: argmax_{x in C \ Q} [ lambda * score(x) - (1 - lambda) * max_{y in Q} sim(x, y) ]
pub fn mmr_queue(
    candidates: &[f32],
    n_candidates: usize,
    base_scores: &[f32],
    k: usize,
    lambda: f32,
    excluded_indices: &[usize],
) -> Vec<usize> {
    let mut queue = Vec::with_capacity(k);
    let mut selected_mask = vec![false; n_candidates];

    for &exc in excluded_indices {
        if exc < n_candidates {
            selected_mask[exc] = true;
        }
    }

    let target_k = k.min(n_candidates.saturating_sub(excluded_indices.len()));

    while queue.len() < target_k {
        let mut best_idx = None;
        let mut best_mmr_score = -1e9f32;

        for idx in 0..n_candidates {
            if selected_mask[idx] {
                continue;
            }

            let relevance = base_scores[idx];
            let cand_emb = &candidates[idx * EMBEDDING_DIM..(idx + 1) * EMBEDDING_DIM];

            // Compute maximum similarity to any track already in queue
            let mut max_sim = 0.0f32;
            for &q_idx in &queue {
                let q_emb = &candidates[q_idx * EMBEDDING_DIM..(q_idx + 1) * EMBEDDING_DIM];
                let sim = dot_product(cand_emb, q_emb);
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            let mmr_score = if queue.is_empty() {
                relevance
            } else {
                lambda * relevance - (1.0 - lambda) * max_sim
            };

            if mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_idx = Some(idx);
            }
        }

        if let Some(chosen) = best_idx {
            queue.push(chosen);
            selected_mask[chosen] = true;
        } else {
            break;
        }
    }

    queue
}

/// Compute Intra-List Diversity (ILD) of a queue:
/// Mean pairwise cosine distance (1.0 - cos(x_i, x_j)) across all pairs in the queue.
pub fn compute_intra_list_diversity(candidates: &[f32], queue: &[usize]) -> f32 {
    if queue.len() < 2 {
        return 0.0;
    }

    let mut dist_sum = 0.0f32;
    let mut pair_count = 0;

    for i in 0..queue.len() {
        let emb_i = &candidates[queue[i] * EMBEDDING_DIM..(queue[i] + 1) * EMBEDDING_DIM];
        for j in (i + 1)..queue.len() {
            let emb_j = &candidates[queue[j] * EMBEDDING_DIM..(queue[j] + 1) * EMBEDDING_DIM];
            let sim = dot_product(emb_i, emb_j);
            dist_sum += 1.0 - sim;
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        dist_sum / pair_count as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_identical() {
        let mut a = vec![0.0f32; EMBEDDING_DIM];
        a[0] = 1.0;
        let sim = dot_product(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_normalization() {
        let mut embs = vec![0.0f32; 2 * EMBEDDING_DIM];
        embs[0] = 1.0;
        embs[EMBEDDING_DIM + 1] = 1.0;
        let c = compute_centroid(&embs, &[0, 1]);
        let norm = c.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_skip_penalty_demotes_track() {
        // Track 0 matches liked centroid
        // Track 1 matches skipped centroid
        let mut embs = vec![0.0f32; 2 * EMBEDDING_DIM];
        embs[0] = 1.0;
        embs[EMBEDDING_DIM + 1] = 1.0;

        let liked = vec![1.0; EMBEDDING_DIM];
        let mut liked_norm = liked.clone();
        let norm_l = liked_norm.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for x in liked_norm.iter_mut() {
            *x /= norm_l;
        }

        let skipped = embs[EMBEDDING_DIM..2 * EMBEDDING_DIM].to_vec();

        // Without skip penalty
        let scores_no_penalty = score_candidates(&embs, 2, &liked_norm, None, 0.0);
        // With skip penalty
        let scores_with_penalty = score_candidates(&embs, 2, &liked_norm, Some(&skipped), 0.5);

        assert!(scores_with_penalty[1] < scores_no_penalty[1]);
    }
}
