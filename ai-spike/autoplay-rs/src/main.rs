use autoplay_rs::*;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("======================================================================");
    println!("UNBOUND AI SPIKE - TASK T6: AUTOPLAY ON REAL EMBEDDINGS (RUST)");
    println!("======================================================================");

    // 1. Load real embeddings and clusters
    let bin_paths = [
        "data/real_embeddings_152.bin",
        "ai-spike/autoplay-rs/data/real_embeddings_152.bin",
        "../autoplay-rs/data/real_embeddings_152.bin",
    ];
    let bin_path = bin_paths
        .iter()
        .find(|p| Path::new(p).exists())
        .expect("Error: real_embeddings_152.bin not found");

    let meta_paths = [
        "data/clusters.json",
        "ai-spike/autoplay-rs/data/clusters.json",
        "../autoplay-rs/data/clusters.json",
    ];
    let meta_path = meta_paths
        .iter()
        .find(|p| Path::new(p).exists())
        .expect("Error: clusters.json not found");

    println!("Loading real embeddings from: {}", bin_path);
    let raw_bytes = fs::read(bin_path).expect("Failed to read embeddings binary");
    let n_floats = raw_bytes.len() / 4;
    let n_tracks = n_floats / EMBEDDING_DIM;

    let mut real_embs = Vec::with_capacity(n_floats);
    for chunk in raw_bytes.chunks_exact(4) {
        real_embs.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
    }
    println!("Loaded {} tracks (512-dim embeddings).", n_tracks);

    let meta_str = fs::read_to_string(meta_path).expect("Failed to read clusters.json");
    // Parse JSON manually or simply extract indices
    let liked_indices: Vec<usize> = extract_json_array(&meta_str, "liked_indices");
    let skipped_indices: Vec<usize> = extract_json_array(&meta_str, "skipped_indices");

    println!("Liked Cluster (Jazz/Lo-fi/Piano) Track Count:    {}", liked_indices.len());
    println!("Skipped Cluster (Metal/Trap/Heavy) Track Count: {}", skipped_indices.len());

    // 2. Behavioral Ledger Simulation on Real Tracks
    println!("\n----------------------------------------------------------------------");
    println!("BEHAVIORAL LEDGER SIMULATION (MEASURED ON REAL EMBEDDINGS)");
    println!("----------------------------------------------------------------------");

    let liked_centroid = compute_centroid(&real_embs, &liked_indices);
    let skipped_centroid = compute_centroid(&real_embs, &skipped_indices);

    // Baseline ranking: affinity only (no skip penalty)
    let scores_before = score_candidates(&real_embs, n_tracks, &liked_centroid, None, 0.0);
    let mut rank_before: Vec<usize> = (0..n_tracks).collect();
    rank_before.sort_by(|&a, &b| scores_before[b].partial_cmp(&scores_before[a]).unwrap());

    // Ranking after skip penalty (beta = 0.5)
    let scores_after = score_candidates(
        &real_embs,
        n_tracks,
        &liked_centroid,
        Some(&skipped_centroid),
        0.5,
    );
    let mut rank_after: Vec<usize> = (0..n_tracks).collect();
    rank_after.sort_by(|&a, &b| scores_after[b].partial_cmp(&scores_after[a]).unwrap());

    println!("Effect of Skip Penalty on Skipped Cluster Ranks:");
    for &skip_idx in &skipped_indices {
        let pos_before = rank_before.iter().position(|&x| x == skip_idx).unwrap() + 1;
        let pos_after = rank_after.iter().position(|&x| x == skip_idx).unwrap() + 1;
        println!(
            "  Track #{:<3} (Skipped Genre): Rank Before = #{:<3} -> Rank After = #{:<3} (Demoted by {} positions)",
            skip_idx,
            pos_before,
            pos_after,
            pos_after as i32 - pos_before as i32
        );
    }

    // Generate Next-10 Queue with MMR (lambda = 0.7 vs greedy lambda = 1.0)
    let greedy_queue = mmr_queue(&real_embs, n_tracks, &scores_after, 10, 1.0, &liked_indices);
    let mmr_q = mmr_queue(&real_embs, n_tracks, &scores_after, 10, 0.7, &liked_indices);

    let greedy_div = compute_intra_list_diversity(&real_embs, &greedy_queue);
    let mmr_div = compute_intra_list_diversity(&real_embs, &mmr_q);

    // Compute Precision of next-10 queue against liked cluster
    let mut liked_in_greedy = 0;
    for &idx in &greedy_queue {
        if liked_indices.contains(&idx) {
            liked_in_greedy += 1;
        }
    }
    let mut liked_in_mmr = 0;
    for &idx in &mmr_q {
        if liked_indices.contains(&idx) {
            liked_in_mmr += 1;
        }
    }

    println!("\nNext-10 Queue Metrics (k = 10):");
    println!("  Greedy Queue (lambda = 1.0):");
    println!("    - Precision@10 (Liked Cluster): {:.1}% ({}/10)", liked_in_greedy as f32 * 10.0, liked_in_greedy);
    println!("    - Intra-List Diversity (ILD):   {:.4}", greedy_div);
    println!("    - Liked Affinity Score:         {:.4}", mean_score(&scores_after, &greedy_queue));
    println!("  MMR Queue (lambda = 0.7):");
    println!("    - Precision@10 (Liked Cluster): {:.1}% ({}/10)", liked_in_mmr as f32 * 10.0, liked_in_mmr);
    println!("    - Intra-List Diversity (ILD):   {:.4} (+{:.1}% diversity)", mmr_div, (mmr_div - greedy_div) / greedy_div * 100.0);
    println!("    - Liked Affinity Score:         {:.4}", mean_score(&scores_after, &mmr_q));

    // 3. Scalability Benchmark: 50,000 Track Candidates
    println!("\n----------------------------------------------------------------------");
    println!("50,000 CANDIDATE SCALABILITY BENCHMARK (TIMING ONLY)");
    println!("----------------------------------------------------------------------");
    let n_candidates = 50_000;
    println!("Generating {} candidates from replicated real embeddings...", n_candidates);

    let mut large_corpus = Vec::with_capacity(n_candidates * EMBEDDING_DIM);
    let reps = (n_candidates + n_tracks - 1) / n_tracks;
    for r in 0..reps {
        for i in 0..n_tracks {
            if large_corpus.len() >= n_candidates * EMBEDDING_DIM {
                break;
            }
            let offset = i * EMBEDDING_DIM;
            let noise = if r > 0 { 0.005 * (r as f32) } else { 0.0 };
            for d in 0..EMBEDDING_DIM {
                large_corpus.push(real_embs[offset + d] + noise);
            }
        }
    }

    // Warmup
    println!("Warming up over 50,000 vectors...");
    for _ in 0..3 {
        let scores = score_candidates(&large_corpus, n_candidates, &liked_centroid, Some(&skipped_centroid), 0.5);
        let _ = mmr_queue(&large_corpus, n_candidates, &scores, 10, 0.7, &[0, 1]);
    }

    // Timed Evaluation (20 iterations)
    println!("Benchmarking 20 timed queue generations over 50,000 tracks...");
    let mut times_scoring = Vec::with_capacity(20);
    let mut times_total = Vec::with_capacity(20);

    for _ in 0..20 {
        let t0 = Instant::now();
        let scores = score_candidates(&large_corpus, n_candidates, &liked_centroid, Some(&skipped_centroid), 0.5);
        let t_score = t0.elapsed();

        let _queue = mmr_queue(&large_corpus, n_candidates, &scores, 10, 0.7, &[0, 1]);
        let t_total = t0.elapsed();

        times_scoring.push(t_score.as_secs_f64() * 1000.0);
        times_total.push(t_total.as_secs_f64() * 1000.0);
    }

    let mean_score_ms = times_scoring.iter().sum::<f64>() / times_scoring.len() as f64;
    let min_score_ms = times_scoring.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_total_ms = times_total.iter().sum::<f64>() / times_total.len() as f64;
    let min_total_ms = times_total.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("\nLatency Summary (N = 50,000 tracks, 512-dim):");
    println!("  Vector Scoring (50k dot products + penalty): Mean = {:.2} ms | Min = {:.2} ms", mean_score_ms, min_score_ms);
    println!("  Total Autoplay Queue Generation (Score + MMR):Mean = {:.2} ms | Min = {:.2} ms", mean_total_ms, min_total_ms);
    println!("======================================================================\n");
}

fn mean_score(scores: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| scores[i]).sum::<f32>() / indices.len() as f32
}

fn extract_json_array(json: &str, key: &str) -> Vec<usize> {
    let mut result = Vec::new();
    let pattern = format!("\"{}\": [", key);
    if let Some(start) = json.find(&pattern) {
        let slice = &json[start + pattern.len()..];
        if let Some(end) = slice.find(']') {
            let inner = &slice[..end];
            for token in inner.split(',') {
                if let Ok(num) = token.trim().parse::<usize>() {
                    result.push(num);
                }
            }
        }
    }
    result
}
