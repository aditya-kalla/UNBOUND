use std::time::Instant;
use xray_rs::analyze_wav_file;

fn main() {
    println!("======================================================================");
    println!("UNBOUND AI SPIKE - TASK T5: RUST ACOUSTIC X-RAY 4-MINUTE BENCHMARK");
    println!("======================================================================");

    let test_paths = [
        "../data/real/audio/USER_145_Nuvvunte Chaley.wav",
        "ai-spike/data/real/audio/USER_145_Nuvvunte Chaley.wav",
        "data/real/audio/USER_145_Nuvvunte Chaley.wav"
    ];
    let test_path = test_paths.iter().find(|p| std::path::Path::new(p).exists())
        .expect("Error: Target 4-minute track not found in expected paths");

    println!("Target Track: {}", test_path);
    let reader = hound::WavReader::open(test_path).expect("Failed to open WAV");
    let spec = reader.spec();
    let n_samples = reader.len() as f64 / spec.channels as f64;
    let duration_sec = n_samples / spec.sample_rate as f64;
    println!(
        "Track Specs: {} Hz, {} channels, {:.2} seconds ({:.2} minutes)",
        spec.sample_rate,
        spec.channels,
        duration_sec,
        duration_sec / 60.0
    );

    // Run Warmup
    println!("\nWarming up...");
    let _ = analyze_wav_file(test_path).expect("Analysis failed");

    // Timed Evaluation (5 iterations)
    println!("Benchmarking 5 timed iterations...");
    let mut run_times = Vec::new();
    let mut last_features = None;

    for i in 1..=5 {
        let start = Instant::now();
        let features = analyze_wav_file(test_path).expect("Analysis failed");
        let elapsed = start.elapsed();
        run_times.push(elapsed.as_secs_f64());
        println!(
            "  Run {}: {:.2} ms (RTF: {:.4}x)",
            i,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() / duration_sec
        );
        last_features = Some(features);
    }

    let mean_time = run_times.iter().sum::<f64>() / run_times.len() as f64;
    let min_time = run_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let rtf = mean_time / duration_sec;

    let f = last_features.unwrap();
    println!("\n----------------------------------------------------------------------");
    println!("EXTRACTED ACOUSTIC FEATURES (4-MINUTE REAL TRACK):");
    println!("----------------------------------------------------------------------");
    println!("- Estimated BPM:         {:.1} BPM", f.bpm);
    println!("- Estimated Musical Key: {} (Confidence: {:.4})", f.key, f.key_confidence);
    println!("- RMS Energy:            {:.4} (-{:.1} dBFS)", f.energy_rms, -20.0 * f.energy_rms.log10());
    println!("- Peak Energy:           {:.4} (-{:.1} dBFS)", f.energy_peak, -20.0 * f.energy_peak.log10());
    println!("- Spectral Centroid:     {:.1} Hz", f.spectral_centroid);
    println!("- Spectral Rolloff (85%):{:.1} Hz", f.spectral_rolloff);
    println!("- Spectral Flatness:     {:.4}", f.spectral_flatness);

    println!("\n----------------------------------------------------------------------");
    println!("PERFORMANCE BENCHMARK SUMMARY (MEASURED):");
    println!("----------------------------------------------------------------------");
    println!("Audio Duration:    {:.2} seconds (4.00 min)", duration_sec);
    println!("Mean Wall Time:    {:.2} ms ({:.3} seconds)", mean_time * 1000.0, mean_time);
    println!("Min Wall Time:     {:.2} ms ({:.3} seconds)", min_time * 1000.0, min_time);
    println!("Real-Time Factor:  {:.5}x (processed {:.1}x faster than real-time)", rtf, 1.0 / rtf);
    println!("======================================================================\n");
}
