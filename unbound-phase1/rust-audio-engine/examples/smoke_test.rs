use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use unbound_audio::{
    engine_abi_version, engine_current_position_ms, engine_new, engine_pause, engine_play,
    engine_release, engine_resume, engine_seek_to, engine_set_pcm_callback, engine_stop,
};

static PCM_CHUNKS_RECEIVED: AtomicUsize = AtomicUsize::new(0);

extern "C" fn on_pcm_chunk(samples: *const f32, count: i32, sample_rate: i32, channels: i32) {
    let chunk_idx = PCM_CHUNKS_RECEIVED.fetch_add(1, Ordering::SeqCst);
    if chunk_idx % 20 == 0 || chunk_idx < 3 {
        let first_val = if !samples.is_null() && count > 0 {
            unsafe { *samples }
        } else {
            0.0
        };
        println!(
            "  [PCM Callback] chunk #{chunk_idx}: {count} samples, {sample_rate}Hz, {channels}ch (first sample: {first_val:.4})"
        );
    }
}

fn main() {
    println!("=== UNBOUND Audio Engine Smoke Test ===");

    let test_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| r"C:\Windows\Media\Alarm01.wav".to_string());

    println!("Target audio file: {}", test_file);
    if !std::path::Path::new(&test_file).exists() {
        eprintln!("Error: Audio file not found at '{}'", test_file);
        std::process::exit(1);
    }

    println!("[1/7] Creating engine handle...");
    let abi = engine_abi_version();
    println!("  ABI version: {abi}");
    assert_eq!(abi, 3, "Expected ABI version 3, got {abi}");
    let engine = engine_new();
    assert!(!engine.is_null(), "engine_new returned null");
    println!("  Engine handle created: {:p}", engine);

    println!("[2/7] Registering PCM callback...");
    engine_set_pcm_callback(engine, Some(on_pcm_chunk));
    println!("  PCM callback registered.");

    println!("[3/7] Playing audio file...");
    let c_path = CString::new(test_file.clone()).expect("CString conversion failed");
    engine_play(engine, c_path.as_ptr());

    println!("[4/7] Monitoring playback for 1.5 seconds...");
    for _ in 0..15 {
        thread::sleep(Duration::from_millis(100));
        let pos = engine_current_position_ms(engine);
        let chunks = PCM_CHUNKS_RECEIVED.load(Ordering::SeqCst);
        println!("  Position: {pos}ms | PCM chunks received: {chunks}");
    }

    println!("[5/7] Testing Pause & Resume...");
    println!("  Pausing...");
    engine_pause(engine);
    let pos_at_pause = engine_current_position_ms(engine);
    thread::sleep(Duration::from_millis(500));
    let pos_after_wait = engine_current_position_ms(engine);
    println!("  Paused pos: {pos_at_pause}ms -> After 500ms: {pos_after_wait}ms (delta: {}ms)", pos_after_wait - pos_at_pause);

    println!("  Resuming...");
    engine_resume(engine);
    thread::sleep(Duration::from_millis(500));
    let pos_after_resume = engine_current_position_ms(engine);
    println!("  Resumed pos: {pos_after_resume}ms");

    println!("[6/7] Testing Seek (to 200ms)...");
    engine_seek_to(engine, 200);
    thread::sleep(Duration::from_millis(500));
    let pos_after_seek = engine_current_position_ms(engine);
    println!("  Position after seek: {pos_after_seek}ms");

    println!("[7/7] Stopping and releasing engine...");
    engine_stop(engine);
    engine_release(engine);
    println!("  Engine released cleanly.");

    let total_chunks = PCM_CHUNKS_RECEIVED.load(Ordering::SeqCst);
    println!("\n=== Smoke Test Summary ===");
    println!("Total PCM chunks received: {}", total_chunks);
    if total_chunks > 0 {
        println!("Status: SUCCESS! Audio engine decoded, streamed, and invoked callbacks properly.");
    } else {
        println!("Status: WARNING - No PCM chunks received (check audio output device).");
    }
}
