//! End-to-end behavioural tests of the real engine code against a *simulated* sound card
//! (see cpal-stub). Run with:  cargo test --release -- --test-threads=1 --nocapture
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::sync::Mutex;
use std::time::Duration;
use unbound_audio::*;

static SERIAL: Mutex<()> = Mutex::new(());
static CAPTURE: Mutex<Vec<f32>> = Mutex::new(Vec::new());
static CB_DELAY_MS: AtomicU64 = AtomicU64::new(0);
static CAP_CH: AtomicU64 = AtomicU64::new(2);
static CAP_RATE: AtomicU64 = AtomicU64::new(48_000);

extern "C" fn capture_cb(samples: *const f32, count: i32, rate: i32, ch: i32) {
    let s = unsafe { std::slice::from_raw_parts(samples, count as usize) };
    CAP_CH.store(ch as u64, SeqCst);
    CAP_RATE.store(rate as u64, SeqCst);
    CAPTURE.lock().unwrap().extend_from_slice(s);
    let d = CB_DELAY_MS.load(SeqCst);
    if d > 0 { std::thread::sleep(Duration::from_millis(d)); }
}

#[cfg(feature = "old")]
fn play(h: *mut c_void, path: &str) -> i32 { let c = CString::new(path).unwrap(); engine_play(h, c.as_ptr()); 0 }
#[cfg(not(feature = "old"))]
fn play(h: *mut c_void, path: &str) -> i32 { let c = CString::new(path).unwrap(); engine_play(h, c.as_ptr()) }

/// 44.1 kHz stereo 16-bit WAV: 440 Hz for 3 s, then 880 Hz for 3 s (continuous phase), amplitude 0.5.
fn make_wav(name: &str) -> String {
    let rate = 44_100u32; let secs = 6u32; let n = rate * secs;
    let mut pcm: Vec<u8> = Vec::new();
    let mut phase = 0.0f64;
    for i in 0..n {
        let f = if (i as f64) / (rate as f64) < 3.0 { 440.0 } else { 880.0 };
        phase += 2.0 * std::f64::consts::PI * f / rate as f64;
        let v = (phase.sin() * 0.5 * 32767.0) as i16;
        for _ in 0..2 { pcm.extend_from_slice(&v.to_le_bytes()); }
    }
    let mut w: Vec<u8> = Vec::new();
    w.extend_from_slice(b"RIFF"); w.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt "); w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&rate.to_le_bytes()); w.extend_from_slice(&(rate * 4).to_le_bytes());
    w.extend_from_slice(&4u16.to_le_bytes()); w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data"); w.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    w.extend_from_slice(&pcm);
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, w).unwrap();
    p.to_string_lossy().into_owned()
}

fn left(cap: &[f32]) -> Vec<f32> { let ch = CAP_CH.load(SeqCst) as usize; cap.iter().step_by(ch).copied().collect() }
fn freq(l: &[f32], start: usize, n: usize, rate: f32) -> f32 {
    if start + n > l.len() { return -1.0; }
    let w = &l[start..start + n]; let mut c = 0;
    for i in 1..w.len() { if w[i - 1] <= 0.0 && w[i] > 0.0 { c += 1; } }
    c as f32 / (n as f32 / rate)
}
fn reset(rate: u32, ch: u32) {
    CAPTURE.lock().unwrap().clear(); CB_DELAY_MS.store(0, SeqCst);
    cpal::DEVICE_RATE.store(rate, SeqCst); cpal::DEVICE_CH.store(ch, SeqCst);
    cpal::MAX_CALLBACK_MICROS.store(0, SeqCst);
}
fn sleep_ms(ms: u64) { std::thread::sleep(Duration::from_millis(ms)); }

fn full_play(rate: u32, ch: u32, label: &str) {
    reset(rate, ch);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new();
    engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    sleep_ms(6_900);
    engine_stop(h); engine_release(h);
    let cap = CAPTURE.lock().unwrap().clone();
    let want = (6 * rate * ch) as f64;
    let got = cap.len() as f64;
    let l = left(&cap); let r = rate as f32;
    let f1 = freq(&l, (0.5 * r) as usize, (0.4 * r) as usize, r);
    let f2 = freq(&l, (4.2 * r) as usize, (0.4 * r) as usize, r);
    let max_step = l.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    println!("[{label}] samples {got:.0} / expected {want:.0} ({:.1}% of expected) | tone@0.5s {f1:.0}Hz (want 440) | tone@4.2s {f2:.0}Hz (want 880) | largest sample jump {max_step:.3} (clean <= 0.065)", got / want * 100.0);
    assert!((got / want - 1.0).abs() < 0.01, "{label}: audio length wrong: {got} vs {want}");
    assert!((f1 - 440.0).abs() < 25.0, "{label}: first tone {f1}");
    assert!((f2 - 880.0).abs() < 40.0, "{label}: second tone {f2}");
    assert!(max_step < 0.07, "{label}: discontinuity (dropped/glitched audio) {max_step}");
}

#[test] fn t1_resampled_44100_to_48000_stereo() { let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); full_play(48_000, 2, "44.1k->48k stereo"); }
#[test] fn t2_no_resample_44100_device() { let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); full_play(44_100, 2, "44.1k->44.1k stereo"); }
#[test] fn t3_stereo_to_mono_device() { let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); full_play(48_000, 1, "44.1k->48k mono device"); }

#[test]
fn t4_seek_discards_stale_audio_and_reports_true_position() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    sleep_ms(1_000);
    let len0 = CAPTURE.lock().unwrap().len();
    engine_seek_to(h, 4_000);
    sleep_ms(400);
    let pos = engine_current_position_ms(h);
    sleep_ms(300);
    engine_stop(h); engine_release(h);
    let cap = CAPTURE.lock().unwrap().clone();
    let l = left(&cap[len0..]);
    let win = 960; // 20 ms
    let mut stale_ms = -1.0;
    let mut i = 0;
    while i + win < l.len() { if freq(&l, i, win, 48_000.0) > 700.0 { stale_ms = i as f32 / 48.0; break; } i += 240; }
    println!("[seek] old-tone audio still heard after seek: {stale_ms:.0} ms (clean <= 150) | position 400ms after seek-to-4000: {pos} ms (want ~4400)");
    assert!(stale_ms >= 0.0 && stale_ms <= 150.0, "stale audio after seek: {stale_ms} ms");
    assert!((4_200..=4_700).contains(&pos), "position after seek {pos}");
}

#[test]
fn t5_slow_pcm_consumer_cannot_stall_the_audio_thread() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    CB_DELAY_MS.store(150, SeqCst);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    sleep_ms(2_000);
    let pos = engine_current_position_ms(h);
    engine_stop(h); engine_release(h);
    let worst = cpal::MAX_CALLBACK_MICROS.load(SeqCst);
    println!("[audio-thread] slowest audio callback with a 150 ms-per-chunk consumer: {worst} us (clean < 5000) | position after 2.0 s: {pos} ms (want ~2000)");
    assert!(worst < 5_000, "audio callback blocked for {worst} us");
    assert!((1_800..=2_200).contains(&pos), "playback fell behind wall clock: {pos}");
}

#[test]
fn t6_position_follows_clock_and_pause_holds() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new();
    assert_eq!(play(h, &wav), 0);
    sleep_ms(1_500);
    let a = engine_current_position_ms(h);
    engine_pause(h); sleep_ms(50); let p1 = engine_current_position_ms(h); sleep_ms(500); let p2 = engine_current_position_ms(h);
    engine_resume(h); sleep_ms(500); let b = engine_current_position_ms(h);
    engine_stop(h); engine_release(h);
    println!("[position] after 1.5s: {a} ms | paused {p1} -> {p2} | 0.5s after resume: {b} ms");
    assert!((1_350..=1_650).contains(&a), "position {a}");
    assert_eq!(p1, p2);
    assert!((b - p2 - 500).abs() < 120, "resume position {b} vs {p2}");
}

#[cfg(not(feature = "old"))]
#[test]
fn t7_bad_file_reports_an_error_code() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let h = engine_new();
    let r = play(h, "/definitely/not/here.mp3");
    let state = engine_state(h);
    engine_release(h);
    println!("[errors] playing a missing file returned code {r} (want negative), state {state} (want 4 ERROR)");
    assert!(r < 0);
    assert_eq!(state, 4);
}

#[test]
fn t8_seek_after_end_of_track_restarts_playback() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    let dur = engine_duration_ms(h);
    assert!((5_900..=6_100).contains(&dur), "duration should be ~6000ms: got {dur}");
    assert_eq!(engine_state(h), 1, "state should be 1 (PLAYING)");

    sleep_ms(7_300); // file is 6 s long: it has ended
    let state_ended = engine_state(h);
    assert_eq!(state_ended, 3, "state should be 3 (ENDED) after track completes: got {state_ended}");

    let len0 = CAPTURE.lock().unwrap().len();
    engine_seek_to(h, 1_000);
    let state_resumed = engine_state(h);
    assert_eq!(state_resumed, 1, "state after seek past ENDED should return to 1 (PLAYING)");

    sleep_ms(900);
    let pos = engine_current_position_ms(h);
    engine_stop(h);
    assert_eq!(engine_state(h), 0, "state after stop should be 0 (IDLE)");
    engine_release(h);

    let cap = CAPTURE.lock().unwrap().clone();
    let got = cap.len() - len0;
    println!("[eof-seek] samples received after seeking back from the end: {got} (want > 40000) | position {pos} ms (want ~1900)");
    assert!(got > 40_000, "no audio after seek past EOF");
    assert!((1_500..=1_950).contains(&pos));
}

#[test]
fn t12_engine_state_and_duration_lifecycle() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new();
    assert_eq!(engine_state(h), 0, "initial state should be 0 IDLE");
    assert_eq!(engine_duration_ms(h), -1, "initial duration should be -1");

    assert_eq!(play(h, &wav), 0);
    let dur = engine_duration_ms(h);
    println!("[state-dur] duration reported: {dur} ms (want ~6000)");
    assert!((5_900..=6_100).contains(&dur));
    assert_eq!(engine_state(h), 1, "state after play should be 1 PLAYING");

    engine_pause(h);
    assert_eq!(engine_state(h), 2, "state after pause should be 2 PAUSED");
    engine_resume(h);
    assert_eq!(engine_state(h), 1, "state after resume should be 1 PLAYING");

    sleep_ms(7_300);
    let state_at_end = engine_state(h);
    println!("[state-dur] state after track end: {state_at_end} (want 3 ENDED)");
    assert_eq!(state_at_end, 3, "state should be 3 ENDED");

    engine_seek_to(h, 2_000);
    assert_eq!(engine_state(h), 1, "seek after ENDED should return to 1 PLAYING");

    engine_stop(h);
    assert_eq!(engine_state(h), 0, "state after stop should be 0 IDLE");
    engine_release(h);
}

#[test]
fn t9_release_while_playing_does_not_hang() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    sleep_ms(300);
    let t = std::time::Instant::now();
    engine_release(h);
    println!("[lifecycle] release took {:?}", t.elapsed());
    assert!(t.elapsed() < Duration::from_secs(2));
}

#[test]
fn t10_stress_rapid_play_stop_and_seek_spam() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    let t = std::time::Instant::now();
    for i in 0..40 {
        assert_eq!(play(h, &wav), 0);
        sleep_ms(20 + (i % 5) * 15);
        engine_seek_to(h, ((i * 733) % 5_500) as i64);
        if i % 3 == 0 { engine_pause(h); sleep_ms(10); engine_resume(h); }
        if i % 2 == 0 { engine_stop(h); }
    }
    // seek spam from another thread while playing
    assert_eq!(play(h, &wav), 0);
    let hp = h as usize;
    let spam = std::thread::spawn(move || { for k in 0..300 { engine_seek_to(hp as *mut c_void, ((k * 977) % 5_800) as i64); std::thread::sleep(Duration::from_millis(3)); } });
    spam.join().unwrap();
    sleep_ms(300);
    let pos = engine_current_position_ms(h);
    engine_release(h);
    println!("[stress] 40 play/stop/seek cycles + 300 seeks survived in {:?}, final position {pos} ms", t.elapsed());
    assert!(pos >= 0 && t.elapsed() < Duration::from_secs(20));
}

#[test]
fn t11_detaching_the_pcm_callback_waits_for_the_in_flight_call() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner()); reset(48_000, 2);
    CB_DELAY_MS.store(200, SeqCst);
    let wav = make_wav("unbound_two_tone.wav");
    let h = engine_new(); engine_set_pcm_callback(h, Some(capture_cb));
    assert_eq!(play(h, &wav), 0);
    sleep_ms(500);
    let t = std::time::Instant::now();
    engine_set_pcm_callback(h, None);
    let waited = t.elapsed();
    let len_after_detach = CAPTURE.lock().unwrap().len();
    sleep_ms(500);
    let len_later = CAPTURE.lock().unwrap().len();
    engine_stop(h); engine_release(h);
    println!("[detach] set_pcm_callback(None) returned after {waited:?}; chunks delivered after return: {}", len_later - len_after_detach);
    assert_eq!(len_later, len_after_detach, "callback fired after it was detached");
}
