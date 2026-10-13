//! UNBOUND native audio engine — Phase 1.
//!
//! Scope: load a local file, play / pause / resume / stop / seek, and expose a raw-PCM
//! callback hook that stays unused until Phase 4/5 (ONNX feature extraction, Synesthesia
//! shaders). No EQ, WSOLA, crossfade or gapless yet.
//!
//! Threads:
//!   * decode thread  (`unbound-decode`)  symphonia decode -> `OutputStage` (resample / channel
//!     remap) -> SPSC ring buffer. Blocks (condvar) when the ring is full; never drops samples.
//!     After end-of-file it parks (instead of exiting) so a later seek can restart playback.
//!   * audio thread   (cpal callback, real-time) pops from the ring, zero-fills on underrun,
//!     counts played frames, and copies ~32 ms chunks into a bounded queue. It never calls
//!     foreign code and (in steady state) never allocates or blocks on anything slow.
//!   * dispatch thread (`unbound-pcm-dispatch`) takes chunks off that queue and calls the
//!     registered PCM callback (Kotlin/JNA). A slow or stuck consumer only makes chunks get
//!     dropped; it can never glitch the audio.
//!   * caller threads  play / pause / resume / stop / seek / position queries.
//!
//! PCM callback rules: it is invoked from the dispatch thread, one call at a time. The
//! `samples` pointer is only valid for the duration of the call. Never call back into the
//! engine from inside the callback (stop/release/set_pcm_callback wait for the dispatch
//! thread -> deadlock). `engine_set_pcm_callback` blocks until any in-flight call to the
//! previous callback has returned (at most one call), so after it returns the old callback
//! object can be freed safely.

mod stage;
pub mod meta;

use std::ffi::{c_void, CStr};
use std::fs::File;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapCons, HeapRb,
};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::{Time, TimeBase};

use stage::OutputStage;

/// Function pointer shape the Kotlin/JNA side registers. Must match
/// `NativeAudioEngineBridge.PcmCallback` exactly.
type PcmCallback = extern "C" fn(samples: *const f32, sample_count: i32, sample_rate: i32, channels: i32);

/// Ring buffer capacity in samples (all channels): ~500 ms of 48 kHz stereo.
const RING_CAPACITY: usize = 48_000;
/// Length of one PCM chunk handed to the consumer.
const PCM_CHUNK_MS: usize = 32;
/// Chunks that can be in flight between the audio thread and the dispatch thread.
const PCM_QUEUE_DEPTH: usize = 8;

/// Why `engine_play` failed. The numeric codes are part of the C ABI (mirrored in Kotlin's
/// `PlayResult`) — only ever append new ones.
#[derive(Clone, Copy, Debug)]
enum PlayError {
    Open,
    Probe,
    NoTrack,
    Decoder,
    NoDevice,
    DeviceConfig,
    Stream,
    Resampler,
    Internal,
}

impl PlayError {
    fn code(self) -> i32 {
        match self {
            PlayError::Open => -1,
            PlayError::Probe => -2,
            PlayError::NoTrack => -3,
            PlayError::Decoder => -4,
            PlayError::NoDevice => -5,
            PlayError::DeviceConfig => -6,
            PlayError::Stream => -7,
            PlayError::Resampler => -8,
            PlayError::Internal => -9,
        }
    }
}

/// The registered PCM callback plus whether a call into it is currently running.
struct PcmSlot {
    callback: Option<PcmCallback>,
    in_call: bool,
}

/// State shared between the decode thread, the audio thread, the dispatch thread and the
/// control methods.
struct SharedState {
    /// Tells the decode thread to exit.
    stop_flag: AtomicBool,
    /// Set by `pause()`, cleared by `resume()`.
    paused: AtomicBool,
    /// Frames actually sent to the device since the start of the file (or since the last
    /// seek, offset to the seek target). Position in ms = frames * 1000 / output_rate.
    /// Counting frames (not ms) avoids the rounding drift of adding truncated ms per callback.
    position_frames: AtomicU64,
    /// Output device sample rate for the current playback.
    output_rate: AtomicU32,
    /// Decode thread sleeps here when the ring is full (or after EOF); the audio thread and
    /// `seek_to` / `stop` wake it.
    ring_space_available: Condvar,
    ring_cv_lock: Mutex<()>,
    /// Registered PCM callback. The lock is only held to read/write the slot, never during
    /// the call itself (a long call must not starve `set_pcm_callback`). `pcm_idle` is
    /// signalled when a call finishes so a detach can wait for it.
    pcm: Mutex<PcmSlot>,
    pcm_idle: Condvar,
    seek_requested: AtomicBool,
    seek_target_ms: AtomicI64,
    /// 0 IDLE, 1 PLAYING, 2 PAUSED, 3 ENDED, 4 ERROR
    state: AtomicI32,
    duration_ms: AtomicI64,
    at_eof: AtomicBool,
}

pub struct Engine {
    shared: Arc<SharedState>,
    /// Serialises play/stop/release so two threads can't tear down / rebuild at once.
    control: Mutex<()>,
    /// The cpal stream. Kept alive so playback continues; dropped on stop().
    stream: Mutex<Option<cpal::Stream>>,
    decode_thread: Mutex<Option<JoinHandle<()>>>,
    dispatch_thread: Mutex<Option<JoinHandle<()>>>,
}

impl Engine {
    fn new() -> Self {
        Engine {
            shared: Arc::new(SharedState {
                stop_flag: AtomicBool::new(false),
                paused: AtomicBool::new(false),
                position_frames: AtomicU64::new(0),
                output_rate: AtomicU32::new(48_000),
                ring_space_available: Condvar::new(),
                ring_cv_lock: Mutex::new(()),
                pcm: Mutex::new(PcmSlot { callback: None, in_call: false }),
                pcm_idle: Condvar::new(),
                seek_requested: AtomicBool::new(false),
                seek_target_ms: AtomicI64::new(0),
                state: AtomicI32::new(0),
                duration_ms: AtomicI64::new(-1),
                at_eof: AtomicBool::new(false),
            }),
            control: Mutex::new(()),
            stream: Mutex::new(None),
            decode_thread: Mutex::new(None),
            dispatch_thread: Mutex::new(None),
        }
    }

    fn play(&self, uri: &str) -> Result<(), PlayError> {
        let res = (|| -> Result<(), PlayError> {
            let _control = self.control.lock().unwrap();

            // Tear down any existing playback first.
            self.stop_internal();
            self.shared.stop_flag.store(false, Ordering::SeqCst);
            self.shared.paused.store(false, Ordering::SeqCst);
            self.shared.position_frames.store(0, Ordering::SeqCst);
            self.shared.seek_requested.store(false, Ordering::SeqCst);
            self.shared.at_eof.store(false, Ordering::SeqCst);
            self.shared.duration_ms.store(-1, Ordering::SeqCst);

            // ── 1. Open the file and probe for a supported codec ──────────
            let file = File::open(uri).map_err(|e| {
                log::error!("[unbound-audio] Failed to open '{uri}': {e}");
                PlayError::Open
            })?;
            let mss = MediaSourceStream::new(Box::new(file), Default::default());

            let mut hint = Hint::new();
            if let Some(ext) = std::path::Path::new(uri).extension().and_then(|e| e.to_str()) {
                hint.with_extension(ext);
            }

            let probed = symphonia::default::get_probe()
                .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
                .map_err(|e| {
                    log::error!("[unbound-audio] Failed to probe '{uri}': {e}");
                    PlayError::Probe
                })?;
            let format_reader = probed.format;

            let track = format_reader
                .tracks()
                .iter()
                .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                .cloned()
                .ok_or_else(|| {
                    log::error!("[unbound-audio] No supported audio track in '{uri}'");
                    PlayError::NoTrack
                })?;

            let track_id = track.id;
            let time_base = track.codec_params.time_base;
            let n_frames = track.codec_params.n_frames;
            let duration_ms = match (n_frames, time_base) {
                (Some(frames), Some(tb)) if frames > 0 => {
                    let time = tb.calc_time(frames);
                    (time.seconds * 1000 + (time.frac * 1000.0) as u64) as i64
                }
                _ => -1,
            };
            self.shared.duration_ms.store(duration_ms, Ordering::SeqCst);

            let decoded_sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
            let decoded_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);

            let decoder = symphonia::default::get_codecs()
                .make(&track.codec_params, &DecoderOptions::default())
                .map_err(|e| {
                    log::error!("[unbound-audio] Failed to create decoder: {e}");
                    PlayError::Decoder
                })?;

            // ── 2. Output device ──────────────────────────────────────────
            let host = cpal::default_host();
            let device = host.default_output_device().ok_or_else(|| {
                log::error!("[unbound-audio] No default output device found");
                PlayError::NoDevice
            })?;
            let default_config = device.default_output_config().map_err(|e| {
                log::error!("[unbound-audio] Failed to get default output config: {e}");
                PlayError::DeviceConfig
            })?;
            let device_sample_rate = default_config.sample_rate().0;
            let device_channels = default_config.channels();
            self.shared.output_rate.store(device_sample_rate, Ordering::SeqCst);

            log::info!(
                "[unbound-audio] Decoded: {decoded_sample_rate}Hz {decoded_channels}ch → Device: {device_sample_rate}Hz {device_channels}ch"
            );

            // ── 3. Resample / channel-remap stage ─────────────────────────
            let stage = OutputStage::new(
                decoded_sample_rate,
                decoded_channels,
                device_sample_rate,
                device_channels as usize,
            )
            .map_err(|e| {
                log::error!("[unbound-audio] Failed to create output stage: {e}");
                PlayError::Resampler
            })?;
            if stage.is_resampling() {
                log::info!("[unbound-audio] Resampling {decoded_sample_rate}Hz -> {device_sample_rate}Hz");
            }

            // ── 4. Ring buffer + PCM hand-off queues ──────────────────────
            let (producer, consumer) = HeapRb::<f32>::new(RING_CAPACITY).split();
            let consumer: Arc<Mutex<HeapCons<f32>>> = Arc::new(Mutex::new(consumer));

            let ch = device_channels as usize;
            let chunk_threshold = device_sample_rate as usize * ch * PCM_CHUNK_MS / 1000;
            let chunk_capacity = chunk_threshold * 2 + 16_384;
            let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<f32>>(PCM_QUEUE_DEPTH);
            let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<f32>>(PCM_QUEUE_DEPTH);
            for _ in 0..PCM_QUEUE_DEPTH {
                let _ = pool_tx.send(Vec::with_capacity(chunk_capacity));
            }
            let mut accumulator: Vec<f32> = Vec::with_capacity(chunk_capacity);

            // ── 5. Build the cpal output stream ───────────────────────────
            let stream_config = cpal::StreamConfig {
                channels: device_channels,
                sample_rate: cpal::SampleRate(device_sample_rate),
                buffer_size: cpal::BufferSize::Default,
            };
            let shared_for_stream = Arc::clone(&self.shared);
            let consumer_for_stream = Arc::clone(&consumer);

            let stream = device
                .build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        if shared_for_stream.paused.load(Ordering::Relaxed) {
                            data.fill(0.0);
                            return;
                        }

                        let to_read;
                        {
                            let mut cons = consumer_for_stream.lock().unwrap();
                            let mut n = data.len().min(cons.occupied_len());
                            n -= n % ch;
                            if n > 0 {
                                cons.pop_slice(&mut data[..n]);
                                shared_for_stream
                                    .position_frames
                                    .fetch_add((n / ch) as u64, Ordering::Relaxed);
                            }
                            to_read = n;
                        }
                        if to_read < data.len() {
                            data[to_read..].fill(0.0);
                        }
                        if to_read == 0 && shared_for_stream.at_eof.load(Ordering::Relaxed) {
                            shared_for_stream.state.store(3, Ordering::Relaxed);
                        }
                        shared_for_stream.ring_space_available.notify_one();

                        if to_read > 0 {
                            accumulator.extend_from_slice(&data[..to_read]);
                            if accumulator.len() >= chunk_threshold {
                                if let Ok(mut buf) = pool_rx.try_recv() {
                                    buf.clear();
                                    buf.extend_from_slice(&accumulator);
                                    let _ = chunk_tx.try_send(buf);
                                }
                                accumulator.clear();
                            }
                        }
                    },
                    move |err| {
                        log::error!("[unbound-audio] cpal stream error: {err}");
                    },
                    None,
                )
                .map_err(|e| {
                    log::error!("[unbound-audio] Failed to build output stream: {e}");
                    PlayError::Stream
                })?;

            // ── 6. PCM dispatch thread ────────────────────────────────────
            let shared_for_dispatch = Arc::clone(&self.shared);
            let dispatch_handle = thread::Builder::new()
                .name("unbound-pcm-dispatch".into())
                .spawn(move || {
                    while let Ok(buf) = chunk_rx.recv() {
                        let cb = {
                            let mut slot = shared_for_dispatch.pcm.lock().unwrap();
                            let cb = slot.callback;
                            slot.in_call = cb.is_some();
                            cb
                        };
                        if let Some(cb) = cb {
                            cb(
                                buf.as_ptr(),
                                buf.len() as i32,
                                device_sample_rate as i32,
                                device_channels as i32,
                            );
                            let mut slot = shared_for_dispatch.pcm.lock().unwrap();
                            slot.in_call = false;
                            shared_for_dispatch.pcm_idle.notify_all();
                        }
                        let _ = pool_tx.try_send(buf);
                    }
                })
                .map_err(|e| {
                    log::error!("[unbound-audio] Failed to spawn dispatch thread: {e}");
                    PlayError::Internal
                })?;

            stream.play().map_err(|e| {
                log::error!("[unbound-audio] Failed to start stream: {e}");
                PlayError::Stream
            })?;

            *self.stream.lock().unwrap() = Some(stream);
            *self.dispatch_thread.lock().unwrap() = Some(dispatch_handle);

            // ── 7. Decode thread ──────────────────────────────────────────
            let shared_for_decode = Arc::clone(&self.shared);
            let consumer_for_decode = Arc::clone(&consumer);
            let decode_handle = thread::Builder::new()
                .name("unbound-decode".into())
                .spawn(move || {
                    decode_loop(
                        shared_for_decode,
                        producer,
                        consumer_for_decode,
                        format_reader,
                        decoder,
                        track_id,
                        time_base,
                        stage,
                    );
                });
            match decode_handle {
                Ok(h) => *self.decode_thread.lock().unwrap() = Some(h),
                Err(e) => {
                    log::error!("[unbound-audio] Failed to spawn decode thread: {e}");
                    self.stop_internal();
                    return Err(PlayError::Internal);
                }
            }

            self.shared.state.store(1, Ordering::SeqCst);
            log::info!("[unbound-audio] Playback started for '{uri}'");
            Ok(())
        })();

        if res.is_err() {
            self.shared.state.store(4, Ordering::SeqCst);
        }
        res
    }

    fn pause(&self) {
        if self.shared.state.load(Ordering::SeqCst) == 1 {
            self.shared.state.store(2, Ordering::SeqCst);
        }
        self.shared.paused.store(true, Ordering::SeqCst);
        if let Some(ref stream) = *self.stream.lock().unwrap() {
            let _ = stream.pause();
        }
    }

    fn resume(&self) {
        if self.shared.state.load(Ordering::SeqCst) == 2 {
            self.shared.state.store(1, Ordering::SeqCst);
        }
        self.shared.paused.store(false, Ordering::SeqCst);
        if let Some(ref stream) = *self.stream.lock().unwrap() {
            let _ = stream.play();
        }
    }

    fn stop(&self) {
        let _control = self.control.lock().unwrap();
        self.stop_internal();
    }

    /// Signal the decode thread, drop the stream (stops the audio callback and closes the
    /// chunk queue), then join both helper threads. Caller must hold `control`.
    fn stop_internal(&self) {
        self.shared.stop_flag.store(true, Ordering::SeqCst);
        self.shared.paused.store(false, Ordering::SeqCst);
        self.shared.at_eof.store(false, Ordering::SeqCst);
        self.shared.state.store(0, Ordering::SeqCst);
        self.shared.duration_ms.store(-1, Ordering::SeqCst);
        self.shared.ring_space_available.notify_all();

        *self.stream.lock().unwrap() = None;

        if let Some(h) = self.decode_thread.lock().unwrap().take() {
            let _ = h.join();
        }
        if let Some(h) = self.dispatch_thread.lock().unwrap().take() {
            let _ = h.join();
        }
        self.shared.position_frames.store(0, Ordering::SeqCst);
    }

    fn seek_to(&self, position_ms: i64) {
        let target = position_ms.max(0);
        self.shared.seek_target_ms.store(target, Ordering::SeqCst);
        self.shared.seek_requested.store(true, Ordering::SeqCst);
        self.shared.at_eof.store(false, Ordering::SeqCst);
        if self.shared.state.load(Ordering::SeqCst) == 3 {
            self.shared.state.store(1, Ordering::SeqCst);
        }
        let rate = self.shared.output_rate.load(Ordering::Relaxed) as u64;
        self.shared
            .position_frames
            .store(target as u64 * rate / 1000, Ordering::SeqCst);
        self.shared.ring_space_available.notify_all();
    }

    fn current_position_ms(&self) -> i64 {
        let rate = self.shared.output_rate.load(Ordering::Relaxed).max(1) as u64;
        (self.shared.position_frames.load(Ordering::Relaxed) * 1000 / rate) as i64
    }

    fn duration_ms(&self) -> i64 {
        self.shared.duration_ms.load(Ordering::Relaxed)
    }

    fn state(&self) -> i32 {
        self.shared.state.load(Ordering::Relaxed)
    }

    /// Swap the callback. If a call into the old one is running, wait for it to return, so
    /// that once this returns the old callback object can be freed.
    fn set_pcm_callback(&self, callback: Option<PcmCallback>) {
        let mut slot = self.shared.pcm.lock().unwrap();
        slot.callback = callback;
        while slot.in_call {
            slot = self.shared.pcm_idle.wait(slot).unwrap();
        }
    }
}

/// Push `samples` into the ring, sleeping while it's full. Returns `false` if playback was
/// stopped or a seek arrived (the rest of this audio is stale and must be dropped).
fn push_blocking(shared: &SharedState, producer: &mut impl Producer<Item = f32>, samples: &[f32]) -> bool {
    let mut offset = 0;
    while offset < samples.len() {
        if shared.stop_flag.load(Ordering::SeqCst) || shared.seek_requested.load(Ordering::SeqCst) {
            return false;
        }
        let vacant = producer.vacant_len();
        if vacant == 0 {
            let guard = shared.ring_cv_lock.lock().unwrap();
            let _ = shared
                .ring_space_available
                .wait_timeout(guard, Duration::from_millis(50))
                .unwrap();
            continue;
        }
        let n = (samples.len() - offset).min(vacant);
        producer.push_slice(&samples[offset..offset + n]);
        offset += n;
    }
    true
}

/// End of file: flush the resampler's tail into the ring.
fn finish_stream(
    shared: &SharedState,
    producer: &mut impl Producer<Item = f32>,
    stage: &mut OutputStage,
    out: &mut Vec<f32>,
) {
    out.clear();
    stage.flush(out);
    push_blocking(shared, producer, out);
    out.clear();
    shared.at_eof.store(true, Ordering::SeqCst);
    log::info!("[unbound-audio] End of track reached");
}

/// The decode thread's main loop. Runs until `stop_flag`; after end-of-file it parks so a
/// seek can restart playback.
#[allow(clippy::too_many_arguments)]
fn decode_loop(
    shared: Arc<SharedState>,
    mut producer: impl Producer<Item = f32>,
    consumer: Arc<Mutex<HeapCons<f32>>>,
    mut format_reader: Box<dyn FormatReader>,
    mut decoder: Box<dyn Decoder>,
    track_id: u32,
    time_base: Option<TimeBase>,
    mut stage: OutputStage,
) {
    let rate = shared.output_rate.load(Ordering::Relaxed) as u64;
    let mut out: Vec<f32> = Vec::with_capacity(16_384);
    let mut at_eof = false;
    let mut bad_packets = 0u32;

    loop {
        if shared.stop_flag.load(Ordering::SeqCst) {
            break;
        }

        // ── Seek ──────────────────────────────────────────────────────
        if shared.seek_requested.swap(false, Ordering::SeqCst) {
            let target_ms = shared.seek_target_ms.load(Ordering::SeqCst).max(0);
            let seek_to = SeekTo::Time {
                time: Time::new(target_ms as u64 / 1000, (target_ms % 1000) as f64 / 1000.0),
                track_id: Some(track_id),
            };
            match format_reader.seek(SeekMode::Coarse, seek_to) {
                Ok(seeked) => {
                    decoder.reset();
                    stage.reset();
                    out.clear();
                    // Coarse seeks land on a packet boundary; report where we really are.
                    let landed_ms = match time_base {
                        Some(tb) => {
                            let t = tb.calc_time(seeked.actual_ts);
                            t.seconds * 1000 + (t.frac * 1000.0) as u64
                        }
                        None => target_ms as u64,
                    };
                    // Discard everything already buffered (it's from before the seek) and
                    // restart the position counter, atomically w.r.t. the audio thread.
                    {
                        let mut cons = consumer.lock().unwrap();
                        cons.clear();
                        shared
                            .position_frames
                            .store(landed_ms * rate / 1000, Ordering::SeqCst);
                    }
                    at_eof = false;
                    shared.at_eof.store(false, Ordering::SeqCst);
                    if shared.state.load(Ordering::SeqCst) == 3 {
                        shared.state.store(1, Ordering::SeqCst);
                    }
                    bad_packets = 0;
                    log::info!("[unbound-audio] Seeked to {target_ms}ms (landed at {landed_ms}ms)");
                }
                Err(e) => log::error!("[unbound-audio] Seek failed: {e}"),
            }
        }

        // ── Parked after EOF: wait for a seek or stop ────────────────
        if at_eof {
            let guard = shared.ring_cv_lock.lock().unwrap();
            let _ = shared
                .ring_space_available
                .wait_timeout(guard, Duration::from_millis(50))
                .unwrap();
            continue;
        }

        // ── Next packet ───────────────────────────────────────────────
        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                finish_stream(&shared, &mut producer, &mut stage, &mut out);
                at_eof = true;
                continue;
            }
            Err(SymError::ResetRequired) => {
                log::warn!("[unbound-audio] Stream requires reset (unsupported); ending track");
                finish_stream(&shared, &mut producer, &mut stage, &mut out);
                at_eof = true;
                continue;
            }
            Err(e) => {
                bad_packets += 1;
                if bad_packets > 50 {
                    log::error!("[unbound-audio] Too many read errors, ending track: {e}");
                    finish_stream(&shared, &mut producer, &mut stage, &mut out);
                    at_eof = true;
                } else {
                    log::warn!("[unbound-audio] Read error (skipping packet): {e}");
                }
                continue;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("[unbound-audio] Frame decode error (skipping): {e}");
                continue;
            }
        };
        bad_packets = 0;

        // ── Convert to interleaved f32, resample / remap, push ───────
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.frames() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        out.clear();
        stage.process(sample_buf.samples(), channels, &mut out);
        // `false` = stop or seek pending; the loop top handles both.
        let _ = push_blocking(&shared, &mut producer, &out);
    }

    log::info!("[unbound-audio] Decode thread exiting");
}

// ---------------------------------------------------------------------
// C ABI — every function here is what the Kotlin JNA interface calls.
// Keep this list and NativeAudioEngineBridge.kt's `RustAudioLib` in sync.
// ---------------------------------------------------------------------

fn engine_ref<'a>(handle: *mut c_void) -> Option<&'a Engine> {
    if handle.is_null() {
        None
    } else {
        Some(unsafe { &*(handle as *mut Engine) })
    }
}

#[no_mangle]
pub extern "C" fn engine_new() -> *mut c_void {
    std::panic::catch_unwind(|| {
        let _ = env_logger::try_init();
        Box::into_raw(Box::new(Engine::new())) as *mut c_void
    })
    .unwrap_or_else(|e| {
        log::error!("[unbound-audio] Panic in engine_new: {:?}", e);
        std::ptr::null_mut()
    })
}

/// Returns 0 on success or a negative `PlayError` code.
#[no_mangle]
pub extern "C" fn engine_play(handle: *mut c_void, uri: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        let Some(engine) = engine_ref(handle) else {
            return PlayError::Internal.code();
        };
        if uri.is_null() {
            return PlayError::Internal.code();
        }
        let uri = unsafe { CStr::from_ptr(uri) }.to_string_lossy();
        match engine.play(&uri) {
            Ok(()) => 0,
            Err(e) => e.code(),
        }
    })
    .unwrap_or_else(|e| {
        log::error!("[unbound-audio] Panic in engine_play: {:?}", e);
        PlayError::Internal.code()
    })
}

#[no_mangle]
pub extern "C" fn engine_pause(handle: *mut c_void) {
    let _ = std::panic::catch_unwind(|| {
        if let Some(e) = engine_ref(handle) {
            e.pause();
        }
    });
}

#[no_mangle]
pub extern "C" fn engine_resume(handle: *mut c_void) {
    let _ = std::panic::catch_unwind(|| {
        if let Some(e) = engine_ref(handle) {
            e.resume();
        }
    });
}

#[no_mangle]
pub extern "C" fn engine_stop(handle: *mut c_void) {
    let _ = std::panic::catch_unwind(|| {
        if let Some(e) = engine_ref(handle) {
            e.stop();
        }
    });
}

#[no_mangle]
pub extern "C" fn engine_seek_to(handle: *mut c_void, position_ms: i64) {
    let _ = std::panic::catch_unwind(|| {
        if let Some(e) = engine_ref(handle) {
            e.seek_to(position_ms);
        }
    });
}

#[no_mangle]
pub extern "C" fn engine_current_position_ms(handle: *mut c_void) -> i64 {
    std::panic::catch_unwind(|| {
        engine_ref(handle).map_or(-1, |e| e.current_position_ms())
    })
    .unwrap_or_else(|e| {
        log::error!("[unbound-audio] Panic in engine_current_position_ms: {:?}", e);
        -1
    })
}

#[no_mangle]
pub extern "C" fn engine_duration_ms(handle: *mut c_void) -> i64 {
    std::panic::catch_unwind(|| {
        engine_ref(handle).map_or(-1, |e| e.duration_ms())
    })
    .unwrap_or_else(|e| {
        log::error!("[unbound-audio] Panic in engine_duration_ms: {:?}", e);
        -1
    })
}

#[no_mangle]
pub extern "C" fn engine_state(handle: *mut c_void) -> i32 {
    std::panic::catch_unwind(|| {
        engine_ref(handle).map_or(4, |e| e.state())
    })
    .unwrap_or_else(|e| {
        log::error!("[unbound-audio] Panic in engine_state: {:?}", e);
        4
    })
}

#[no_mangle]
pub extern "C" fn engine_set_pcm_callback(handle: *mut c_void, callback: Option<PcmCallback>) {
    let _ = std::panic::catch_unwind(|| {
        if let Some(e) = engine_ref(handle) {
            e.set_pcm_callback(callback);
        }
    });
}

#[no_mangle]
pub extern "C" fn engine_release(handle: *mut c_void) {
    let _ = std::panic::catch_unwind(|| {
        if handle.is_null() {
            return;
        }
        // Stop playback (joins all threads) before freeing.
        unsafe { &*(handle as *mut Engine) }.stop();
        unsafe {
            drop(Box::from_raw(handle as *mut Engine));
        }
    });
}

/// Returns the ABI version of this library. The Kotlin bridge checks this
/// right after `Native.load` and throws if it doesn't match the expected
/// version, catching stale DLL copies immediately.
#[no_mangle]
pub extern "C" fn engine_abi_version() -> i32 {
    3
}

/// Test-only function that forces a panic inside an extern "C" function to prove
/// catch_unwind safely catches it without aborting the process.
#[no_mangle]
pub extern "C" fn engine_test_panic() -> i32 {
    std::panic::catch_unwind(|| {
        panic!("Deliberate test panic inside extern C function");
    })
    .unwrap_or(-99)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panic_safety_survives_process() {
        let code = engine_test_panic();
        assert_eq!(code, -99);
    }
}
