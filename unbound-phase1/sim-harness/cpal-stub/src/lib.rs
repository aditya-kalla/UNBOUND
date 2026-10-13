//! TEST-ONLY stand-in for the `cpal` API surface that unbound-audio uses.
//! Simulates a sound card: calls the data callback every 10 ms of audio, in real time,
//! from its own thread, and records how long each callback took.
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub static DEVICE_RATE: AtomicU32 = AtomicU32::new(48_000);
pub static DEVICE_CH: AtomicU32 = AtomicU32::new(2);
/// Longest single data-callback invocation seen, in microseconds.
pub static MAX_CALLBACK_MICROS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
pub struct SampleRate(pub u32);
#[derive(Clone, Copy, Debug)]
pub enum BufferSize { Default }
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig { pub channels: u16, pub sample_rate: SampleRate, pub buffer_size: BufferSize }
pub struct SupportedStreamConfig { channels: u16, rate: SampleRate }
impl SupportedStreamConfig {
    pub fn sample_rate(&self) -> SampleRate { self.rate }
    pub fn channels(&self) -> u16 { self.channels }
}
pub struct OutputCallbackInfo;

macro_rules! err { ($n:ident) => {
    #[derive(Debug)] pub struct $n;
    impl std::fmt::Display for $n { fn fmt(&self, f:&mut std::fmt::Formatter)->std::fmt::Result{ write!(f, stringify!($n)) } }
}}
err!(BuildStreamError); err!(PlayStreamError); err!(PauseStreamError); err!(StreamError); err!(DefaultStreamConfigError);

pub struct Host; pub struct Device;
pub fn default_host() -> Host { Host }

struct Ctl { playing: AtomicBool, stop: AtomicBool }
pub struct Stream { ctl: Arc<Ctl>, handle: Option<thread::JoinHandle<()>>, _not_send: PhantomData<*mut ()> }

pub mod traits {
    use super::*;
    pub trait HostTrait { fn default_output_device(&self) -> Option<Device>; }
    pub trait DeviceTrait {
        fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError>;
        fn build_output_stream<D, E>(&self, config: &StreamConfig, data: D, err: E, timeout: Option<Duration>)
            -> Result<Stream, BuildStreamError>
        where D: FnMut(&mut [f32], &OutputCallbackInfo) + Send + 'static, E: FnMut(StreamError) + Send + 'static;
    }
    pub trait StreamTrait { fn play(&self) -> Result<(), PlayStreamError>; fn pause(&self) -> Result<(), PauseStreamError>; }
}
use traits::*;

impl HostTrait for Host { fn default_output_device(&self) -> Option<Device> { Some(Device) } }
impl DeviceTrait for Device {
    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Ok(SupportedStreamConfig { channels: DEVICE_CH.load(Ordering::SeqCst) as u16, rate: SampleRate(DEVICE_RATE.load(Ordering::SeqCst)) })
    }
    fn build_output_stream<D, E>(&self, config: &StreamConfig, mut data: D, _err: E, _t: Option<Duration>)
        -> Result<Stream, BuildStreamError>
    where D: FnMut(&mut [f32], &OutputCallbackInfo) + Send + 'static, E: FnMut(StreamError) + Send + 'static {
        let ctl = Arc::new(Ctl { playing: AtomicBool::new(false), stop: AtomicBool::new(false) });
        let c2 = Arc::clone(&ctl);
        let frames = config.sample_rate.0 as usize / 100; // 10 ms per callback
        let ch = config.channels as usize;
        let handle = thread::spawn(move || {
            let mut buf = vec![0.0f32; frames * ch];
            let period = Duration::from_millis(10);
            let mut next = Instant::now();
            while !c2.stop.load(Ordering::SeqCst) {
                next += period;
                if c2.playing.load(Ordering::SeqCst) {
                    let t = Instant::now();
                    data(&mut buf, &OutputCallbackInfo);
                    MAX_CALLBACK_MICROS.fetch_max(t.elapsed().as_micros() as u64, Ordering::SeqCst);
                }
                let now = Instant::now();
                if next > now { thread::sleep(next - now); } else { next = now; }
            }
        });
        Ok(Stream { ctl, handle: Some(handle), _not_send: PhantomData })
    }
}
impl StreamTrait for Stream {
    fn play(&self) -> Result<(), PlayStreamError> { self.ctl.playing.store(true, Ordering::SeqCst); Ok(()) }
    fn pause(&self) -> Result<(), PauseStreamError> { self.ctl.playing.store(false, Ordering::SeqCst); Ok(()) }
}
impl Drop for Stream {
    fn drop(&mut self) { self.ctl.stop.store(true, Ordering::SeqCst); if let Some(h) = self.handle.take() { let _ = h.join(); } }
}
