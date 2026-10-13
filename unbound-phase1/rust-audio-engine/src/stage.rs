//! Output stage: converts decoded audio (the file's sample rate / channel layout) into what
//! the output device wants. Kept separate from `lib.rs` so it can be unit-tested without a
//! sound card (`cargo test --lib`).
//!
//! Why this exists: rubato's `SincFixedIn` only accepts *exactly* `input_frames_next()`
//! frames per call (1024 here), but decoders hand us packets of arbitrary size (MP3: 1152,
//! FLAC: 4096, AAC: 1024, WAV: whatever). The first version of the engine used the first
//! 1024 frames of each packet and silently threw the rest away, which lost ~11% of every MP3
//! and ~75% of every FLAC that needed resampling. This stage keeps the leftover frames and
//! carries them into the next call, and flushes the tail at end-of-file.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Frames per rubato call. Larger = slightly more efficient, smaller = lower latency.
const CHUNK_FRAMES: usize = 1024;

pub struct OutputStage {
    src_rate: u32,
    dst_rate: u32,
    src_ch: usize,
    dst_ch: usize,
    ratio: f64,
    resampler: Option<SincFixedIn<f32>>,
    /// Per-source-channel frames waiting for a full rubato chunk.
    pending: Vec<Vec<f32>>,
    in_frames: u64,
    out_frames: u64,
}

impl OutputStage {
    pub fn new(src_rate: u32, src_ch: usize, dst_rate: u32, dst_ch: usize) -> Result<Self, String> {
        if src_rate == 0 || dst_rate == 0 || src_ch == 0 || dst_ch == 0 {
            return Err(format!(
                "invalid format: {src_rate}Hz/{src_ch}ch -> {dst_rate}Hz/{dst_ch}ch"
            ));
        }
        let mut s = OutputStage {
            src_rate,
            dst_rate,
            src_ch,
            dst_ch,
            ratio: dst_rate as f64 / src_rate as f64,
            resampler: None,
            pending: Vec::new(),
            in_frames: 0,
            out_frames: 0,
        };
        s.rebuild(src_ch)?;
        Ok(s)
    }

    fn rebuild(&mut self, src_ch: usize) -> Result<(), String> {
        self.src_ch = src_ch;
        self.pending = vec![Vec::new(); src_ch];
        self.in_frames = 0;
        self.out_frames = 0;
        if self.src_rate == self.dst_rate {
            self.resampler = None;
        } else {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            // NOTE: rubato 0.16 already compensates the filter delay (measured: an impulse
            // at input frame N comes out at N*ratio), so no output trimming is needed.
            self.resampler = Some(
                SincFixedIn::<f32>::new(self.ratio, 2.0, params, CHUNK_FRAMES, src_ch)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    pub fn is_resampling(&self) -> bool {
        self.resampler.is_some()
    }

    /// Forget all buffered audio (call after a seek).
    pub fn reset(&mut self) {
        for p in &mut self.pending {
            p.clear();
        }
        self.in_frames = 0;
        self.out_frames = 0;
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
    }

    /// Feed one decoded packet (interleaved, `channels` per frame). Converted audio is
    /// appended to `out` (interleaved, device layout).
    pub fn process(&mut self, interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
        if channels == 0 || interleaved.is_empty() {
            return;
        }
        if channels != self.src_ch {
            log::warn!(
                "[unbound-audio] packet has {channels} channels, expected {}; rebuilding stage",
                self.src_ch
            );
            if let Err(e) = self.rebuild(channels) {
                log::error!("[unbound-audio] stage rebuild failed: {e}");
                return;
            }
        }
        let frames = interleaved.len() / channels;

        if self.resampler.is_none() {
            for f in 0..frames {
                let base = f * channels;
                push_frame(|c| interleaved[base + c], channels, self.dst_ch, out);
            }
            return;
        }

        for f in 0..frames {
            for c in 0..channels {
                self.pending[c].push(interleaved[f * channels + c]);
            }
        }
        self.in_frames += frames as u64;
        self.run_full_chunks(out);
    }

    fn run_full_chunks(&mut self, out: &mut Vec<f32>) {
        let Some(rs) = self.resampler.as_mut() else { return };
        loop {
            let need = rs.input_frames_next();
            if self.pending[0].len() < need {
                break;
            }
            let result = {
                let input: Vec<&[f32]> = self.pending.iter().map(|v| &v[..need]).collect();
                rs.process(&input, None)
            };
            for p in &mut self.pending {
                p.drain(..need);
            }
            match result {
                Ok(planar) => emit_planar(&planar, &mut self.out_frames, None, self.dst_ch, out),
                Err(e) => log::warn!("[unbound-audio] resample error: {e}"),
            }
        }
    }

    /// End of stream: push out whatever is still buffered (including the filter's tail) so
    /// the last ~20 ms of the track aren't lost. Output length ends up at exactly
    /// round(total_input_frames * ratio).
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        let Some(rs) = self.resampler.as_mut() else { return };
        let target = (self.in_frames as f64 * self.ratio).round() as u64;

        if !self.pending[0].is_empty() {
            let result = {
                let input: Vec<&[f32]> = self.pending.iter().map(|v| v.as_slice()).collect();
                rs.process_partial(Some(&input), None)
            };
            if let Ok(planar) = result {
                emit_planar(&planar, &mut self.out_frames, Some(target), self.dst_ch, out);
            }
            for p in &mut self.pending {
                p.clear();
            }
        }
        if self.out_frames < target {
            // Push the filter's internal delay line out with silence.
            let result = rs.process_partial(None::<&[Vec<f32>]>, None);
            if let Ok(planar) = result {
                emit_planar(&planar, &mut self.out_frames, Some(target), self.dst_ch, out);
            }
        }
    }
}

/// Interleave planar resampler output into `out`, with an optional cap on the total number
/// of frames emitted.
fn emit_planar(
    planar: &[Vec<f32>],
    emitted: &mut u64,
    cap_total: Option<u64>,
    dst_ch: usize,
    out: &mut Vec<f32>,
) {
    let src_ch = planar.len();
    if src_ch == 0 {
        return;
    }
    let n = planar[0].len();
    for f in 0..n {
        if let Some(cap) = cap_total {
            if *emitted >= cap {
                break;
            }
        }
        push_frame(|c| planar[c][f], src_ch, dst_ch, out);
        *emitted += 1;
    }
}

/// Append one frame, converting `src_ch` -> `dst_ch` channels.
///  - same count: copy
///  - to mono: average all channels
///  - mono to N: put it in the first two channels (L+R), silence in the rest
///  - anything else (stereo -> 5.1, 5.1 -> stereo): keep the first `min` channels, silence the rest
///    (a proper surround downmix can come later; local music is almost always stereo)
fn push_frame(get: impl Fn(usize) -> f32, src_ch: usize, dst_ch: usize, out: &mut Vec<f32>) {
    if src_ch == dst_ch {
        for c in 0..dst_ch {
            out.push(get(c));
        }
    } else if dst_ch == 1 {
        let mut sum = 0.0;
        for c in 0..src_ch {
            sum += get(c);
        }
        out.push(sum / src_ch as f32);
    } else if src_ch == 1 {
        let v = get(0);
        for c in 0..dst_ch {
            out.push(if c < 2 { v } else { 0.0 });
        }
    } else {
        for c in 0..dst_ch {
            out.push(if c < src_ch { get(c) } else { 0.0 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_stereo(rate: u32, secs: f32, hz: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        let mut v = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = ((2.0 * std::f64::consts::PI * hz as f64 * i as f64 / rate as f64).sin() * 0.5) as f32;
            v.push(s);
            v.push(s);
        }
        v
    }

    fn feed_in_packets(st: &mut OutputStage, input: &[f32], ch: usize, packet_frames: &[usize]) -> Vec<f32> {
        let mut out = Vec::new();
        let (mut pos, mut i) = (0usize, 0usize);
        while pos < input.len() {
            let n = (packet_frames[i % packet_frames.len()] * ch).min(input.len() - pos);
            st.process(&input[pos..pos + n], ch, &mut out);
            pos += n;
            i += 1;
        }
        st.flush(&mut out);
        out
    }

    #[test]
    fn resampling_keeps_every_frame_for_mp3_flac_and_odd_packet_sizes() {
        // 1152 = MP3, 4096 = FLAC, 300 = smaller than a rubato chunk, 1024 = AAC
        for sizes in [vec![1152], vec![4096], vec![300], vec![1024], vec![1152, 4096, 300, 5000, 77]] {
            let input = sine_stereo(44_100, 3.0, 440.0);
            let mut st = OutputStage::new(44_100, 2, 48_000, 2).unwrap();
            let out = feed_in_packets(&mut st, &input, 2, &sizes);
            let in_frames = input.len() / 2;
            let want = (in_frames as f64 * 48_000.0 / 44_100.0).round() as i64;
            let got = (out.len() / 2) as i64;
            assert_eq!(got, want, "packet sizes {sizes:?}: frames out");
        }
    }

    #[test]
    fn resampled_sine_matches_an_ideal_sine_with_no_time_shift() {
        let input = sine_stereo(44_100, 3.0, 440.0);
        let mut st = OutputStage::new(44_100, 2, 48_000, 2).unwrap();
        let out = feed_in_packets(&mut st, &input, 2, &[1152]);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        // Compare against the mathematically ideal 48 kHz sine, frame by frame. Skip 300 frames
        // at each end: the test signal switches on/off abruptly there, and a band-limited
        // resampler legitimately rings around such an edge (real music fades in/out).
        // The resampler has a constant sub-frame timing offset (~0.08 frame = 1.7 us, inaudible),
        // so allow a fractional shift and require the remaining error to be tiny and uniform.
        let mut best = f64::MAX;
        let mut shift = -0.3f64;
        while shift <= 0.3 {
            let mut worst = 0.0f64;
            for k in 300..left.len() - 300 {
                let ideal = (2.0 * std::f64::consts::PI * 440.0 * (k as f64 + shift) / 48_000.0).sin() * 0.5;
                worst = worst.max((left[k] as f64 - ideal).abs());
            }
            best = best.min(worst);
            shift += 0.02;
        }
        assert!(best < 0.0003, "output deviates from an ideal sine by {best} (glitch or large time shift)");
    }

    #[test]
    fn same_rate_is_bit_exact_passthrough() {
        let input = sine_stereo(48_000, 0.5, 1000.0);
        let mut st = OutputStage::new(48_000, 2, 48_000, 2).unwrap();
        assert!(!st.is_resampling());
        let out = feed_in_packets(&mut st, &input, 2, &[1152]);
        assert_eq!(out, input);
    }

    #[test]
    fn channel_conversion() {
        let mut out = Vec::new();
        let mut st = OutputStage::new(48_000, 1, 48_000, 2).unwrap();
        st.process(&[0.25, -0.5], 1, &mut out);
        assert_eq!(out, vec![0.25, 0.25, -0.5, -0.5]); // mono -> L+R

        let mut out = Vec::new();
        let mut st = OutputStage::new(48_000, 2, 48_000, 1).unwrap();
        st.process(&[0.2, 0.6, -1.0, 1.0], 2, &mut out);
        assert_eq!(out, vec![0.4, 0.0]); // stereo -> mono average

        let mut out = Vec::new();
        let mut st = OutputStage::new(48_000, 2, 48_000, 6).unwrap();
        st.process(&[0.1, 0.2], 2, &mut out);
        assert_eq!(out, vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0]); // stereo -> 5.1: L,R only
    }

    #[test]
    fn reset_discards_buffered_audio() {
        let mut st = OutputStage::new(44_100, 2, 48_000, 2).unwrap();
        let mut out = Vec::new();
        st.process(&sine_stereo(44_100, 0.01, 440.0), 2, &mut out); // < one chunk: stays pending
        assert!(out.is_empty());
        st.reset();
        st.flush(&mut out);
        assert!(out.is_empty(), "flush after reset must not emit pre-reset audio");
    }

    #[test]
    fn rejects_invalid_formats() {
        assert!(OutputStage::new(0, 2, 48_000, 2).is_err());
        assert!(OutputStage::new(44_100, 0, 48_000, 2).is_err());
    }
}
