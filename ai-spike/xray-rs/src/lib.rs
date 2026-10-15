//! Acoustic X-Ray DSP Analysis Engine in Native Rust.
//! Extracts BPM, Musical Key, RMS/Peak Energy, and Spectral Descriptors
//! (Centroid, Rolloff 85%, Flatness) without any machine learning dependencies.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

pub const PITCH_CLASSES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

// Temperley cognitive profiles for key estimation
const TEMPERLEY_MAJOR: [f32; 12] = [
    5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0,
];
const TEMPERLEY_MINOR: [f32; 12] = [
    5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0,
];

// Krumhansl-Schmuckler key profiles
const KRUMHANSL_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KRUMHANSL_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

#[derive(Debug, Clone, PartialEq)]
pub struct XRayFeatures {
    pub bpm: f32,
    pub key: String,
    pub key_confidence: f32,
    pub energy_rms: f32,
    pub energy_peak: f32,
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub spectral_flatness: f32,
}

pub fn analyze_audio(samples: &[f32], sample_rate: u32) -> XRayFeatures {
    if samples.is_empty() {
        return XRayFeatures {
            bpm: 0.0,
            key: "Unknown".to_string(),
            key_confidence: 0.0,
            energy_rms: 0.0,
            energy_peak: 0.0,
            spectral_centroid: 0.0,
            spectral_rolloff: 0.0,
            spectral_flatness: 0.0,
        };
    }

    // 1. Energy Features
    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    for &s in samples {
        let abs_s = s.abs();
        if abs_s > peak {
            peak = abs_s;
        }
        sum_sq += s * s;
    }
    let energy_rms = (sum_sq / samples.len() as f32).sqrt();
    let energy_peak = peak;

    // 2. STFT Spectral Analysis
    let n_fft = 2048;
    let hop_size = 512;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Precompute Hann window
    let mut hann = vec![0.0f32; n_fft];
    for i in 0..n_fft {
        hann[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos());
    }

    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_size + 1
    } else {
        0
    };

    let n_bins = n_fft / 2 + 1;
    let bin_freq_step = sample_rate as f32 / n_fft as f32;

    let mut prev_spec = vec![0.0f32; n_bins];
    let mut novelty_curve = Vec::with_capacity(n_frames);
    let mut chroma_accum = vec![0.0f32; 12];

    let mut mean_centroid = 0.0f32;
    let mut mean_rolloff = 0.0f32;
    let mut mean_flatness = 0.0f32;

    let mut fft_buffer = vec![Complex { re: 0.0f32, im: 0.0f32 }; n_fft];

    for f_idx in 0..n_frames {
        let offset = f_idx * hop_size;
        for i in 0..n_fft {
            fft_buffer[i] = Complex {
                re: samples[offset + i] * hann[i],
                im: 0.0,
            };
        }

        fft.process(&mut fft_buffer);

        // Compute magnitude spectrum
        let mut frame_spec = vec![0.0f32; n_bins];
        let mut frame_energy = 0.0f32;
        let mut weighted_freq_sum = 0.0f32;
        let mut log_sum = 0.0f32;

        for k in 0..n_bins {
            let re = fft_buffer[k].re;
            let im = fft_buffer[k].im;
            let mag = (re * re + im * im).sqrt();
            frame_spec[k] = mag;
            frame_energy += mag;

            let freq = k as f32 * bin_freq_step;
            weighted_freq_sum += freq * mag;

            let power = mag * mag + 1e-12;
            log_sum += power.ln();

            // Accumulate chroma for pitched frequencies [65 Hz, 2500 Hz]
            if freq >= 65.0 && freq <= 2500.0 {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                let pc = (midi.round() as i32).rem_euclid(12) as usize;
                let weight = 1.0 / freq.sqrt();
                chroma_accum[pc] += mag * weight;
            }
        }

        // Spectral Centroid
        let centroid = if frame_energy > 1e-6 {
            weighted_freq_sum / frame_energy
        } else {
            0.0
        };
        mean_centroid += centroid;

        // Spectral Rolloff (85% of total energy)
        let rolloff_threshold = frame_energy * 0.85;
        let mut cum_energy = 0.0f32;
        let mut rolloff_freq = 0.0f32;
        for k in 0..n_bins {
            cum_energy += frame_spec[k];
            if cum_energy >= rolloff_threshold {
                rolloff_freq = k as f32 * bin_freq_step;
                break;
            }
        }
        mean_rolloff += rolloff_freq;

        // Spectral Flatness (geometric mean / arithmetic mean)
        let geom_mean = (log_sum / n_bins as f32).exp();
        let arith_mean = frame_energy / n_bins as f32;
        let flatness = if arith_mean > 1e-6 {
            (geom_mean / arith_mean).min(1.0)
        } else {
            0.0
        };
        mean_flatness += flatness;

        // Spectral Flux Novelty
        let mut flux = 0.0f32;
        for k in 0..n_bins {
            let diff = frame_spec[k] - prev_spec[k];
            if diff > 0.0 {
                flux += diff;
            }
        }
        novelty_curve.push(flux);
        prev_spec = frame_spec;
    }

    if n_frames > 0 {
        mean_centroid /= n_frames as f32;
        mean_rolloff /= n_frames as f32;
        mean_flatness /= n_frames as f32;
    }

    // 3. BPM Estimation from Novelty Curve Autocorrelation
    let fps = sample_rate as f32 / hop_size as f32;
    let bpm = estimate_bpm(&novelty_curve, fps);

    // 4. Key Estimation from Chromagram & Temperley Profiles
    let (key, key_conf) = estimate_key(&chroma_accum);

    XRayFeatures {
        bpm,
        key,
        key_confidence: key_conf,
        energy_rms,
        energy_peak,
        spectral_centroid: mean_centroid,
        spectral_rolloff: mean_rolloff,
        spectral_flatness: mean_flatness,
    }
}

fn estimate_bpm(novelty: &[f32], fps: f32) -> f32 {
    if novelty.len() < 20 {
        return 120.0;
    }

    // Zero-mean the novelty curve
    let mean_nov = novelty.iter().sum::<f32>() / novelty.len() as f32;
    let centered: Vec<f32> = novelty.iter().map(|&x| x - mean_nov).collect();

    let min_bpm = 50.0f32;
    let max_bpm = 220.0f32;
    let min_lag = (fps * 60.0 / max_bpm).round() as usize;
    let max_lag = (fps * 60.0 / min_bpm).round() as usize;

    let mut best_lag = min_lag;
    let mut best_score = -1e9f32;
    let mut lag_corrs = vec![0.0f32; max_lag + 2];

    for lag in min_lag..=max_lag.min(centered.len() - 1) {
        let mut corr = 0.0f32;
        let count = centered.len() - lag;
        for i in 0..count {
            corr += centered[i] * centered[i + lag];
        }
        corr /= count as f32;
        if lag < lag_corrs.len() {
            lag_corrs[lag] = corr;
        }

        // Apply a gentle log-normal prior around 120 BPM
        let current_bpm = (fps * 60.0) / lag as f32;
        let log_ratio = (current_bpm / 120.0).log2();
        let prior = (-0.5 * (log_ratio / 0.55).powi(2)).exp();

        let score = corr * prior;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    let mut final_bpm = (fps * 60.0) / best_lag as f32;

    // Octave error correction: check octave double (half lag)
    let half_lag = best_lag / 2;
    if half_lag >= min_lag && final_bpm < 95.0 {
        let half_corr = lag_corrs[half_lag];
        let base_corr = lag_corrs[best_lag];
        if half_corr > 0.42 * base_corr {
            final_bpm *= 2.0;
        }
    } else if best_lag * 2 <= max_lag && final_bpm > 165.0 {
        let double_lag = best_lag * 2;
        let double_corr = lag_corrs[double_lag];
        let base_corr = lag_corrs[best_lag];
        if double_corr > 0.45 * base_corr {
            final_bpm *= 0.5;
        }
    }

    final_bpm
}

fn estimate_key(chroma: &[f32]) -> (String, f32) {
    let chroma_norm_factor = chroma.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if chroma_norm_factor < 1e-6 {
        return ("C major".to_string(), 0.0);
    }
    let norm_chroma: Vec<f32> = chroma.iter().map(|&x| x / chroma_norm_factor).collect();

    let temp_maj_norm = normalize_profile(&TEMPERLEY_MAJOR);
    let temp_min_norm = normalize_profile(&TEMPERLEY_MINOR);
    let krum_maj_norm = normalize_profile(&KRUMHANSL_MAJOR);
    let krum_min_norm = normalize_profile(&KRUMHANSL_MINOR);

    let mut best_score = -1e9f32;
    let mut best_key = "C major".to_string();

    for shift in 0..12 {
        // Test Major (Blended Temperley & Krumhansl)
        let mut score_maj = 0.0f32;
        for i in 0..12 {
            let prof_idx = (i + 12 - shift) % 12;
            let blended = 0.5 * (temp_maj_norm[prof_idx] + krum_maj_norm[prof_idx]);
            score_maj += norm_chroma[i] * blended;
        }
        if score_maj > best_score {
            best_score = score_maj;
            best_key = format!("{} major", PITCH_CLASSES[shift]);
        }

        // Test Minor (Blended Temperley & Krumhansl)
        let mut score_min = 0.0f32;
        for i in 0..12 {
            let prof_idx = (i + 12 - shift) % 12;
            let blended = 0.5 * (temp_min_norm[prof_idx] + krum_min_norm[prof_idx]);
            score_min += norm_chroma[i] * blended;
        }
        if score_min > best_score {
            best_score = score_min;
            best_key = format!("{} minor", PITCH_CLASSES[shift]);
        }
    }

    (best_key, best_score.max(0.0))
}

fn normalize_profile(prof: &[f32; 12]) -> [f32; 12] {
    let norm = prof.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let mut out = [0.0f32; 12];
    for i in 0..12 {
        out[i] = prof[i] / norm;
    }
    out
}

pub fn analyze_wav_file(path: &str) -> Result<XRayFeatures, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            let raw: Vec<i32> = reader.samples::<i32>().filter_map(Result::ok).collect();
            // Downmix to mono
            raw.chunks(channels)
                .map(|ch| ch.iter().map(|&s| s as f32 / max_val).sum::<f32>() / channels as f32)
                .collect()
        }
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().filter_map(Result::ok).collect();
            raw.chunks(channels)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                .collect()
        }
    };

    Ok(analyze_audio(&samples, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_and_flatness_silence() {
        let silence = vec![0.0f32; 44100];
        let features = analyze_audio(&silence, 44100);
        assert_eq!(features.energy_rms, 0.0);
        assert_eq!(features.energy_peak, 0.0);
    }

    #[test]
    fn test_sine_wave_features() {
        let sr = 44100;
        let freq = 440.0f32; // A4
        let mut sine = vec![0.0f32; sr as usize];
        for i in 0..sine.len() {
            sine[i] = 0.5 * (2.0 * PI * freq * i as f32 / sr as f32).sin();
        }

        let features = analyze_audio(&sine, sr);
        assert!(features.energy_rms > 0.3 && features.energy_rms < 0.4);
        assert!((features.energy_peak - 0.5).abs() < 1e-4);
        // Centroid of a pure 440Hz sine should be approximately 440 Hz
        assert!((features.spectral_centroid - 440.0).abs() < 50.0);
    }
}
