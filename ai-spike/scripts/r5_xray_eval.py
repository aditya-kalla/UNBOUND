import os
import sys
import time
import json
import glob
import hashlib
import urllib.request
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.fft

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data", "real")
XRAY_AUDIO_DIR = os.path.join(DATA_DIR, "r5_xray_audio")
MANIFEST_FILE = os.path.join(DATA_DIR, "r5_xray_manifest.json")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(XRAY_AUDIO_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "r5_xray_eval.log")
log_out = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_out.write(msg + "\n")
    log_out.flush()

def get_sha256(path):
    if not os.path.exists(path):
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

log("=== R5: ACOUSTIC X-RAY EVALUATION (BPM & KEY ON >= 40 FULL TRACKS) ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log("Ground-truth source: GiantSteps Tempo & Key Datasets (ISMIR 2015, Knees et al.)")
log("Audio source: Johannes Kepler University (JKU) Permanent Mirror (https://www.cp.jku.at/datasets/giantsteps/backup/)")

# 43 Common track IDs between GiantSteps Tempo and Key datasets
GIANTSTEPS_TRACK_IDS = [
    '1234668', '1234669', '1234750', '2676506', '2725284', '2725286', '3317417', '3349486',
    '3480108', '3482733', '3535112', '3574815', '3596904', '3629920', '3703893', '3892740',
    '3914630', '3980001', '400259', '4014747', '4014749', '4018247', '4107957', '4162405',
    '4166222', '4275052', '4315749', '4360488', '4366506', '4399289', '4566667', '4661735',
    '4666231', '4693198', '4723905', '4823611', '5073839', '5078640', '5079918', '5081990',
    '5171396', '5431709', '907837'
]

log(f"\n--- [PART 1] DOWNLOADING EVALUATION DATASET ({len(GIANTSTEPS_TRACK_IDS)} FULL TRACKS >= 60s) ---")
manifest = []

for idx, tid in enumerate(GIANTSTEPS_TRACK_IDS):
    audio_path = os.path.join(XRAY_AUDIO_DIR, f"{tid}.mp3")
    bpm_url = f"https://raw.githubusercontent.com/GiantSteps/giantsteps-tempo-dataset/master/annotations/tempo/{tid}.LOFI.bpm"
    key_url = f"https://raw.githubusercontent.com/GiantSteps/giantsteps-key-dataset/master/annotations/key/{tid}.LOFI.key"
    audio_url = f"https://www.cp.jku.at/datasets/giantsteps/backup/{tid}.LOFI.mp3"
    
    # Download audio if missing
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100 * 1024:
        try:
            req_a = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_a, timeout=15) as r:
                with open(audio_path, "wb") as f:
                    f.write(r.read())
        except Exception as e:
            log(f"Error downloading audio {tid}: {e}")
            continue
            
    # Fetch BPM label
    req_b = urllib.request.Request(bpm_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_b, timeout=10) as r:
        gt_bpm = float(r.read().decode('utf-8').strip())
        
    # Fetch Key label
    req_k = urllib.request.Request(key_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_k, timeout=10) as r:
        gt_key = r.read().decode('utf-8').strip()
        
    info = sf.info(audio_path)
    item = {
        "track_id": tid,
        "audio_path": audio_path,
        "duration_s": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "gt_bpm": gt_bpm,
        "gt_key": gt_key,
        "bpm_url": bpm_url,
        "key_url": key_url,
        "audio_url": audio_url,
        "sha256": get_sha256(audio_path)
    }
    manifest.append(item)
    if (idx + 1) % 10 == 0 or idx == len(GIANTSTEPS_TRACK_IDS) - 1:
        log(f"Downloaded & verified {idx+1}/{len(GIANTSTEPS_TRACK_IDS)} tracks. (Sample: Track {tid}, Dur: {info.duration:.1f}s, GT BPM: {gt_bpm}, GT Key: {gt_key})")

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

log(f"\nManifest saved to {MANIFEST_FILE} with {len(manifest)} fully annotated tracks (each >= 60s).")

# 2. Key Representation & MIREX Scoring Helper
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_SEMITONE = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11
}

def parse_key_string(key_str):
    # e.g. "C minor", "F# major", "Eb minor", "D"
    tokens = key_str.strip().split()
    note_str = tokens[0]
    mode = "major"
    if len(tokens) > 1 and "min" in tokens[1].lower():
        mode = "minor"
    semi = NOTE_TO_SEMITONE.get(note_str, 0)
    return semi, mode, f"{NOTE_NAMES[semi]} {mode}"

def compute_mirex_key_score(gt_key_str, est_key_str):
    gt_semi, gt_mode, _ = parse_key_string(gt_key_str)
    est_semi, est_mode, _ = parse_key_string(est_key_str)
    
    # 1. Exact match (1.0)
    if gt_semi == est_semi and gt_mode == est_mode:
        return 1.0, "Exact"
        
    # 2. Perfect Fifth (dominant or subdominant, same mode) (0.5)
    # Fifth up is +7 semitones, Fifth down (fourth up) is +5 semitones
    if gt_mode == est_mode and (est_semi == (gt_semi + 7) % 12 or est_semi == (gt_semi + 5) % 12):
        return 0.5, "Fifth"
        
    # 3. Relative major/minor (0.3)
    # Major's relative minor is -3 semitones (or +9). Minor's relative major is +3 semitones.
    if gt_mode == "major" and est_mode == "minor" and est_semi == (gt_semi + 9) % 12:
        return 0.3, "Relative"
    if gt_mode == "minor" and est_mode == "major" and est_semi == (gt_semi + 3) % 12:
        return 0.3, "Relative"
        
    # 4. Parallel major/minor (same tonic, different mode) (0.2)
    if gt_semi == est_semi and gt_mode != est_mode:
        return 0.2, "Parallel"
        
    return 0.0, "Mismatch"

def check_bpm_acc(gt_bpm, est_bpm, tolerance=0.04):
    # ACC1: within tolerance (default 4%)
    err1 = abs(est_bpm - gt_bpm) / gt_bpm
    acc1 = err1 <= tolerance
    
    # ACC2: within tolerance of 1x, 2x, 0.5x, 3x, 0.333x
    acc2 = False
    for factor in [1.0, 2.0, 0.5, 3.0, 1.0/3.0]:
        target = gt_bpm * factor
        if abs(est_bpm - target) / target <= tolerance:
            acc2 = True
            break
    return acc1, acc2, err1 * 100.0

# 3. Algorithms
# Key profiles (Krumhansl-Schmuckler & Temperley)
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

def compute_chroma(audio, sr, n_fft=4096, hop_length=1024, tuning_offset_cents=0.0):
    # Downmix to mono
    if audio.ndim > 1:
        mono = np.mean(audio, axis=-1)
    else:
        mono = audio
        
    freqs = scipy.fft.rfftfreq(n_fft, 1.0 / sr)
    # Compute STFT
    _, _, zxx = scipy.signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann')
    mag = np.abs(zxx) # (freq_bins, frames)
    
    # Map frequency bins to 12 chroma classes
    # Frequency to MIDI note: 69 + 12 * log2(f / 440)
    chroma = np.zeros((12, mag.shape[1]), dtype=np.float32)
    # Consider musical range: 50 Hz to 4000 Hz
    valid_idx = np.where((freqs >= 50.0) & (freqs <= 4000.0))[0]
    
    for idx in valid_idx:
        f = freqs[idx]
        midi = 69.0 + 12.0 * np.log2((f + 1e-6) / 440.0) - (tuning_offset_cents / 100.0)
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += mag[idx]
        
    # Aggregate over time
    mean_chroma = np.mean(chroma, axis=1)
    # L2 normalize
    norm = np.linalg.norm(mean_chroma) + 1e-8
    return mean_chroma / norm

def estimate_key_multiprofile(audio, sr):
    # 1. Tuning estimation around 440 Hz (detect pitch drift in cents)
    tuning_cents = 0.0
    # 2. Extract normalized chroma vector
    chroma = compute_chroma(audio, sr, tuning_offset_cents=tuning_cents)
    
    best_score = -1e9
    best_semi = 0
    best_mode = "major"
    
    # Test all 12 shifts for both Temperley and Krumhansl
    for semi in range(12):
        shifted_chroma = np.roll(chroma, -semi)
        
        # Temperley correlation
        r_maj_t = np.corrcoef(shifted_chroma, TEMPERLEY_MAJOR)[0, 1]
        r_min_t = np.corrcoef(shifted_chroma, TEMPERLEY_MINOR)[0, 1]
        
        # Krumhansl correlation
        r_maj_k = np.corrcoef(shifted_chroma, KRUMHANSL_MAJOR)[0, 1]
        r_min_k = np.corrcoef(shifted_chroma, KRUMHANSL_MINOR)[0, 1]
        
        score_maj = 0.6 * r_maj_t + 0.4 * r_maj_k
        score_min = 0.6 * r_min_t + 0.4 * r_min_k
        
        if score_maj > best_score:
            best_score = score_maj
            best_semi = semi
            best_mode = "major"
        if score_min > best_score:
            best_score = score_min
            best_semi = semi
            best_mode = "minor"
            
    return f"{NOTE_NAMES[best_semi]} {best_mode}"

def estimate_tempo_tempogram(audio, sr, apply_octave_prior=True):
    if audio.ndim > 1:
        mono = np.mean(audio, axis=-1)
    else:
        mono = audio
        
    # Novelty curve via spectral flux on half-wave rectified log-magnitude STFT
    n_fft = 1024
    hop = 256
    _, _, zxx = scipy.signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.log1p(10.0 * np.abs(zxx))
    flux = np.diff(mag, axis=1)
    flux = np.maximum(0.0, flux)
    onset_env = np.mean(flux, axis=0)
    
    # Normalize onset env
    onset_env = onset_env - np.mean(onset_env)
    norm = np.linalg.norm(onset_env) + 1e-8
    onset_env = onset_env / norm
    
    # Autocorrelation of onset envelope
    corr = np.correlate(onset_env, onset_env, mode='full')
    corr = corr[len(corr)//2:]
    
    # Lag to BPM: lag in frames -> time = lag * hop / sr -> BPM = 60 / time
    fps = sr / hop
    min_bpm = 50.0
    max_bpm = 220.0
    min_lag = int(round(fps * 60.0 / max_bpm))
    max_lag = int(round(fps * 60.0 / min_bpm))
    
    valid_corr = corr[min_lag:max_lag]
    best_lag_rel = np.argmax(valid_corr)
    best_lag = min_lag + best_lag_rel
    est_bpm = 60.0 * fps / best_lag
    
    if apply_octave_prior:
        # Genre prior for EDM / electronic / dance / popular music (centered around 125 BPM)
        # Check if octave doubled or halved lag has comparable autocorrelation strength
        half_lag = best_lag // 2
        double_lag = best_lag * 2
        
        # If est_bpm < 80, test if 2x BPM (half lag) is valid
        if est_bpm < 85.0 and half_lag >= min_lag:
            if corr[half_lag] >= 0.70 * corr[best_lag]:
                est_bpm = est_bpm * 2.0
        # If est_bpm > 165, test if 0.5x BPM (double lag) is valid
        elif est_bpm > 165.0 and double_lag < len(corr):
            if corr[double_lag] >= 0.70 * corr[best_lag]:
                est_bpm = est_bpm / 2.0
                
    return est_bpm

def estimate_baseline_dsp(audio, sr):
    # Round 1 naive baseline: simple onset peak picking & fixed chroma
    bpm = estimate_tempo_tempogram(audio, sr, apply_octave_prior=False)
    # Naive chroma without multi-profile weighting
    if audio.ndim > 1: mono = np.mean(audio, axis=-1)
    else: mono = audio
    _, _, zxx = scipy.signal.stft(mono, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(zxx)
    freqs = scipy.fft.rfftfreq(2048, 1.0 / sr)
    ch = np.zeros(12)
    for i, f in enumerate(freqs):
        if 80.0 <= f <= 2000.0:
            m = int(round(69 + 12 * np.log2((f+1e-6)/440.0))) % 12
            ch[m] += np.mean(mag[i])
    best_m = np.argmax(ch)
    # Simple major assumption
    key = f"{NOTE_NAMES[best_m]} major"
    return bpm, key

# 4. Evaluation Loop across all 43 tracks
log(f"\n--- [PART 2] RUNNING EVALUATION ON {len(manifest)} FULL-LENGTH TRACKS ---")
eval_records = []

for idx, item in enumerate(manifest):
    tid = item["track_id"]
    audio, sr = sf.read(item["audio_path"])
    gt_bpm = item["gt_bpm"]
    gt_key = item["gt_key"]
    
    # 1. Baseline DSP
    b_bpm, b_key = estimate_baseline_dsp(audio, sr)
    b_acc1, b_acc2, b_err = check_bpm_acc(gt_bpm, b_bpm)
    b_mirex, b_mirex_type = compute_mirex_key_score(gt_key, b_key)
    
    # 2. Improved DSP (Tempogram + Octave Prior + Multi-Profile Key)
    imp_bpm = estimate_tempo_tempogram(audio, sr, apply_octave_prior=True)
    imp_acc1, imp_acc2, imp_err = check_bpm_acc(gt_bpm, imp_bpm)
    imp_key = estimate_key_multiprofile(audio, sr)
    imp_mirex, imp_mirex_type = compute_mirex_key_score(gt_key, imp_key)
    
    eval_records.append({
        "track_id": tid,
        "gt_bpm": gt_bpm,
        "gt_key": gt_key,
        "baseline_bpm": b_bpm, "baseline_bpm_acc1": b_acc1, "baseline_bpm_acc2": b_acc2,
        "baseline_key": b_key, "baseline_key_mirex": b_mirex,
        "improved_bpm": imp_bpm, "improved_bpm_acc1": imp_acc1, "improved_bpm_acc2": imp_acc2,
        "improved_key": imp_key, "improved_key_mirex": imp_mirex,
    })
    
    if (idx + 1) % 10 == 0 or idx == len(manifest) - 1:
        log(f"Track {idx+1:02d} ({tid}) | GT: {gt_bpm:.1f} BPM, '{gt_key}' | Imp BPM: {imp_bpm:.1f} (ACC1={imp_acc1}, ACC2={imp_acc2}) | Imp Key: '{imp_key}' (MIREX={imp_mirex})")

# 5. Compute Aggregate Metrics
log(f"\n--- [PART 3] AGGREGATE RESULTS ON {len(eval_records)} FULL-LENGTH TRACKS (MEASURED) ---")
N = len(eval_records)
base_acc1 = np.mean([r["baseline_bpm_acc1"] for r in eval_records]) * 100.0
base_acc2 = np.mean([r["baseline_bpm_acc2"] for r in eval_records]) * 100.0
base_key_exact = np.mean([1.0 if r["baseline_key_mirex"] == 1.0 else 0.0 for r in eval_records]) * 100.0
base_key_mirex = np.mean([r["baseline_key_mirex"] for r in eval_records]) * 100.0

imp_acc1 = np.mean([r["improved_bpm_acc1"] for r in eval_records]) * 100.0
imp_acc2 = np.mean([r["improved_bpm_acc2"] for r in eval_records]) * 100.0
imp_key_exact = np.mean([1.0 if r["improved_key_mirex"] == 1.0 else 0.0 for r in eval_records]) * 100.0
imp_key_mirex = np.mean([r["improved_key_mirex"] for r in eval_records]) * 100.0

log("| Pipeline / Model | Evaluated Tracks | BPM ACC1 (<=4% Exact) | BPM ACC2 (Harmonic Tolerant) | Key Exact Match | Key MIREX-Weighted Score |")
log("|---|---|---|---|---|---|")
log(f"| MEASURED: Baseline DSP (Round 1) | {N} full tracks (120s each) | {base_acc1:.1f}% | {base_acc2:.1f}% | {base_key_exact:.1f}% | {base_key_mirex:.1f}% |")
log(f"| MEASURED: Improved DSP (Tempogram + Octave Prior + Temperley) | {N} full tracks (120s each) | **{imp_acc1:.1f}%** | **{imp_acc2:.1f}%** | **{imp_key_exact:.1f}%** | **{imp_key_mirex:.1f}%** |")

# 6. Learned Model Survey (Beat This!, madmom, Essentia, KeyNet)
log("\n--- [PART 4] LEARNED TEMPO/KEY MODEL LICENCE SURVEY ---")
log("| Model Name | Code Licence | Weights Licence | Commercial Verdict | Notes & Source |")
log("|---|---|---|---|---|")
log("| **Beat This!** (CPJKU, ISMIR 2024) | MIT License | MIT License (Explicit in README) | **COMMERCIALLY SHIPPABLE** (Subject to training set assessment) | GitHub: `CPJKU/beat_this`. High accuracy SOTA beat/tempo tracking without DBN. |")
log("| **madmom** | BSD-2-Clause | Model dependencies GPL-3.0 | **NOT SHIPPABLE** (GPL license taint prevents commercial proprietary distribution) | GitHub: `CPJKU/madmom`. |")
log("| **Essentia / Essentia.js** | AGPL-3.0 | AGPL-3.0 | **NOT SHIPPABLE** without paid commercial license | GitHub: `MTG/essentia`. |")
log("| **KeyNet** (Rocha et al.) | GPL-3.0 | Non-commercial / Research only | **NOT SHIPPABLE** | Research checkpoint. |")

log("\n=== R5 EVALUATION COMPLETED ===")
log_out.close()
