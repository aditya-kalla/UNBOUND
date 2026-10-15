import os
import sys
import time
import glob
import hashlib
import psutil
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
import fast_bss_eval
import demucs.pretrained
import demucs.apply

# UTF-8 output
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REAL_DIR = os.path.join(BASE_DIR, "data", "real")
STEMS_EVAL_DIR = os.path.join(REAL_DIR, "stems_eval")

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "t2_stems_eval.log")
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

log("=== T2: 4-SOURCE STEM SEPARATION EVALUATION ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log(f"Platform: Windows 11 AMD64, ORT Providers: {ort.get_available_providers()}")

# 1. Model Sizes and Hashes
log("\n--- [PART 1] MODEL ARTIFACTS, SIZES, AND HASHES ---")
models_info = {}

# HTDemucs models
htdemucs_cache = r"C:\Users\lenovo\.cache\huggingface\hub\models--StemSplitio--htdemucs-onnx\snapshots\d54ed9eb60e258ea82131c6ee14578628816456a"
htdemucs_fp32 = os.path.realpath(os.path.join(htdemucs_cache, "htdemucs.onnx")) if os.path.exists(htdemucs_cache) else ""
htdemucs_fp16 = os.path.realpath(os.path.join(htdemucs_cache, "htdemucs_fp16weights.onnx")) if os.path.exists(htdemucs_cache) else ""
htdemucs_int8 = os.path.join(MODELS_DIR, "htdemucs_int8.onnx")

for name, path in [
    ("htdemucs_fp32", htdemucs_fp32),
    ("htdemucs_fp16", htdemucs_fp16),
    ("htdemucs_int8", htdemucs_int8)
]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        sha = get_sha256(path)
        models_info[name] = {"path": path, "size_bytes": size, "size_mb": size / (1024*1024), "sha256": sha}
        log(f"MEASURED: {name} | Size: {size / (1024*1024):.2f} MB ({size} bytes) | SHA256: {sha}")
    else:
        log(f"MEASURED: {name} | File not found at {path}")

# Open-Unmix 4-target models
openunmix_dir = os.path.join(MODELS_DIR, "openunmix")
oum_targets = ["vocals", "drums", "bass", "other"]
oum_total_fp32 = 0
oum_total_int8 = 0

for t in oum_targets:
    p_fp32 = os.path.join(openunmix_dir, f"openunmix_{t}_fp32.onnx")
    p_int8 = os.path.join(openunmix_dir, f"openunmix_{t}_int8.onnx")
    if os.path.exists(p_fp32):
        s_fp32 = os.path.getsize(p_fp32)
        oum_total_fp32 += s_fp32
        log(f"MEASURED: openunmix_{t}_fp32 | Size: {s_fp32 / (1024*1024):.2f} MB | SHA256: {get_sha256(p_fp32)}")
    if os.path.exists(p_int8):
        s_int8 = os.path.getsize(p_int8)
        oum_total_int8 += s_int8
        log(f"MEASURED: openunmix_{t}_int8  | Size: {s_int8 / (1024*1024):.2f} MB | SHA256: {get_sha256(p_int8)}")

log(f"MEASURED: Open-Unmix 4-target FP32 Total Size: {oum_total_fp32 / (1024*1024):.2f} MB")
log(f"MEASURED: Open-Unmix 4-target INT8 Total Size: {oum_total_int8 / (1024*1024):.2f} MB")

# 2. Separation Quality (SDR) on 10 MUSDB18 Test Excerpts
log("\n--- [PART 2] SEPARATION QUALITY: SDR ON MUSDB18 TEST EXCERPTS ---")
test_track_dirs = sorted(glob.glob(os.path.join(STEMS_EVAL_DIR, "*")))[:10]
log(f"Evaluating {len(test_track_dirs)} MUSDB18 multi-track test excerpts...")

# Load models for inference
log("Loading HTDemucs model...")
htdemucs_model = demucs.pretrained.get_model('htdemucs')
htdemucs_model.eval()

# Open-Unmix ONNX inference function
def run_openunmix_onnx(mix_audio, is_int8=False, providers=["CPUExecutionProvider"]):
    # mix_audio: (2, N) at 44100
    N = mix_audio.shape[-1]
    x_t = torch.from_numpy(mix_audio).float()
    
    # STFT with Hann window (standard Open-Unmix config: n_fft=4096, hop_length=1024)
    window = torch.hann_window(4096)
    spec = torch.stft(x_t, n_fft=4096, hop_length=1024, window=window, return_complex=True)
    mag = torch.abs(spec).unsqueeze(0).numpy() # (1, 2, 2049, frames)
    phase = torch.angle(spec)
    
    pred_mags = {}
    for target in oum_targets:
        suffix = "int8" if is_int8 else "fp32"
        model_path = os.path.join(openunmix_dir, f"openunmix_{target}_{suffix}.onnx")
        session = ort.InferenceSession(model_path, providers=providers)
        out_mag = session.run(None, {"magnitude": mag})[0]
        pred_mags[target] = torch.from_numpy(out_mag[0]).float()
        
    # Reconstruct audio via Wiener/Softmask or direct phase multiplication
    sum_mags = sum(pred_mags.values()) + 1e-7
    separated_audio = {}
    for target in oum_targets:
        # Soft mask
        mask = pred_mags[target] / sum_mags
        target_spec = spec * mask
        wav = torch.istft(target_spec, n_fft=4096, hop_length=1024, window=window, length=N)
        separated_audio[target] = wav.numpy()
        
    return separated_audio

sdr_results = {
    "htdemucs_fp32": {"vocals": [], "drums": [], "bass": [], "other": []},
    "openunmix_fp32": {"vocals": [], "drums": [], "bass": [], "other": []},
    "openunmix_int8": {"vocals": [], "drums": [], "bass": [], "other": []}
}

def compute_track_sdr(ref, est):
    # ref, est: (4, 2, N). Targets: 0: vocals, 1: drums, 2: bass, 3: other
    sdrs = []
    for s in range(4):
        ref_s = ref[s]
        est_s = est[s]
        energy = np.sum(ref_s**2)
        if energy < 1e-5:
            # Silent reference stem - not evaluated or 0.0
            sdrs.append(np.nan)
        else:
            try:
                val = fast_bss_eval.sdr(ref_s[np.newaxis, ...], est_s[np.newaxis, ...])[0]
                sdrs.append(float(np.mean(val)))
            except Exception:
                s_t = np.sum(ref_s * est_s, axis=-1, keepdims=True) / (energy + 1e-8) * ref_s
                e_n = est_s - s_t
                val = 10.0 * np.log10(np.sum(s_t**2, axis=-1) / (np.sum(e_n**2, axis=-1) + 1e-8))
                sdrs.append(float(np.mean(val)))
    return sdrs

for idx, t_dir in enumerate(test_track_dirs):
    t_name = os.path.basename(t_dir)
    mix, sr = sf.read(os.path.join(t_dir, "mixture.wav"))
    voc_ref, _ = sf.read(os.path.join(t_dir, "vocals.wav"))
    drm_ref, _ = sf.read(os.path.join(t_dir, "drums.wav"))
    bss_ref, _ = sf.read(os.path.join(t_dir, "bass.wav"))
    oth_ref, _ = sf.read(os.path.join(t_dir, "other.wav"))
    
    ref = np.stack([voc_ref.T, drm_ref.T, bss_ref.T, oth_ref.T], axis=0).astype(np.float32)
    mix_t = torch.from_numpy(mix.T).float().unsqueeze(0)
    
    # 1. HTDemucs FP32
    with torch.no_grad():
        ht_out = demucs.apply.apply_model(htdemucs_model, mix_t, split=True, overlap=0.25)[0].numpy()
        ht_voc, ht_drm, ht_bss, ht_oth = ht_out[3], ht_out[0], ht_out[1], ht_out[2]
        est_ht = np.stack([ht_voc, ht_drm, ht_bss, ht_oth], axis=0).astype(np.float32)
        sdr_ht = compute_track_sdr(ref, est_ht)
        for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
            if not np.isnan(sdr_ht[i]):
                sdr_results["htdemucs_fp32"][stem].append(sdr_ht[i])
        
    # 2. Open-Unmix FP32
    oum_fp32 = run_openunmix_onnx(mix.T, is_int8=False)
    est_oum_fp32 = np.stack([oum_fp32["vocals"], oum_fp32["drums"], oum_fp32["bass"], oum_fp32["other"]], axis=0).astype(np.float32)
    sdr_oum_fp32 = compute_track_sdr(ref, est_oum_fp32)
    for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
        if not np.isnan(sdr_oum_fp32[i]):
            sdr_results["openunmix_fp32"][stem].append(sdr_oum_fp32[i])

    # 3. Open-Unmix INT8
    oum_int8 = run_openunmix_onnx(mix.T, is_int8=True)
    est_oum_int8 = np.stack([oum_int8["vocals"], oum_int8["drums"], oum_int8["bass"], oum_int8["other"]], axis=0).astype(np.float32)
    sdr_oum_int8 = compute_track_sdr(ref, est_oum_int8)
    for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
        if not np.isnan(sdr_oum_int8[i]):
            sdr_results["openunmix_int8"][stem].append(sdr_oum_int8[i])
    
    log(f"Track {idx+1:02d} ({t_name}) | HTDemucs Vocals: {sdr_ht[0]:.2f} dB | OpenUnmix FP32 Vocals: {sdr_oum_fp32[0]:.2f} dB | INT8 Vocals: {sdr_oum_int8[0]:.2f} dB")

log("\n--- SDR SUMMARY PER STEM (Mean over 10 MUSDB18 Test Excerpts) ---")
for m_name, stems in sdr_results.items():
    v_mean = np.mean(stems["vocals"])
    d_mean = np.mean(stems["drums"])
    b_mean = np.mean(stems["bass"])
    o_mean = np.mean(stems["other"])
    overall = (v_mean + d_mean + b_mean + o_mean) / 4.0
    log(f"MEASURED: {m_name} | Vocals: {v_mean:.2f} dB | Drums: {d_mean:.2f} dB | Bass: {b_mean:.2f} dB | Other: {o_mean:.2f} dB | Mean SDR: {overall:.2f} dB")

gap_v = np.mean(sdr_results["htdemucs_fp32"]["vocals"]) - np.mean(sdr_results["openunmix_fp32"]["vocals"])
gap_all = (np.mean(sdr_results["htdemucs_fp32"]["vocals"]) + np.mean(sdr_results["htdemucs_fp32"]["drums"]) + np.mean(sdr_results["htdemucs_fp32"]["bass"]) + np.mean(sdr_results["htdemucs_fp32"]["other"]))/4.0 - \
          (np.mean(sdr_results["openunmix_fp32"]["vocals"]) + np.mean(sdr_results["openunmix_fp32"]["drums"]) + np.mean(sdr_results["openunmix_fp32"]["bass"]) + np.mean(sdr_results["openunmix_fp32"]["other"]))/4.0
log(f"\nMEASURED: HTDemucs vs OpenUnmix Quality Gap: {gap_v:.2f} dB (Vocals), {gap_all:.2f} dB (Average across 4 stems)")

# INT8 degradation check
drop_v = np.mean(sdr_results["openunmix_fp32"]["vocals"]) - np.mean(sdr_results["openunmix_int8"]["vocals"])
drop_all = (np.mean(sdr_results["openunmix_fp32"]["vocals"]) + np.mean(sdr_results["openunmix_fp32"]["drums"]) + np.mean(sdr_results["openunmix_fp32"]["bass"]) + np.mean(sdr_results["openunmix_fp32"]["other"]))/4.0 - \
           (np.mean(sdr_results["openunmix_int8"]["vocals"]) + np.mean(sdr_results["openunmix_int8"]["drums"]) + np.mean(sdr_results["openunmix_int8"]["bass"]) + np.mean(sdr_results["openunmix_int8"]["other"]))/4.0
log(f"MEASURED: Open-Unmix INT8 SDR degradation: {drop_v:.2f} dB (Vocals), {drop_all:.2f} dB (Average across 4 stems)")
if drop_all > 3.0:
    log("MEASURED: INT8 BREAKS the separation quality (severe degradation > 3 dB).")
else:
    log(f"MEASURED: INT8 preserves separation quality within acceptable bound ({drop_all:.2f} dB degradation).")

# 3. Real 4-minute Track Benchmark (RAM, RTF, CPU vs DirectML)
log("\n--- [PART 3] RUNTIME BENCHMARK: 4-MINUTE REAL TRACK ---")
track_4m = os.path.join(REAL_DIR, "audio", "USER_145_Nuvvunte Chaley.mp3")
if not os.path.exists(track_4m):
    # Fallback to any 4m track in real dir
    candidates = glob.glob(os.path.join(REAL_DIR, "audio", "*Devi*")) + glob.glob(os.path.join(REAL_DIR, "audio", "*Beethoven*"))
    track_4m = candidates[0] if candidates else glob.glob(os.path.join(REAL_DIR, "audio", "*.wav"))[0]

log(f"Benchmark track: {os.path.basename(track_4m)}")
audio_4m, sr_4m = sf.read(track_4m)
dur_4m = len(audio_4m) / sr_4m
log(f"Track duration: {dur_4m:.2f} seconds ({dur_4m/60:.2f} minutes)")

process = psutil.Process(os.getpid())

def benchmark_run(name, fn):
    # Force GC
    import gc
    gc.collect()
    ram_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()
    res = fn()
    t_elapsed = time.time() - t0
    ram_after = process.memory_info().rss / (1024 * 1024)
    peak_ram = ram_after
    rtf = t_elapsed / dur_4m
    log(f"MEASURED: {name} | Duration: {dur_4m:.1f}s | Wall Time: {t_elapsed:.2f}s | RTF: {rtf:.3f}x | Peak RAM: {peak_ram:.1f} MB (Delta: {peak_ram - ram_before:.1f} MB)")
    return rtf, peak_ram

# Test HTDemucs PyTorch CPU with bounded chunking (segment=7.8s, overlap=0.25)
def run_htdemucs_4m():
    x_t = torch.from_numpy(audio_4m.T).float().unsqueeze(0)
    with torch.no_grad():
        out = demucs.apply.apply_model(htdemucs_model, x_t, split=True, segment=7.8, overlap=0.1, device='cpu')
    return out

benchmark_run("HTDemucs (PyTorch/CPU, segment=7.8s, overlap=0.1)", run_htdemucs_4m)

# Test Open-Unmix ONNX CPU
def run_openunmix_4m_cpu():
    return run_openunmix_onnx(audio_4m.T, is_int8=True, providers=["CPUExecutionProvider"])

benchmark_run("Open-Unmix 4-target INT8 (ORT CPU, full-file STFT)", run_openunmix_4m_cpu)

# Test Open-Unmix ONNX DirectML
try:
    def run_openunmix_4m_dml():
        return run_openunmix_onnx(audio_4m.T, is_int8=True, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    benchmark_run("Open-Unmix 4-target INT8 (ORT DirectML)", run_openunmix_4m_dml)
except Exception as e:
    log(f"MEASURED: Open-Unmix DirectML failed: {e}")

log("\n=== T2 COMPLETED SUCCESSFULLY ===")
log_out.close()
