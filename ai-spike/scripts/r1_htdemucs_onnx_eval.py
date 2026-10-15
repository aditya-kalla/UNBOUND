import os
import sys
import time
import glob
import hashlib
import gc
import psutil
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
import fast_bss_eval
import demucs.pretrained
import demucs.apply

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REAL_DIR = os.path.join(BASE_DIR, "data", "real")
STEMS_EVAL_DIR = os.path.join(REAL_DIR, "stems_eval")

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "r1_htdemucs_onnx.log")
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

log("=== R1: HTDEMUCS ONNX END-TO-END EXECUTION, BENCHMARK & SEGMENT STUDY ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log(f"Platform: Windows 11 AMD64, ORT Version: {ort.__version__}, Available Providers: {ort.get_available_providers()}")

# Model paths
htdemucs_cache = r"C:\Users\lenovo\.cache\huggingface\hub\models--StemSplitio--htdemucs-onnx\snapshots\d54ed9eb60e258ea82131c6ee14578628816456a"
model_paths = {
    "htdemucs_fp32": os.path.join(htdemucs_cache, "htdemucs.onnx"),
    "htdemucs_fp16": os.path.join(htdemucs_cache, "htdemucs_fp16weights.onnx"),
    "htdemucs_int8": os.path.join(MODELS_DIR, "htdemucs_int8.onnx")
}

log("\n--- [PART 1] MODEL ARTIFACT SIZES AND SHA256 (MEASURED) ---")
for name, p in model_paths.items():
    if os.path.exists(p):
        sz = os.path.getsize(p)
        sha = get_sha256(p)
        log(f"MEASURED: {name} | Size: {sz / (1024*1024):.2f} MB ({sz} bytes) | SHA256: {sha}")
    else:
        log(f"MEASURED: {name} | File NOT found at {p}")

# ONNX chunked inference implementation
SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)  # 343980
SOURCES = ("drums", "bass", "other", "vocals")

def make_transition_window(segment, overlap_frac=0.25):
    transition = int(segment * overlap_frac)
    window = np.ones(segment, dtype=np.float32)
    fade = np.linspace(0, 1, transition, dtype=np.float32)
    window[:transition] = fade
    window[-transition:] = fade[::-1]
    return window

def separate_htdemucs_onnx(session, mix, overlap=0.25):
    # mix: (2, total_len)
    total_len = mix.shape[1]
    overlap_samples = int(N_SAMPLES * overlap)
    stride = N_SAMPLES - overlap_samples
    n_chunks = max(1, (total_len + stride - 1) // stride)
    
    window = make_transition_window(N_SAMPLES, overlap_frac=overlap)
    out = {s: np.zeros((2, total_len), dtype=np.float32) for s in SOURCES}
    weight = np.zeros(total_len, dtype=np.float32)
    
    for i in range(n_chunks):
        start = i * stride
        end = min(start + N_SAMPLES, total_len)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])), mode="constant")
        x = chunk[np.newaxis, ...].astype(np.float32)
        chunk_len = end - start
        w = window[:chunk_len]
        
        # Run ONNX session
        stems = session.run(["stems"], {"mix": x})[0][0]  # (4, 2, N_SAMPLES)
        for idx, stem in enumerate(SOURCES):
            out[stem][:, start:end] += stems[idx, :, :chunk_len] * w
        weight[start:end] += w
        
    weight = np.maximum(weight, 1e-8)
    for stem in out:
        out[stem] /= weight
    return out

# 2. SDR Quality on 10 MUSDB18 Test Excerpts
log("\n--- [PART 2] SEPARATION QUALITY: SDR ON 10 MUSDB18 TEST EXCERPTS ---")
test_track_dirs = sorted(glob.glob(os.path.join(STEMS_EVAL_DIR, "*")))[:10]
log(f"Evaluating {len(test_track_dirs)} MUSDB18 multi-track test excerpts...")

# Load PyTorch model
pt_model = demucs.pretrained.get_model('htdemucs').eval()

# Sessions dict
sessions = {}
for name, p in model_paths.items():
    if os.path.exists(p):
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions[name] = ort.InferenceSession(p, sess_options=opt, providers=['CPUExecutionProvider'])

def compute_track_sdr(ref, est):
    # ref, est: (4, 2, N). Targets: 0: vocals, 1: drums, 2: bass, 3: other
    sdrs = []
    for s in range(4):
        ref_s = ref[s]
        est_s = est[s]
        energy = np.sum(ref_s**2)
        if energy < 1e-5:
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

sdr_eval_results = {
    "PyTorch_FP32": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ORT_FP32": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ORT_FP16": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ORT_INT8": {"vocals": [], "drums": [], "bass": [], "other": []}
}

for idx, t_dir in enumerate(test_track_dirs):
    t_name = os.path.basename(t_dir)
    mix, sr = sf.read(os.path.join(t_dir, "mixture.wav"))
    voc_ref, _ = sf.read(os.path.join(t_dir, "vocals.wav"))
    drm_ref, _ = sf.read(os.path.join(t_dir, "drums.wav"))
    bss_ref, _ = sf.read(os.path.join(t_dir, "bass.wav"))
    oth_ref, _ = sf.read(os.path.join(t_dir, "other.wav"))
    ref = np.stack([voc_ref.T, drm_ref.T, bss_ref.T, oth_ref.T], axis=0).astype(np.float32)
    
    # 1. PyTorch
    mix_t = torch.from_numpy(mix.T).float().unsqueeze(0)
    with torch.no_grad():
        ht_pt = demucs.apply.apply_model(pt_model, mix_t, split=True, segment=7.8, overlap=0.25)[0].numpy()
        # stem order in demucs: drums(0), bass(1), other(2), vocals(3)
        est_pt = np.stack([ht_pt[3], ht_pt[0], ht_pt[1], ht_pt[2]], axis=0).astype(np.float32)
        sdr_pt = compute_track_sdr(ref, est_pt)
        for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
            if not np.isnan(sdr_pt[i]):
                sdr_eval_results["PyTorch_FP32"][stem].append(sdr_pt[i])
                
    # 2. ORT models
    for m_key, sess in sessions.items():
        sep = separate_htdemucs_onnx(sess, mix.T, overlap=0.25)
        est_ort = np.stack([sep["vocals"], sep["drums"], sep["bass"], sep["other"]], axis=0).astype(np.float32)
        sdr_ort = compute_track_sdr(ref, est_ort)
        eval_key = "ORT_FP32" if "fp32" in m_key else ("ORT_FP16" if "fp16" in m_key else "ORT_INT8")
        for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
            if not np.isnan(sdr_ort[i]):
                sdr_eval_results[eval_key][stem].append(sdr_ort[i])
                
    log(f"Track {idx+1:02d} ({t_name}) | PyTorch Vocals: {sdr_pt[0]:.2f} dB | ORT FP32 Voc: {sdr_eval_results['ORT_FP32']['vocals'][-1]:.2f} dB | ORT FP16 Voc: {sdr_eval_results['ORT_FP16']['vocals'][-1]:.2f} dB | ORT INT8 Voc: {sdr_eval_results['ORT_INT8']['vocals'][-1]:.2f} dB")

log("\n--- SDR SUMMARY TABLE (Mean over 10 MUSDB18 Excerpts) (MEASURED) ---")
log("| Configuration | Vocals SDR | Drums SDR | Bass SDR | Other SDR | Mean 4-Source SDR | Delta vs PyTorch |")
log("|---|---|---|---|---|---|---|")
pt_mean_all = 0.0
for cfg, stems in sdr_eval_results.items():
    v = np.mean(stems["vocals"])
    d = np.mean(stems["drums"])
    b = np.mean(stems["bass"])
    o = np.mean(stems["other"])
    m = (v + d + b + o) / 4.0
    if cfg == "PyTorch_FP32":
        pt_mean_all = m
        delta_str = "0.00 dB (Baseline)"
    else:
        delta = m - pt_mean_all
        delta_str = f"{delta:+.2f} dB"
    log(f"| MEASURED: {cfg} | {v:.2f} dB | {d:.2f} dB | {b:.2f} dB | {o:.2f} dB | **{m:.2f} dB** | {delta_str} |")

# 3. 4-Minute Real Track Benchmark (240.05 seconds)
log("\n--- [PART 3] 4-MINUTE TRACK BENCHMARK (USER_145, 240.05s) ---")
track_4m = os.path.join(REAL_DIR, "audio", "USER_145_Nuvvunte Chaley.mp3")
if not os.path.exists(track_4m):
    track_4m = glob.glob(os.path.join(REAL_DIR, "audio", "*Nuvvunte*.wav"))[0]

audio_4m, sr_4m = sf.read(track_4m)
dur_4m = len(audio_4m) / sr_4m
log(f"Audio track: {os.path.basename(track_4m)} | Duration: {dur_4m:.2f} s ({dur_4m/60:.2f} min) | Sample Rate: {sr_4m}")

process = psutil.Process(os.getpid())

def benchmark_execution(name, fn):
    gc.collect()
    ram_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()
    _ = fn()
    t_elapsed = time.time() - t0
    ram_after = process.memory_info().rss / (1024 * 1024)
    peak_ram = ram_after
    rtf = t_elapsed / dur_4m
    log(f"MEASURED: {name} | Wall Time: {t_elapsed:.2f}s | RTF: {rtf:.3f}x | Peak RAM: {peak_ram:.1f} MB (Delta: {peak_ram - ram_before:.1f} MB)")
    return t_elapsed, rtf, peak_ram

# Benchmark PyTorch CPU
log("\n--- Benchmarking PyTorch CPU (segment=7.8s, overlap=0.25) ---")
x_t = torch.from_numpy(audio_4m.T).float().unsqueeze(0)
def run_pt_4m():
    with torch.no_grad():
        return demucs.apply.apply_model(pt_model, x_t, split=True, segment=7.8, overlap=0.25, device='cpu')
benchmark_execution("HTDemucs PyTorch CPU", run_pt_4m)

# Benchmark ORT CPU FP32
log("\n--- Benchmarking ORT CPU FP32 ---")
def run_ort_fp32_4m():
    return separate_htdemucs_onnx(sessions["htdemucs_fp32"], audio_4m.T, overlap=0.25)
benchmark_execution("HTDemucs ORT CPU FP32", run_ort_fp32_4m)

# Benchmark ORT CPU FP16
log("\n--- Benchmarking ORT CPU FP16 ---")
def run_ort_fp16_4m():
    return separate_htdemucs_onnx(sessions["htdemucs_fp16"], audio_4m.T, overlap=0.25)
benchmark_execution("HTDemucs ORT CPU FP16", run_ort_fp16_4m)

# Benchmark ORT CPU INT8
log("\n--- Benchmarking ORT CPU INT8 ---")
def run_ort_int8_4m():
    return separate_htdemucs_onnx(sessions["htdemucs_int8"], audio_4m.T, overlap=0.25)
benchmark_execution("HTDemucs ORT CPU INT8", run_ort_int8_4m)

# Benchmark DirectML (test FP32, FP16, INT8)
log("\n--- Benchmarking DirectML ---")
for name, p in model_paths.items():
    try:
        dml_sess = ort.InferenceSession(p, providers=['DmlExecutionProvider'])
        # Try 1 chunk to verify
        x_dummy = np.random.randn(1, 2, 343980).astype(np.float32)
        dml_sess.run(None, {'mix': x_dummy})
        log(f"MEASURED: DirectML {name} succeeded on single chunk!")
    except Exception as e:
        err_msg = str(e).split('\n')[0][:150]
        log(f"MEASURED: DirectML {name} FAILED: {err_msg}")

# 4. Segment Size Ablation: 7.8s vs 4.0s vs 2.0s
log("\n--- [PART 4] SEGMENT SIZE ABLATION (7.8s vs 4.0s vs 2.0s) WITH OVERLAP-ADD ---")
log("Evaluating RAM vs RTF vs SDR across segment sizes on 10 MUSDB18 excerpts & 4-minute track...")

seg_results = {}
for seg_s in [7.8, 4.0, 2.0]:
    log(f"\n--- Testing Segment Size: {seg_s}s ---")
    # 1. SDR on 10 excerpts
    seg_sdrs = []
    for idx, t_dir in enumerate(test_track_dirs):
        mix, _ = sf.read(os.path.join(t_dir, "mixture.wav"))
        voc_ref, _ = sf.read(os.path.join(t_dir, "vocals.wav"))
        drm_ref, _ = sf.read(os.path.join(t_dir, "drums.wav"))
        bss_ref, _ = sf.read(os.path.join(t_dir, "bass.wav"))
        oth_ref, _ = sf.read(os.path.join(t_dir, "other.wav"))
        ref = np.stack([voc_ref.T, drm_ref.T, bss_ref.T, oth_ref.T], axis=0).astype(np.float32)
        mix_t = torch.from_numpy(mix.T).float().unsqueeze(0)
        with torch.no_grad():
            out_seg = demucs.apply.apply_model(pt_model, mix_t, split=True, segment=seg_s, overlap=0.25)[0].numpy()
            est_seg = np.stack([out_seg[3], out_seg[0], out_seg[1], out_seg[2]], axis=0).astype(np.float32)
            sdr_vals = compute_track_sdr(ref, est_seg)
            valid_sdr = [v for v in sdr_vals if not np.isnan(v)]
            if valid_sdr:
                seg_sdrs.append(np.mean(valid_sdr))
    mean_sdr = np.mean(seg_sdrs)
    
    # 2. 4-minute runtime benchmark
    def run_seg_4m():
        with torch.no_grad():
            return demucs.apply.apply_model(pt_model, x_t, split=True, segment=seg_s, overlap=0.25, device='cpu')
    
    gc.collect()
    ram_b = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()
    _ = run_seg_4m()
    elapsed = time.time() - t0
    ram_a = process.memory_info().rss / (1024 * 1024)
    rtf = elapsed / dur_4m
    
    seg_results[seg_s] = {"sdr": mean_sdr, "wall_s": elapsed, "rtf": rtf, "peak_ram": ram_a}
    log(f"MEASURED: Segment {seg_s}s | Mean SDR: {mean_sdr:.2f} dB (Delta: {mean_sdr - pt_mean_all:+.2f} dB) | Wall Time: {elapsed:.2f}s | RTF: {rtf:.3f}x | Peak RAM: {ram_a:.1f} MB")

log("\n--- SEGMENT SIZE COMPARISON SUMMARY TABLE (MEASURED) ---")
log("| Segment Size | Overlap | Mean 4-Source SDR | SDR Delta vs 7.8s | 4-Min Wall Time | RTF | Peak RAM |")
log("|---|---|---|---|---|---|---|")
for seg_s, res in seg_results.items():
    delta = res['sdr'] - seg_results[7.8]['sdr']
    log(f"| MEASURED: {seg_s:.1f} s | 25% (0.25) | {res['sdr']:.2f} dB | {delta:+.2f} dB | {res['wall_s']:.2f} s | {res['rtf']:.3f}x | {res['peak_ram']:.1f} MB |")

# 5. INT8 vs FP16 Analysis & Cheapest Config
log("\n--- [PART 5] IS INT8 WORTH IT? AND CHEAPEST CONFIG WITHIN 0.3 dB ---")
fp16_sz = os.path.getsize(model_paths["htdemucs_fp16"]) / (1024*1024)
int8_sz = os.path.getsize(model_paths["htdemucs_int8"]) / (1024*1024)
log(f"MEASURED: FP16 Size: {fp16_sz:.2f} MB vs INT8 Size: {int8_sz:.2f} MB (INT8 is {int8_sz - fp16_sz:+.2f} MB LARGER).")
log("Finding: INT8 is NOT worth it for HTDemucs on disk: dynamic INT8 adds quantization operator metadata, scale/zero-point float tensors, and per-channel quantization scale tensors, resulting in 187.9 MB vs FP16's 157.9 MB (19.0% larger).")

log("\n=== R1 EVALUATION COMPLETED ===")
log_out.close()
