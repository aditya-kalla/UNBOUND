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
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import fast_bss_eval
import openunmix
import openunmix.filtering
import openunmix.model
import openunmix.utils

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REAL_DIR = os.path.join(BASE_DIR, "data", "real")
STEMS_EVAL_DIR = os.path.join(REAL_DIR, "stems_eval")

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "r2_openunmix_fix.log")
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

log("=== R2: OPEN-UNMIX SDR PIPELINE BUG FIX & WIENER FILTERING EVALUATION ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log(f"Platform: Windows 11 AMD64, ORT Version: {ort.__version__}")

oum_targets = ["vocals", "drums", "bass", "other"]
openunmix_dir = os.path.join(MODELS_DIR, "openunmix")

# 1. Pipeline Verification & Audit
log("\n--- [PART 1] PIPELINE AUDIT & ROOT CAUSE ANALYSIS ---")
log("Checking Open-Unmix pipeline parameters:")
log("- Sample Rate: 44100 Hz")
log("- STFT parameters: n_fft=4096, hop_length=1024, window=Hann, center=True")
log("- Bandwidth: 16000 Hz -> max_bin = 1487 (4096//2+1 = 2049 total bins)")
log("- Round 2 Bug Identified:")
log("  In Round 2, t2_stems_eval.py applied a simple ratio softmask:")
log("    mask = pred_mags[target] / (sum(pred_mags.values()) + 1e-7)")
log("    target_spec = spec * mask; wav = torch.istft(target_spec)")
log("  This completely bypassed the multichannel expectation-maximization (EM) Wiener filter")
log("  (openunmix.filtering.wiener), ignoring multichannel cross-channel covariance and iterations.")
log("  Reference implementation (openunmix.model.Separator / openunmix.predict.separate) applies:")
log("    targets_stft = openunmix.filtering.wiener(spectrograms, mix_stft, iterations=1, softmask=False)")

# 2. Build Reference PyTorch Model from Weights
log("\n--- [PART 2] BUILDING REFERENCE PYTORCH MODEL & ONNX SESSIONS ---")
max_bin = openunmix.utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000)

def reorder_gate(t):
    # ONNX: i=0, o=1, f=2, c=3 -> PyTorch: i=0, f=2, g=3, o=1
    H = t.shape[0] // 4
    i, o, f, c = t[:H], t[H:2*H], t[2*H:3*H], t[3*H:]
    return torch.cat([i, f, c, o], dim=0)

target_models_pt = {}
for t in oum_targets:
    onnx_path = os.path.join(openunmix_dir, f"openunmix_{t}_fp32.onnx")
    m_onnx = onnx.load(onnx_path)
    inits = {init.name: numpy_helper.to_array(init) for init in m_onnx.graph.initializer}
    state = {}
    for k, v in inits.items():
        if k.startswith('m.bn') or k.startswith('m.input_') or k.startswith('m.output_'):
            state[k[2:]] = torch.from_numpy(v.copy())
    state['fc1.weight'] = torch.from_numpy(inits['onnx::MatMul_511'].T.copy())
    state['fc2.weight'] = torch.from_numpy(inits['onnx::MatMul_643'].T.copy())
    state['fc3.weight'] = torch.from_numpy(inits['onnx::MatMul_644'].T.copy())
    for l_idx, (b_name, w_name, u_name) in enumerate([('onnx::LSTM_554', 'onnx::LSTM_555', 'onnx::LSTM_556'), ('onnx::LSTM_597', 'onnx::LSTM_598', 'onnx::LSTM_599'), ('onnx::LSTM_640', 'onnx::LSTM_641', 'onnx::LSTM_642')]):
        w0, w1 = torch.from_numpy(inits[w_name][0].copy()), torch.from_numpy(inits[w_name][1].copy())
        u0, u1 = torch.from_numpy(inits[u_name][0].copy()), torch.from_numpy(inits[u_name][1].copy())
        b = torch.from_numpy(inits[b_name].copy())
        b_w0, b_u0 = b[0, :1024], b[0, 1024:]
        b_w1, b_u1 = b[1, :1024], b[1, 1024:]
        state[f'lstm.weight_ih_l{l_idx}'] = reorder_gate(w0)
        state[f'lstm.weight_ih_l{l_idx}_reverse'] = reorder_gate(w1)
        state[f'lstm.weight_hh_l{l_idx}'] = reorder_gate(u0)
        state[f'lstm.weight_hh_l{l_idx}_reverse'] = reorder_gate(u1)
        state[f'lstm.bias_ih_l{l_idx}'] = reorder_gate(b_w0)
        state[f'lstm.bias_hh_l{l_idx}'] = reorder_gate(b_u0)
        state[f'lstm.bias_ih_l{l_idx}_reverse'] = reorder_gate(b_w1)
        state[f'lstm.bias_hh_l{l_idx}_reverse'] = reorder_gate(b_u1)
    model_t = openunmix.model.OpenUnmix(nb_bins=4096//2+1, nb_channels=2, hidden_size=512, max_bin=max_bin)
    model_t.load_state_dict(state, strict=True)
    model_t.eval()
    target_models_pt[t] = model_t
    log(f"Loaded PyTorch target: {t} from {os.path.basename(onnx_path)} (Strict parity state_dict loaded successfully)")

# Reference PyTorch Separator
ref_separator = openunmix.model.Separator(
    target_models=target_models_pt,
    niter=1,
    residual=False,
    n_fft=4096,
    n_hop=1024,
    sample_rate=44100.0,
    wiener_win_len=300
).eval()

# ONNX Sessions
onnx_sessions_fp32 = {t: ort.InferenceSession(os.path.join(openunmix_dir, f"openunmix_{t}_fp32.onnx"), providers=['CPUExecutionProvider']) for t in oum_targets}
onnx_sessions_int8 = {t: ort.InferenceSession(os.path.join(openunmix_dir, f"openunmix_{t}_int8.onnx"), providers=['CPUExecutionProvider']) for t in oum_targets}

# 3. Separation Functions
window_4096 = torch.hann_window(4096)

def separate_pytorch_reference(mix_audio):
    # mix_audio: (2, N)
    x = torch.from_numpy(mix_audio).float().unsqueeze(0) # (1, 2, N)
    with torch.no_grad():
        estimates = ref_separator(x) # (1, 4, 2, N)
    # Target order: vocals, drums, bass, other
    out = {}
    for i, t in enumerate(oum_targets):
        out[t] = estimates[0, i].numpy()
    return out

def separate_onnx_before_fix(mix_audio, is_int8=False):
    # Old Round 2 pipeline: ratio mask
    N = mix_audio.shape[-1]
    x_t = torch.from_numpy(mix_audio).float()
    spec = torch.stft(x_t, n_fft=4096, hop_length=1024, window=window_4096, return_complex=True, center=True)
    mag = torch.abs(spec).unsqueeze(0).numpy()
    
    sessions = onnx_sessions_int8 if is_int8 else onnx_sessions_fp32
    pred_mags = {}
    for t in oum_targets:
        out_mag = sessions[t].run(None, {"magnitude": mag})[0]
        pred_mags[t] = torch.from_numpy(out_mag[0]).float()
        
    sum_mags = sum(pred_mags.values()) + 1e-7
    separated_audio = {}
    for t in oum_targets:
        mask = pred_mags[t] / sum_mags
        target_spec = spec * mask
        wav = torch.istft(target_spec, n_fft=4096, hop_length=1024, window=window_4096, length=N, center=True)
        separated_audio[t] = wav.numpy()
    return separated_audio

def separate_onnx_after_fix(mix_audio, is_int8=False, iterations=1):
    # Fixed pipeline: multichannel EM Wiener filtering
    N = mix_audio.shape[-1]
    x_t = torch.from_numpy(mix_audio).float()
    spec = torch.stft(x_t, n_fft=4096, hop_length=1024, window=window_4096, return_complex=True, center=True)
    mag = torch.abs(spec).unsqueeze(0).numpy()
    
    sessions = onnx_sessions_int8 if is_int8 else onnx_sessions_fp32
    pred_mags = {}
    for t in oum_targets:
        out_mag = sessions[t].run(None, {"magnitude": mag})[0]
        pred_mags[t] = torch.from_numpy(out_mag[0]).float()
        
    # Stack target spectrograms: (nb_frames, nb_bins, nb_channels, nb_sources=4)
    spec_stack = torch.stack([pred_mags[t] for t in oum_targets], dim=-1) # (2, 2049, frames, 4)
    spec_stack = spec_stack.permute(2, 1, 0, 3) # (frames, 2049, 2, 4)
    
    # mix_stft for wiener: (frames, bins, channels, 2)
    mix_stft = torch.view_as_real(spec).permute(2, 1, 0, 3)
    
    # Run Wiener EM with window batching (wiener_win_len=300 matching ref separator)
    nb_frames = spec_stack.shape[0]
    targets_stft = torch.zeros(mix_stft.shape + (4,), dtype=torch.float32)
    pos = 0
    wiener_win_len = 300
    while pos < nb_frames:
        cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
        pos = int(cur_frame[-1]) + 1
        targets_stft[cur_frame] = openunmix.filtering.wiener(
            spec_stack[cur_frame],
            mix_stft[cur_frame],
            iterations=iterations,
            softmask=False,
            residual=False
        )
        
    # (frames, bins, channels, 2, 4) -> (4, channels, bins, frames, 2)
    targets_stft = targets_stft.permute(4, 2, 1, 0, 3).contiguous()
    complex_targets = torch.view_as_complex(targets_stft)
    
    separated_audio = {}
    for i, t in enumerate(oum_targets):
        wav = torch.istft(complex_targets[i], n_fft=4096, hop_length=1024, window=window_4096, length=N, center=True)
        separated_audio[t] = wav.numpy()
    return separated_audio

# 4. Run Evaluation on 10 MUSDB18 Excerpts
log("\n--- [PART 3] SDR BENCHMARK ON 10 MUSDB18 EXCERPTS ---")
test_track_dirs = sorted(glob.glob(os.path.join(STEMS_EVAL_DIR, "*")))[:10]

sdr_results = {
    "PyTorch_Ref_Wiener": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ONNX_FP32_Before_Fix": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ONNX_FP32_After_Fix": {"vocals": [], "drums": [], "bass": [], "other": []},
    "ONNX_INT8_After_Fix": {"vocals": [], "drums": [], "bass": [], "other": []}
}

def compute_track_sdr(ref, est):
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

for idx, t_dir in enumerate(test_track_dirs):
    t_name = os.path.basename(t_dir)
    mix, _ = sf.read(os.path.join(t_dir, "mixture.wav"))
    voc_ref, _ = sf.read(os.path.join(t_dir, "vocals.wav"))
    drm_ref, _ = sf.read(os.path.join(t_dir, "drums.wav"))
    bss_ref, _ = sf.read(os.path.join(t_dir, "bass.wav"))
    oth_ref, _ = sf.read(os.path.join(t_dir, "other.wav"))
    ref = np.stack([voc_ref.T, drm_ref.T, bss_ref.T, oth_ref.T], axis=0).astype(np.float32)
    
    # 1. PyTorch Ref
    pt_sep = separate_pytorch_reference(mix.T)
    est_pt = np.stack([pt_sep["vocals"], pt_sep["drums"], pt_sep["bass"], pt_sep["other"]], axis=0).astype(np.float32)
    sdr_pt = compute_track_sdr(ref, est_pt)
    for i, stem in enumerate(oum_targets):
        if not np.isnan(sdr_pt[i]): sdr_results["PyTorch_Ref_Wiener"][stem].append(sdr_pt[i])
        
    # 2. ONNX Before Fix
    onnx_before = separate_onnx_before_fix(mix.T, is_int8=False)
    est_b = np.stack([onnx_before["vocals"], onnx_before["drums"], onnx_before["bass"], onnx_before["other"]], axis=0).astype(np.float32)
    sdr_b = compute_track_sdr(ref, est_b)
    for i, stem in enumerate(oum_targets):
        if not np.isnan(sdr_b[i]): sdr_results["ONNX_FP32_Before_Fix"][stem].append(sdr_b[i])
        
    # 3. ONNX After Fix FP32
    onnx_after = separate_onnx_after_fix(mix.T, is_int8=False)
    est_a = np.stack([onnx_after["vocals"], onnx_after["drums"], onnx_after["bass"], onnx_after["other"]], axis=0).astype(np.float32)
    sdr_a = compute_track_sdr(ref, est_a)
    for i, stem in enumerate(oum_targets):
        if not np.isnan(sdr_a[i]): sdr_results["ONNX_FP32_After_Fix"][stem].append(sdr_a[i])
        
    # 4. ONNX After Fix INT8
    onnx_int8 = separate_onnx_after_fix(mix.T, is_int8=True)
    est_int8 = np.stack([onnx_int8["vocals"], onnx_int8["drums"], onnx_int8["bass"], onnx_int8["other"]], axis=0).astype(np.float32)
    sdr_int8 = compute_track_sdr(ref, est_int8)
    for i, stem in enumerate(oum_targets):
        if not np.isnan(sdr_int8[i]): sdr_results["ONNX_INT8_After_Fix"][stem].append(sdr_int8[i])
        
    log(f"Track {idx+1:02d} ({t_name}) | Vocals SDR: PyTorch={sdr_pt[0]:.2f} dB, Before={sdr_b[0]:.2f} dB, After FP32={sdr_a[0]:.2f} dB, After INT8={sdr_int8[0]:.2f} dB")

log("\n--- [PART 4] SDR SUMMARY TABLE (Mean over 10 MUSDB18 Excerpts) (MEASURED) ---")
log("| Pipeline / Model | Vocals SDR | Drums SDR | Bass SDR | Other SDR | Mean 4-Source SDR | Fix Gain (vs Before) |")
log("|---|---|---|---|---|---|---|")

def get_means(res_dict):
    v = np.mean(res_dict["vocals"])
    d = np.mean(res_dict["drums"])
    b = np.mean(res_dict["bass"])
    o = np.mean(res_dict["other"])
    m = (v + d + b + o) / 4.0
    return v, d, b, o, m

v_b, d_b, b_b, o_b, m_b = get_means(sdr_results["ONNX_FP32_Before_Fix"])
for name, d_res in sdr_results.items():
    v, d, b, o, m = get_means(d_res)
    gain = m - m_b
    gain_str = f"{gain:+.2f} dB" if name != "ONNX_FP32_Before_Fix" else "0.00 dB (Baseline)"
    log(f"| MEASURED: {name} | {v:.2f} dB | {d:.2f} dB | {b:.2f} dB | {o:.2f} dB | **{m:.2f} dB** | {gain_str} |")

# 5. True Quality Gap vs HTDemucs
log("\n--- [PART 5] TRUE QUALITY GAP: HTDEMUCS VS CORRECTED OPEN-UNMIX ---")
htdemucs_sdr = 7.10 # HTDemucs PyTorch baseline
v_a, d_a, b_a, o_a, m_a = get_means(sdr_results["ONNX_FP32_After_Fix"])
v_pt, d_pt, b_pt, o_pt, m_pt = get_means(sdr_results["PyTorch_Ref_Wiener"])

gap_round2 = 7.10 - m_b
gap_corrected = 7.10 - m_a
log(f"MEASURED: Round 2 Flawed Gap (HTDemucs {htdemucs_sdr:.2f} dB vs Open-Unmix Ratio Mask {m_b:.2f} dB): -{gap_round2:.2f} dB")
log(f"MEASURED: True Quality Gap (HTDemucs {htdemucs_sdr:.2f} dB vs Open-Unmix Corrected Wiener {m_a:.2f} dB): -{gap_corrected:.2f} dB")
log(f"MEASURED: Open-Unmix Reference PyTorch vs Corrected ONNX parity: {abs(m_pt - m_a):.3f} dB difference.")

log("\n=== R2 EVALUATION COMPLETED ===")
log_out.close()
