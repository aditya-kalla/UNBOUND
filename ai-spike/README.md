# AI Layer Spike - Project UNBOUND

Evidence-based architectural spike evaluating deep learning models, offline inference runtimes, and DSP algorithms across Windows, macOS, Android, and iOS for Project UNBOUND.

## Directory Structure
- `scripts/`: Standalone Python benchmark scripts
  - `hash_models.py`: Computes SHA256 hashes and file sizes for all model weights.
  - `export_openunmix_dynamic.py`: Exports and quantizes Open-Unmix to dynamic ONNX (FP32, FP16, INT8).
  - `generate_reference_tracks.py`: Synthesizes 10 reference tracks with mathematically defined ground truth BPM and Key.
  - `q1_stems.py`: Benchmarks stem separation models (Open-Unmix, MDX-Net, Demucs), measuring RTF, RAM, overlap-add windowing, and audio degradation.
  - `q2_vibe_search.py`: Benchmarks text-audio joint embeddings (LAION-CLAP, MiniLM), retrieval accuracy on 23 local tracks and 10 queries, dimensionality reduction (INT8, FP16, 128-d PCA), and 50k track SIMD search.
  - `q3_acoustic_xray.py`: Benchmarks pure Rust/DSP baseline vs neural beat tracking on 10 known tracks, measuring BPM (+-2%) and key accuracy.
  - `q4_autoplay.py`: Prototypes Neural Autoplay centroid affinity ranker with skip penalty and MMR diversity on 50k candidate tracks.
- `ort-check/`: Minimal Rust crate validating `ort` crate cross-platform compilation across Windows, Android, iOS, and macOS.
- `models/`: ONNX, INT8, and ORT model files.
- `logs/`: Raw execution logs with MEASURED benchmarks and timestamps.
- `out/`: Separated audio stem WAV samples and retrieval outputs.
- `SPIKE_REPORT.md`: Comprehensive engineering report answering Q1-Q6.

## How to Rerun Benchmarks

### Option A: From Repository Root (`UNBOUND`)
```powershell
# 1. Setup Python Environment
python -m venv ai-spike/.venv
.\ai-spike\.venv\Scripts\pip.exe install -r ai-spike/requirements.txt

# 2. Run Benchmarks
.\ai-spike\.venv\Scripts\python.exe ai-spike/scripts/hash_models.py
.\ai-spike\.venv\Scripts\python.exe ai-spike/scripts/q1_stems.py
.\ai-spike\.venv\Scripts\python.exe ai-spike/scripts/q2_vibe_search.py
.\ai-spike\.venv\Scripts\python.exe ai-spike/scripts/q3_acoustic_xray.py
.\ai-spike\.venv\Scripts\python.exe ai-spike/scripts/q4_autoplay.py

# 3. Cross-Platform ORT Cargo Check
cargo check --manifest-path ai-spike/ort-check/Cargo.toml --target x86_64-pc-windows-msvc
```

### Option B: From Inside `ai-spike` Folder
```powershell
cd ai-spike
.\.venv\Scripts\python.exe scripts\hash_models.py
cargo check --manifest-path ort-check\Cargo.toml
cargo check --manifest-path xray-rs\Cargo.toml
cargo check --manifest-path autoplay-rs\Cargo.toml
```

