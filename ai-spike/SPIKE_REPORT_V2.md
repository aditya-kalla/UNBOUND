# UNBOUND AI SPIKE - ROUND 2 FINAL REPORT

## Executive Decision Table

| Capability | Round 1 Claim | Round 2 Measured Reality | Recommended Model / Stack | Size / Footprint | RTF / Latency | Licence Verdict |
|---|---|---|---|---|---|---|
| **Stems (4-source)** | "~4x cost, vocals only" | HTDemucs 7.10 dB SDR, Open-Unmix 2.96 dB SDR (INT8: 2.97 dB) | **HTDemucs ONNX** (Quality) / **Open-Unmix INT8** (Fallback) | HTDemucs: 187.9 MB (INT8), Open-Unmix: 34.5 MB (INT8) | HTDemucs CPU: 1.191x RTF; Open-Unmix CPU: 0.068x RTF | **NEEDS LEGAL REVIEW** (MUSDB18-HQ training data) |
| **Vibe Search (Audio)** | "150 MB INT8, 48 MB RAM" | 33.2 MB INT8 ONNX, 54.5 MB load RAM delta, Cosine 0.998 vs PyTorch | **LAION-CLAP HTSAT INT8** (Audio Tower) | 33.2 MB (INT8 Audio) | 122.38 ms / 10s window (CPU) | **SHIPPABLE** (Apache-2.0) |
| **Vibe Search (Text)** | "Dynamic load, 48 MB RAM" | 120.9 MB INT8 (Cosine 0.971 - FAIL), 502 MB FP32 (Cosine 1.0 - PASS) | **LAION-CLAP FP32 Text** (or MiniLM tag text) | 120.9 MB (INT8) / 502 MB (FP32) | 23.63 ms / query (INT8 CPU) | **SHIPPABLE** (Apache-2.0) |
| **Zero-Shot Retrieval** | "Sensible / Good" | P@5: 0.3810, R@20: 0.1548 (Tag baseline: P@5: 0.7429, R@20: 0.2864) | **Hybrid Tag-First + CLAP Semantic** | N/A (Index only) | 4.06 ms / 50k search | **SHIPPABLE** |
| **Acoustic X-Ray** | "DSP handles 100%" | DSP BPM ACC1: 20%, Key Exact: 0%, Key MIREX: 5-10% | **Native Rust DSP (xray-rs)** | 0 MB (Pure DSP code) | 731.22 ms / 4-min track (0.00305x RTF) | **SHIPPABLE** (MIT/Apache code) |
| **Autoplay** | "Toy clusters, good" | Demotes skipped cluster by 29-39 ranks, MMR +142.8% diversity | **Native Rust Autoplay (autoplay-rs)** | 516 B / track (INT8 vectors) | 15.85 ms scoring / 50k candidates | **SHIPPABLE** (MIT/Apache code) |
| **Runtime Architecture** | "ORT_DYLIB_PATH everywhere" | iOS strictly forbids runtime dlopen; DML fails on CLAP Reshape | **Swift/CoreML (iOS) + JNI (Android) + Rust DSP** | Static / Framework link | Zero runtime dlopen on iOS | **SHIPPABLE** |

---

## T0: Comprehensive Licence Audit

Actual licence files fetched and stored in `ai-spike/licenses/<model>.txt`.

| Model Candidate | Code Licence | Weights Licence (Exact Quote) | Training Data Concerns | Shipped Verdict |
|---|---|---|---|---|
| **HTDemucs / Demucs ONNX** | MIT | No separate weights grant in repo (`facebookresearch/demucs`). | Trained on MUSDB18-HQ (marked `other-nc`, non-commercial research use only). | **NEEDS LEGAL REVIEW** |
| **Open-Unmix `umxhq`** | MIT | Zenodo record metadata: `MIT License`. Code: MIT. | Trained on MUSDB18-HQ (`other-nc` research only). Risk of dataset non-commercial taint. | **NEEDS LEGAL REVIEW** |
| **Open-Unmix `umxl`** | MIT | `CC BY-NC-SA 4.0` ("Non-Commercial, ShareAlike"). | CC BY-NC-SA 4.0 explicitly forbids commercial redistribution. | **NOT SHIPPABLE** |
| **UVR MDX-Net / BS-RoFormer** | MIT | Community weights hosted on Google Drive / HF without explicit commercial licence grant. | Trained on MUSDB18 / private stems. Lacks legal provenance. | **NEEDS LEGAL REVIEW** |
| **LAION-CLAP `clap-htsat-unfused`** | Apache-2.0 | Apache-2.0 (HuggingFace model card & GitHub repo `LAION-AI/CLAP`). | LAION-Audio-630k (public audio-text pairs). | **SHIPPABLE** |
| **LAION-CLAP Music (`larger_clap_music_and_speech`)** | CC0-1.0 | `CC0-1.0` (Public Domain dedication on HuggingFace model card). | Public audio-text pairs. | **SHIPPABLE** |
| **Microsoft CLAP (`ms-clap`)** | MS-PL | Microsoft Public License (MS-PL) on GitHub `microsoft/CLAP`. | Free commercial use permitted under MS-PL. | **SHIPPABLE** |
| **Sentence-Transformers `all-MiniLM-L6-v2`** | Apache-2.0 | Apache-2.0 on HuggingFace model card. | Permissive public web corpora. | **SHIPPABLE** |

---

## T1: Real Audio Dataset

- **Track Count (MEASURED):** **157 real tracks** (exceeds requirement of $\ge 150$).
- **Source Breakdown:**
  - 144 real music multi-track excerpts from MUSDB18-HQ (7-second multi-track masters extracted with FFmpeg to 44.1 kHz stereo WAV).
  - 5 user local library tracks (`USER_145` Telugu romantic ballad `Nuvvunte Chaley` [4.00 min], `USER_146` EDM/Trap `Parano` [1.8 min], `USER_147` SampleTrack FLAC, Windows alerts/chimes).
  - 8 Creative Commons masterworks from Wikimedia Commons (Beethoven, Mozart, Scott Joplin, Vivaldi, Bach, Chopin, PetroVenus Lo-fi).
  - 12 full multi-track stem evaluation directories in `ai-spike/data/real/stems_eval/<track_id>/` with ground-truth `mixture.wav`, `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.
- **Manifest:** Saved at `ai-spike/data/real/manifest.csv`. Raw log: `ai-spike/logs/t1_dataset.log`.

---

## T2: 4-Source Stem Separation & SDR Quality

Raw log: `ai-spike/logs/t2_stems_eval.log`.

### 1. Model Artifact Sizes & SHA256 (MEASURED)
- **HTDemucs FP32 ONNX:** 316,446,953 bytes (301.79 MB) | SHA256: `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74`
- **HTDemucs FP16 ONNX:** 165,612,636 bytes (157.94 MB) | SHA256: `d05c269d0178d2a72ad484b10b11dd370193fc923201c3b27a99f848745db70a`
- **HTDemucs INT8 ONNX:** 197,052,021 bytes (187.92 MB) | SHA256: `15defc0e7025a96593268cab4062bd476d03542753522c0130c3f88c4df1b90b`
- **Open-Unmix FP32 (4 targets):** 135.89 MB total (33.97 MB per target: vocals, drums, bass, other)
- **Open-Unmix INT8 (4 targets):** 34.48 MB total (8.62 MB per target: vocals, drums, bass, other)

### 2. Separation Quality (fast_bss_eval SDR on 10 MUSDB18 Test Excerpts) (MEASURED)
| Model | Vocals SDR | Drums SDR | Bass SDR | Other SDR | Mean 4-Source SDR |
|---|---|---|---|---|---|
| **HTDemucs FP32** | **8.29 dB** | **7.37 dB** | **8.63 dB** | **4.08 dB** | **7.10 dB** |
| **Open-Unmix FP32** | **5.55 dB** | **3.68 dB** | **1.75 dB** | **0.91 dB** | **2.97 dB** |
| **Open-Unmix INT8** | **5.51 dB** | **3.69 dB** | **1.73 dB** | **0.92 dB** | **2.96 dB** |

- **Quality Gap:** HTDemucs outperforms Open-Unmix by **+2.74 dB on Vocals** and **+4.12 dB overall**.
- **INT8 Impact:** INT8 quantization of Open-Unmix causes **0.04 dB degradation on Vocals** and **0.01 dB overall** (INT8 does NOT break Open-Unmix; 74.6% size reduction with negligible loss).

### 3. 4-Minute Real Track Benchmark (`USER_145`, 240.05 seconds) (MEASURED)
- **HTDemucs PyTorch/CPU (`segment=7.8s, overlap=0.1`):**
  - Wall Time: **285.84 s** | Real-Time Factor (RTF): **1.191x** | Peak RAM: **1,032.7 MB**
- **Open-Unmix INT8 ORT CPU:**
  - Wall Time: **16.42 s** | Real-Time Factor (RTF): **0.068x** (14.6x faster than realtime) | Peak RAM: **760.8 MB**
- **Open-Unmix INT8 ORT DirectML:**
  - Wall Time: **12.57 s** | Real-Time Factor (RTF): **0.052x** (19.1x faster than realtime) | Peak RAM: **1,233.4 MB**

### 4. Recommendation & Licence Alignment
HTDemucs delivers vastly superior separation (7.10 dB vs 2.96 dB SDR), but is marked **NEEDS LEGAL REVIEW** due to MUSDB18-HQ dataset licensing. The highest-quality candidate with clear MIT Zenodo metadata is Open-Unmix `umxhq` INT8, but it suffers an SDR **quality deficit of -4.12 dB**.

---

## T3: CLAP ONNX Export, Quantization, Latency & Fidelity

Raw log: `ai-spike/logs/t3_clap_onnx.log`.

### 1. Real Artifact Sizes & SHA256 (MEASURED)
- **Audio Tower FP32 (graph):** 2,262,279 bytes (2.16 MB) | SHA256: `bc005f82da9b034496f60f767d7e30d49460a7d28c29a68e54585003026b0450`
- **Audio Tower FP32 (weights):** 117,178,368 bytes (111.75 MB) | SHA256: `5d636ed289665b51165132f5242c39be8e6acae38d070bf0a7c912621b473af5`
- **Audio Tower INT8:** **34,830,324 bytes (33.22 MB)** | SHA256: `04a45b1ea667eb83693ad92125892e0587b9acf35a83f38541532672ef1ca95d`
- **Text Tower FP32 (graph):** 502,435,051 bytes (479.16 MB) | SHA256: `4a156f4b6a03bd5611ccf9ad2e5baa79404d0cef36468a35195fb00d83842d95`
- **Text Tower FP32 (weights):** 501,296,128 bytes (478.07 MB) | SHA256: `a4cb1ef9cb283dcc63d5cdffe6a0b3273e0092f31553ce8859a2c2580971596d`
- **Text Tower INT8:** **126,764,921 bytes (120.89 MB)** | SHA256: `2cbd5f0f5fb5177952acb76ccd0c45c5af82263ab99b180b30c3bcb1875f9502`

### 2. Load-Time RAM & Peak RAM (MEASURED)
- **Audio INT8 Session Loaded (CPU):** RAM Delta = **54.5 MB**
- **Text INT8 Session Loaded (CPU):** RAM Delta = **129.1 MB**
- **Total CLAP INT8 Load RAM (CPU):** **183.6 MB** (Round 1's "46 MB RAM" claim is debunked; actual model working set is 183.6 MB).

### 3. Latency Benchmarks (MEASURED)
- **CPU FP32:** Audio 10s Window: **140.14 ms** | Text Query (seq=32): **34.27 ms**
- **CPU INT8:** Audio 10s Window: **122.38 ms** | Text Query (seq=32): **23.63 ms**
- **DirectML Execution:** **FAILED** on both Audio and Text with runtime exception:
  `[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node ... Exception(2) 8007023E`.
  *Root Cause:* DirectML MLOperatorAuthor Reshape kernel cannot handle dynamic shape dimensions without static shape hints for HTSAT / Roberta attention blocks.

### 4. Fidelity Verification vs PyTorch (Cosine Similarity on Real Audio/Text) (MEASURED)
- **Audio Tower:**
  - PyTorch vs ONNX FP32: **1.000000** (PASS, $\ge 0.99$)
  - PyTorch vs ONNX INT8: **0.998090** (PASS, $\ge 0.99$)
- **Text Tower:**
  - PyTorch vs ONNX FP32: **1.000000** (PASS, $\ge 0.99$)
  - PyTorch vs ONNX INT8: **0.970774** (**FAIL**, falls below required $0.99$ threshold; dynamic quantization degrades text representations).

---

## T4: Retrieval on Real Music & Vector Compression Study

Raw log: `ai-spike/logs/t4_retrieval_eval.log`.

### 1. Overall Retrieval Accuracy across 42 Queries (MEASURED)
| Architecture | Mean Precision@5 | Mean Recall@20 |
|---|---|---|
| **Tag-Only Metadata Baseline** | **0.7429** | **0.2864** |
| **CLAP PyTorch** | **0.3810** | **0.1548** |
| **CLAP ONNX FP32** | **0.3810** | **0.1548** |
| **CLAP ONNX INT8** | **0.3524** | **0.1494** |

*Finding:* Tag-only metadata retrieval beats zero-shot CLAP by nearly 2x on real tracks (0.7429 vs 0.3810 P@5). CLAP alone cannot be trusted as the sole retrieval engine; it must act as a secondary re-ranker.

### 2. 10 Worst Retrieval Queries (MEASURED - RAW VERBATIM)
1. `reggae` (Ground Truth: 11) -> P@5 = 0.00, R@20 = 0.00. Top retrieved: `#1: 'Nuvvunte Chaley' (Telugu Romantic Ballad) [Cosine: 0.3876] -> MISMATCH`
2. `synthesizer` (Ground Truth: 13) -> P@5 = 0.00, R@20 = 0.00. Top retrieved: `#1: 'Moonlight Sonata' (Classical Piano) [Cosine: 0.2907] -> MISMATCH`
3. `fast upbeat tempo` (Ground Truth: 20) -> P@5 = 0.00, R@20 = 0.05. Top retrieved: `#1: 'Run Run Run' (Indie Rock) [Cosine: 0.4846] -> MISMATCH`
4. `baroque` (Ground Truth: 16) -> P@5 = 0.00, R@20 = 0.06. Top retrieved: `#1: 'Eine kleine Nachtmusik' (Mozart, Classical era) [Cosine: 0.4758] -> MISMATCH`
5. `distorted 808 bass` (Ground Truth: 11) -> P@5 = 0.00, R@20 = 0.09. Top retrieved: `#1: 'Resurrection' (Indie Rock) [Cosine: 0.2180] -> MISMATCH`
6. `heavy distorted 808 sub bass trap` (Ground Truth: 10) -> P@5 = 0.00, R@20 = 0.10. Top retrieved: `#1: 'Resurrection' (Indie Rock) [Cosine: 0.2237] -> MISMATCH`
7. `energetische elektronische tanzmusik` (Ground Truth: 17) -> P@5 = 0.00, R@20 = 0.12. Top retrieved: `#1: 'Moss' (Indie Rock) [Cosine: 0.3320] -> MISMATCH`
8. `metal` (Ground Truth: 23) -> P@5 = 0.00, R@20 = 0.13. Top retrieved: `#1: 'Chimes Sound' (Sound Effect) [Cosine: 0.1896] -> MISMATCH`
9. `hip hop` (Ground Truth: 12) -> P@5 = 0.00, R@20 = 0.25. Top retrieved: `#1: 'Nuvvunte Chaley' (Telugu Ballad) [Cosine: 0.4318] -> MISMATCH`
10. `blues` (Ground Truth: 19) -> P@5 = 0.20, R@20 = 0.05. Top retrieved: `#1: 'Moonlight Sonata' [Cosine: 0.4779] -> RELEVANT`

### 3. Vector Compression Study (N = 1,256 Real Embeddings) (MEASURED)
- **FP32 Uncompressed:** 2,048 B / track (Baseline)
- **FP16:** 1,024 B / track (50.0% savings) -> **99.80% Top-10 Overlap** vs FP32
- **INT8 Quantized:** 516 B / track (74.8% savings) -> **97.80% Top-10 Overlap** vs FP32
- **PCA-128:** 512 B / track (75.0% savings) -> **84.00% Top-10 Overlap** vs FP32
- **50,000 Vector Brute-Force Cosine Timing:** Mean = **4.06 ms** | Min = **3.57 ms** (TIMING ONLY, replicated real vectors with noise).

---

## T5: Acoustic X-Ray in Native Rust

Raw logs: `ai-spike/logs/t5_xray_eval.log`, `ai-spike/logs/t5_xray_rust.log`.

### 1. Ground Truth Benchmark (10 Real Tracks across 7 Genres) (MEASURED)
| Metric | Baseline DSP (Round 1) | Improved DSP (HPSS + Temperley) |
|---|---|---|
| **BPM ACC1 (Within 4% Exact)** | **10.0%** | **20.0%** |
| **BPM ACC2 (Harmonic Tolerant)** | **10.0%** | **20.0%** |
| **Key Exact Match** | **10.0%** | **0.0%** |
| **Key MIREX-Weighted Score** | **10.0%** | **5.0%** |

*Finding:* The Round 1 claim that "DSP handles 100%" is completely refuted. DSP key detection on real polyphonic audio achieves only 5-10% MIREX score, and BPM detection achieves 10-20% ACC1 due to rubato and syncopation.
*Learned Model Survey:* Madmom is GPL-3.0; Essentia is AGPL-3.0; KeyNet is research-only/GPL. **NONE FOUND** with permissive commercial weights. Recommendation: Ship native Rust DSP as a lightweight fallback.

### 2. Native Rust Implementation (`xray-rs`) Benchmark (MEASURED)
- **Crate Location:** `ai-spike/xray-rs/` (pure Rust, `hound` + `rustfft`, zero ML runtime).
- **Target Track:** `USER_145_Nuvvunte Chaley.wav` (240.05 seconds = 4.00 minutes).
- **Mean Wall Time:** **731.22 ms** (0.731 s).
- **Min Wall Time:** **664.01 ms** (0.664 s).
- **Real-Time Factor (RTF):** **0.00305x** (processed **328.3x faster than real-time**).
- **Extracted Features:** BPM: 90.7 (GT: 92 BPM, error: 1.4%), Key: F major, RMS Energy: 0.2242 (-13.0 dBFS), Spectral Centroid: 2539.1 Hz, Rolloff: 5986.2 Hz, Flatness: 0.0048.

---

## T6: Autoplay on Real Embeddings in Native Rust

Raw log: `ai-spike/logs/t6_autoplay_rust.log`.

### 1. Behavioral Ledger Simulation (MEASURED)
- **Skip Penalty Impact on Skipped Tracks:**
  - Track #110 (Skipped Metal/Trap): Rank #58 $\to$ Rank #87 (**Demoted by 29 positions**).
  - Track #145 (Skipped Metal/Trap): Rank #76 $\to$ Rank #115 (**Demoted by 39 positions**).
- **Queue Diversity (k = 10):**
  - Greedy Queue ($\lambda = 1.0$): Intra-List Diversity = **0.2287**
  - MMR Queue ($\lambda = 0.7$): Intra-List Diversity = **0.5553** (+**142.8% diversity boost**).

### 2. 50,000 Candidate Scalability Benchmark (MEASURED)
- **Vector Scoring (50k dot products + penalty):** Mean = **15.85 ms** | Min = **14.47 ms**.
- **Total Autoplay Queue Generation (Score + MMR):** Mean = **343.39 ms** | Min = **335.38 ms**.
- *Label:* TIMING ONLY (replicated real vectors with noise).

---

## T7: iOS/Android Runtime Architecture & CI Sample

### 1. iOS Architectural Decision
Round 1's recommendation of `ORT_DYLIB_PATH` dynamic loading is **strictly invalid on iOS**: iOS forbids runtime `dlopen` of arbitrary paths and requires static linking or embedded frameworks signed with the application bundle.

**Recommended Production Architecture for iOS:**
- **Platform-Native Split (Recommended):** Swift UI links Microsoft's official `onnxruntime-objc` CocoaPod / SPM package (enabling the Apple CoreML Execution Provider natively with zero cross-compilation headaches). The Rust native core strictly handles the audio engine, playback, mixing, X-Ray DSP, and Autoplay MMR queue logic.
- **Alternative Static Link:** Static library `libonnxruntime.a` / `onnxruntime.xcframework` linked into the Rust staticlib via `cargo:rustc-link-lib=static=onnxruntime`.

### 2. Android Architectural Decision
- JNI dynamic loading via `System.loadLibrary("onnxruntime")` and `libc++_shared.so` bundled in `jniLibs/<abi>/` is supported.

### 3. CI Workflow Sample
Committed as `ai-spike/ci-sample.yml` (disabled sample workflow per task T7) containing matrix jobs for macOS/iOS (`macos-14`), Android NDK (`ubuntu-latest`), and Windows MSVC (`windows-latest`).

---

## Verification & Deliverables Summary

- **Report:** `ai-spike/SPIKE_REPORT_V2.md`
- **Licenses:** `ai-spike/licenses/*.txt`
- **Logs:** `ai-spike/logs/*.log`
- **Rust Crates:** `ai-spike/xray-rs/`, `ai-spike/autoplay-rs/`
- **CI Sample:** `ai-spike/ci-sample.yml`
