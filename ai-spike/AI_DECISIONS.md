# UNBOUND - AI layer decisions (closing the feasibility spike)

Source: Antigravity spike rounds 1-3, reviewed by Lead Architect. Numbers are MEASURED on a Ryzen 5 5500U laptop unless marked. This file is the input for Phase 4.

## 1. Stems (Acoustic X-Ray stem separation + karaoke mixer)
- **Architecture:** a `StemModel` slot (id, files, sample rate, targets, STFT params, licence status). The engine only knows "4 stems on disk"; the model is swappable. This is what lets the legal build and the community build ship different models.
- **Desktop (Win/Mac):** HTDemucs, FP16-weights ONNX (158 MB), ORT CPU, 7.8 s segments, 25% overlap. 7.13 dB mean SDR (same as PyTorch 7.10). About 190 s for a 4-minute track, 1.7-3 GB peak RAM. Background job, cached, never blocks playback.
- **Rejected:** HTDemucs INT8 (188 MB, 3.1x slower, SDR collapses to 3.56 dB); segments of 4 s or 2 s (-0.33 / -1.23 dB, 2-4x slower); DirectML (out-of-memory error on all variants, so CPU only for now).
- **Phones:** Open-Unmix INT8 + multichannel Wiener (34 MB for 4 targets, 3.63 dB mean; vocals 5.9, bass 2.5, other 1.5). Opt-in, charging only, labelled "lite quality". Never auto-run.
- **Licence gate (blocks store release, not development):** Spike round 3 marked HTDemucs "MIT / shippable". I overrule that: the repo states MIT for code only, and the weights are trained on research-only MUSDB18-HQ. Both HTDemucs and Open-Unmix `umxhq` are NEEDS LEGAL REVIEW. `umxl` is non-commercial. Need a legal opinion or the authors' permission before any store release.

## 2. Vibe Search and "More like this"
- **Index:** CLAP audio tower (INT8, cosine 0.998 vs PyTorch), 512-dim, stored as INT8 (516 B/track; 97.8% top-10 overlap vs FP32). Brute-force cosine, about 4 ms for 50k tracks. No ANN index.
- **More like this / Neural Autoplay (the strong feature):** audio-to-audio P@5 0.47, nDCG@10 0.46. Make this the flagship.
- **Free-text search (weaker, ship as beta):** P@5 0.19 vs 0.09 random. Text tower FP16 (240 MB, optional download). INT8 variants fail fidelity (cosine 0.92-0.98). Do not oversell.
- **Open items for Phase 4:** (a) evaluate the music-trained CLAP checkpoint, since the tested model is the general audio one; (b) verify the audio-tower INT8 file size (round 2 measured 33 MB, the round 3 table says 154 MB); (c) measure retrieval quality of the 1,000-phrase vocabulary option (round 3 measured the wrong metric).

## 3. Acoustic X-Ray analysis (BPM, key, energy, timbre, sections)
- Pure Rust DSP, no weights, 0.63 s per 4-minute track.
- **BPM:** ACC2 81%, ACC1 49% on 43 EDM tracks. Prefer the ID3 BPM tag when present, DSP otherwise. Evaluate Beat This! (code MIT; confirm the weights licence ourselves) in Phase 4.
- **Key:** 11.6% exact, 19.8% MIREX on those tracks: near chance (4%), so likely a bug (mapping or mode). Fix with unit tests on known chord progressions. Ship as best-effort with a confidence value and prefer the ID3 key tag when present.
- Energy, brightness and section boundaries are not yet evaluated on real music; add listening checks in Phase 4.

## 4. Neural Autoplay and Synesthesia
- Rust only, no weights. Autoplay: recent-listens centroid, skip penalty, MMR diversity (+143% list diversity); 16 ms scoring for 50k candidates. Synesthesia: real-time DSP features from the engine.

## 5. Runtime
- One ONNX Runtime stack. Windows / macOS / Android: `ort` crate inside the Rust core. iOS: no runtime dlopen; static link, or `onnxruntime-objc` from Swift with the pipeline logic kept in Rust. NOT VERIFIED; to be settled by the CI workflow on a Mac runner.
- Mobile accelerators (NNAPI, CoreML) are not worth it for these models (LSTM and ConvTranspose fragment the graph). Use CPU / XNNPACK.

## 6. Phase 4 build order
1. Rust X-Ray DSP + key fix + Synesthesia features
2. ORT in the Rust core + CI for Android / iOS / macOS
3. CLAP audio indexing + "More like this" + Neural Autoplay
4. Stems desktop (HTDemucs slot) + multi-stream mixer; phones with the lite slot
5. Free-text Vibe Search (beta) after the music-CLAP comparison
