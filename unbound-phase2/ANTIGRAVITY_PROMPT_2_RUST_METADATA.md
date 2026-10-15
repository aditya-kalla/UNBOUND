# Prompt 2 for Antigravity - Rust tag reader + engine state (run AFTER Prompt 1 is merged)

---

You are a senior Rust + Kotlin engineer. Project: UNBOUND (Windows). Read `PHASE2_ARCHITECTURE.md` (D4 and "Known gaps" 1-3). Work in `rust-audio-engine` and the Kotlin bridges. Keep the existing C ABI behaviour unchanged except where stated; bump `engine_abi_version()` to **3** and the Kotlin `EXPECTED_ABI_VERSION` to 3.

## Task A - panic safety (do this first)
`Cargo.toml` has `[profile.release] panic = "abort"`: any panic (e.g. inside a tag parser on a malformed file) would kill the whole JVM. Change to unwind, and wrap the BODY of **every** `extern "C"` function in `std::panic::catch_unwind` (return an error code / null / -1 on panic, log it). Add a test that forces a panic inside a wrapped function and proves the process survives. Re-run the existing sim-harness suite (copy the new lib.rs/stage.rs into `sim-harness/engine/src`) - all 11 must still pass.

## Task B - engine end-of-track signal and duration
- `engine_duration_ms(handle) -> i64`: from symphonia `n_frames` / `time_base` of the playing track; -1 if unknown. Set at `play()`.
- `engine_state(handle) -> i32`: 0 IDLE, 1 PLAYING, 2 PAUSED, 3 ENDED, 4 ERROR. ENDED = decoder reached EOF **and** the ring buffer has been fully drained by the audio callback (set an atomic from the audio thread when it pops 0 samples while `at_eof` is set and the ring is empty; never allocate or lock anything new in that callback). A seek after ENDED returns the state to PLAYING. `stop()` -> IDLE; a failed `play()` -> ERROR.
- Kotlin: `expect class AudioEngine` gets `fun durationMs(): Long` and `fun playbackState(): PlaybackState` (`enum class PlaybackState { IDLE, PLAYING, PAUSED, ENDED, ERROR }`), implemented in both actuals through the bridge. Extend the sim-harness tests (ENDED reached ~at track length, seek-after-end resumes) and `DesktopSmokeTest`.

## Task C - tag/metadata reader (the new FFI)
Add a module `meta.rs` using the `lofty` crate (also `blake3` for hashing; `image` with default-features off + jpeg/png/webp for thumbnails):
```
extern "C" fn meta_read_json(path: *const c_char, art_dir: *const c_char) -> *mut c_char   // UTF-8 JSON, caller frees
extern "C" fn meta_free_string(ptr: *mut c_char)
```
Success JSON (all keys always present, `null` when unknown; arrays may be empty):
`{"ok":true,"title":..,"artists":[..],"album_artist":..,"album":..,"genres":[..],"track_no":..,"disc_no":..,"year":..,"duration_ms":..,"bitrate_kbps":..,"sample_rate":..,"channels":..,"bit_depth":..,"codec":"mp3|flac|aac|alac|vorbis|opus|wav|aiff|ape|wavpack|other","lossless":bool,"rg_track_gain_db":..,"rg_track_peak":..,"rg_album_gain_db":..,"rg_album_peak":..,"has_embedded_lyrics":bool,"artwork":{"hash":..,"path":..,"width":..,"height":..}|null,"fingerprint":..,"meta_version":1}`
Failure: `{"ok":false,"error":"..."}` (never panic, never return null except on allocation failure).
Rules: multi-value artist/genre fields -> arrays (ID3v2.4 and Vorbis multi-value, plus ID3v2.3 `/` or null-separated where lofty exposes them); do NOT split a single string on `/ & ,` (AC/DC, Simon & Garfunkel). ReplayGain: parse strings like "-6.50 dB" to f64. Year from recording date if the year tag is missing. `lossless` for flac/wav/aiff/ape/wavpack/alac. **fingerprint** = `"<file_size>:<blake3 hex of first 64 KiB + last 64 KiB>"` (hash the whole file if <= 128 KiB). **Artwork**: prefer front cover else first picture; hash the original bytes with blake3; if `<art_dir>/<hash>.jpg` doesn't exist, decode, downscale so the longest side <= 600 px, save JPEG q85 (if decoding fails, save the original bytes with their extension); report the stored file's width/height. Run `catch_unwind` as in Task A. It must handle: no tags, truncated files, zero-byte files, non-audio files, non-UTF-8 text, paths with spaces/non-ASCII characters.
Tests (generate fixtures in code; do not commit binaries): tagged WAV/FLAC/MP3-like files written with lofty; multi-artist; ReplayGain; embedded art dedupe (same art in two files -> one hash/path); corrupt files return `ok:false`; fingerprint is stable and changes when bytes change; unicode path.

## Task D - Kotlin side in `:data` (jvm source set shared by android+desktop)
Create an intermediate source set (e.g. `jvmCommonMain`, `dependsOn(commonMain)`, with androidMain and desktopMain depending on it) containing `NativeMetadataReader : MetadataReader` (uses JNA, `Native.load("unbound_audio")`, same UTF-8 `jna.encoding` + ABI check as the audio bridge; frees the native string in `finally`). Parse the JSON with kotlinx.serialization (add the plugin/dependency) into `ParsedTrack` (+`ParsedArtwork`); `ok:false` -> return null. The `read(file)` call is blocking: callers dispatch it on an IO dispatcher. Provide `DesktopMediaFileSource : MediaFileSource` (recursive directory walk for mp3, flac, wav, m4a, aac, ogg, opus, wma, aiff, ape, wv; skip hidden dirs; size+mtime from `File`; throws if the root doesn't exist or isn't readable) and a `desktopRun`-style smoke program that scans a folder you pass in, writes it to a temp database through `SqlDelightLibraryWriter`, prints the summary and 5 sample rows, then rescans and prints the (all-unchanged) summary.

## Deliverable
(1) every changed/new file in full, (2) complete console output: cargo tests, sim-harness run, the Kotlin scan smoke program on a real music folder of mine (`C:\Users\lenovo\Music` or any folder with mp3/flac) including timings, (3) anything unverified. Keep all earlier behaviour and tests green.
