# UNBOUND — Phase 1 scaffold

Kotlin Multiplatform + Rust FFI bridge. This is the foundation only: it proves
the KMP↔Rust wiring end-to-end (build, load, call, callback) on desktop and
Android. It does **not** yet decode or play audio — see "What's stubbed" below.

## Layout

```
settings.gradle.kts / build.gradle.kts   root Gradle config
shared/                                  KMP module (androidTarget + jvm("desktop"))
  src/commonMain/.../AudioEngine.kt          expect class + PcmListener contract
  src/commonMain/.../NativeAudioEngineBridge.kt  JNA binding to the Rust cdylib
  src/androidMain/.../AudioEngine.kt         actual (Android)
  src/jvmMain/.../AudioEngine.kt             actual (Windows/macOS/Linux)
rust-audio-engine/
  Cargo.toml
  src/lib.rs                             C ABI + Engine skeleton (cpal + symphonia deps declared, decode not wired)
```

## Building

```bash
# 1. Build the Rust cdylib for your host platform
cd rust-audio-engine
cargo build --release
# -> target/release/{libunbound_audio.so | libunbound_audio.dylib | unbound_audio.dll}

# 2. Put the built library where JNA/the JVM will find it, e.g. on java.library.path
#    or shared/src/desktopMain/resources/<os-arch>/ once the desktop app module exists.

# 3. Build the KMP module
cd ..
./gradlew :shared:build
```

Android cross-compilation (aarch64-linux-android, armv7-linux-androideabi,
x86_64-linux-android via `cargo-ndk`) isn't wired yet — add it when the
`androidApp` module lands in Phase 3, alongside `jniLibs` packaging.

## What's stubbed (and why)

`engine_play` in `lib.rs` doesn't decode or output audio yet. It validates
the FFI path (handle lifecycle, string marshalling, callback registration)
but the actual symphonia decode thread → ring buffer → cpal output stream
wiring is left as a `TODO`.

That's deliberate, not an oversight: per the project's delegation protocol,
real-time audio DSP is exactly the kind of token-heavy, correctness-critical
work that should go to Antigravity as an isolated task and come back for
review, rather than being written inline here. Suggested delegation prompt:

> **System role:** Senior Rust audio engineer, expert in `cpal` and `symphonia`.
>
> **Task:** Complete the `TODO` in `Engine::play()` in the attached `lib.rs`.
> Requirements:
> 1. Use `symphonia::default::get_probe()` to open the file at `uri`, pick the
>    default audio track, and decode packets into interleaved `f32` samples
>    at the track's native sample rate/channel count.
> 2. Run decoding on a dedicated thread; push decoded samples into the
>    existing `HeapRb<f32>` ring buffer (`EngineInner.ring`). Handle
>    backpressure by blocking the decode thread when the ring is full —
>    never drop samples silently.
> 3. Build a `cpal` output stream on the default output device
>    (`cpal::default_host().default_output_device()`), matching the
>    device's supported sample rate via a resampler if the decoded rate
>    doesn't match (use `rubato` for resampling).
> 4. In the cpal output callback: pull samples from the ring to fill the
>    output buffer (zero-fill on underrun, never panic or block), and
>    accumulate pulled samples until you have ~32ms worth, then invoke
>    `EngineInner.pcm_callback` (if set) with a copy of that chunk plus the
>    stream's actual sample rate and channel count.
> 5. Update `EngineInner.position_ms` as decoding progresses (based on
>    samples consumed by the output callback, not samples decoded, so it
>    reflects what's actually audible).
> 6. Keep the existing public `Engine`/C-ABI surface unchanged — only fill
>    in `play()`, `pause()`/`resume()` (stream pause, don't tear down),
>    `stop()`, and `seek_to()` (flush the ring, reposition the symphonia
>    reader, restart decode).
>
> Do not add gapless playback, crossfade, EQ, or WSOLA — those are later
> phases. Return the complete updated `lib.rs`.

## Next steps

1. Run the delegation prompt above through Antigravity, review the diff,
   merge it back into `rust-audio-engine/src/lib.rs`.
2. Smoke-test: a minimal JVM `main()` in `shared` (or a throwaway test) that
   constructs `AudioEngine()`, calls `play()` with a local MP3/FLAC path, and
   confirms audio comes out of the default device.
3. Once that's solid, move to Phase 2 (DuckDB + SQLDelight telemetry layer).
