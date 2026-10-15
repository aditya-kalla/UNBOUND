package com.unbound.audio

/**
 * The shared contract every platform's AudioEngine implementation must satisfy.
 *
 * The real work happens in Rust (rust-audio-engine, built on miniaudio). Each
 * `actual` binds to that native library through the generated uniffi/JNA
 * bindings. Nothing in commonMain touches JNI or platform audio APIs directly
 * — that isolation is the whole point of the bridge.
 */
expect class AudioEngine() {

    /** Loads and begins playback of a local file. `uri` is a plain filesystem path in Phase 1. */
    fun play(uri: String)

    fun pause()

    fun resume()

    fun stop()

    /** Absolute seek, in milliseconds from the start of the current track. */
    fun seekTo(positionMs: Long)

    /** Current playback position in milliseconds; -1 if nothing is loaded. */
    fun currentPositionMs(): Long

    /** Duration of the playing track in milliseconds; -1 if unknown or nothing is loaded. */
    fun durationMs(): Long

    /** Current playback state of the engine. */
    fun playbackState(): PlaybackState

    /**
     * Registers a listener that receives raw PCM chunks (~32ms each) as they're
     * mixed, off the UI thread. This is dormant in Phase 1 — no consumer yet —
     * but the hook has to exist now so Phase 4 (ONNX feature extraction) and
     * Phase 5 (Synesthesia RMS/flux visuals) can attach without touching the
     * Rust engine again.
     */
    fun setPcmListener(listener: PcmListener?)

    /** Releases the native engine. Must be called when the engine is no longer needed. */
    fun release()
}

enum class PlaybackState { IDLE, PLAYING, PAUSED, ENDED, ERROR }

class AudioEngineException(val errorCode: Int, message: String) : RuntimeException(message)

/**
 * Receives raw PCM audio chunks (~32 ms each) from the native engine's
 * dedicated dispatch thread (`unbound-pcm-dispatch`), one call at a time.
 *
 * **Thread-safety contract — never call back into the engine from [onPcmChunk].**
 * [AudioEngine.stop], [AudioEngine.release], and [AudioEngine.setPcmListener]
 * join or wait for the dispatch thread before returning. If you call any of
 * them from inside [onPcmChunk] you will deadlock (the dispatch thread waits
 * for itself). Offload any engine-control work to a coroutine or handler on
 * another thread.
 *
 * The [pcm] array is only valid for the duration of the call; copy it if you
 * need it later.
 */
fun interface PcmListener {
    /**
     * @param pcm       interleaved 32-bit float samples for this chunk
     * @param sampleRate e.g. 44100 / 48000
     * @param channels   1 = mono, 2 = stereo
     */
    fun onPcmChunk(pcm: FloatArray, sampleRate: Int, channels: Int)
}
