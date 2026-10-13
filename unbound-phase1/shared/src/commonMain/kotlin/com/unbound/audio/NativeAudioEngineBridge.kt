package com.unbound.audio

import com.sun.jna.Callback
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer

/**
 * Thin JNA binding over rust-audio-engine's C ABI (see rust-audio-engine/src/lib.rs).
 *
 * NOTE: lives in commonMain for now because only the android/jvm targets use it
 * (JNA has no iOS story). When iOS is added in a later phase, move this to a
 * `jvmCommon` intermediate source set shared by androidMain + jvmMain, and give
 * iOS its own cinterop-based bridge instead — the commonMain `expect`/`actual`
 * contract in AudioEngine.kt doesn't need to change either way.
 */
internal class NativeAudioEngineBridge(libraryName: String) {

    private interface RustAudioLib : Library {
        fun engine_abi_version(): Int
        fun engine_new(): Pointer
        fun engine_play(handle: Pointer, uriUtf8: String): Int
        fun engine_pause(handle: Pointer)
        fun engine_resume(handle: Pointer)
        fun engine_stop(handle: Pointer)
        fun engine_seek_to(handle: Pointer, positionMs: Long)
        fun engine_current_position_ms(handle: Pointer): Long
        fun engine_duration_ms(handle: Pointer): Long
        fun engine_state(handle: Pointer): Int
        fun engine_set_pcm_callback(handle: Pointer, callback: PcmCallback?)
        fun engine_release(handle: Pointer)
    }

    /** Must match the extern "C" fn pointer signature in lib.rs exactly. */
    private interface PcmCallback : Callback {
        fun invoke(samples: Pointer, sampleCount: Int, sampleRate: Int, channels: Int)
    }

    init {
        // Rust expects UTF-8 paths; JNA defaults to the OS charset (breaks non-English song titles on Windows).
        System.setProperty("jna.encoding", "UTF-8")
    }

    private val lib: RustAudioLib = Native.load(libraryName, RustAudioLib::class.java).also { loaded ->
        val version = loaded.engine_abi_version()
        require(version == EXPECTED_ABI_VERSION) {
            "Stale native library: expected ABI version $EXPECTED_ABI_VERSION but loaded $version. " +
                "Replace the DLL/so/dylib with a matching build."
        }
    }
    private val handle: Pointer = lib.engine_new()

    // Held so it isn't garbage-collected out from under the native callback.
    private var activeCallback: PcmCallback? = null

    /** Codes: -1 can't open file, -2 unrecognised format, -3 no audio track, -4 decoder, -5 no output device,
     *  -6 device config, -7 stream, -8 resampler, -9 internal. */
    fun play(uri: String) {
        val code = lib.engine_play(handle, uri)
        if (code != 0) throw IllegalStateException("engine_play failed for '$uri' (code $code)")
    }
    fun pause() = lib.engine_pause(handle)
    fun resume() = lib.engine_resume(handle)
    fun stop() = lib.engine_stop(handle)
    fun seekTo(positionMs: Long) = lib.engine_seek_to(handle, positionMs)
    fun currentPositionMs(): Long = lib.engine_current_position_ms(handle)
    fun durationMs(): Long = lib.engine_duration_ms(handle)

    fun playbackState(): PlaybackState = when (lib.engine_state(handle)) {
        0 -> PlaybackState.IDLE
        1 -> PlaybackState.PLAYING
        2 -> PlaybackState.PAUSED
        3 -> PlaybackState.ENDED
        else -> PlaybackState.ERROR
    }

    fun setPcmListener(listener: PcmListener?) {
        if (listener == null) {
            activeCallback = null
            lib.engine_set_pcm_callback(handle, null)
            return
        }
        val callback = object : PcmCallback {
            override fun invoke(samples: Pointer, sampleCount: Int, sampleRate: Int, channels: Int) {
                val floats = samples.getFloatArray(0, sampleCount)
                listener.onPcmChunk(floats, sampleRate, channels)
            }
        }
        activeCallback = callback
        lib.engine_set_pcm_callback(handle, callback)
    }

    fun release() {
        activeCallback = null
        lib.engine_release(handle)
    }

    private companion object {
        /** Must match the value returned by `engine_abi_version()` in lib.rs. */
        const val EXPECTED_ABI_VERSION = 3
    }
}
