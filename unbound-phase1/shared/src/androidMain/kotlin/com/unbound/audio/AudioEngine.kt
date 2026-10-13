package com.unbound.audio

/**
 * Android actual. Loads `libunbound_audio.so` (built by rust-audio-engine for
 * arm64-v8a / armeabi-v7a / x86_64) and delegates to the shared JNA bridge.
 *
 * Kept as a thin wrapper: all the real binding logic lives in
 * NativeAudioEngineBridge (in this same source set's parent, shared across
 * android/desktop) so the two `actual` classes don't drift.
 */
actual class AudioEngine actual constructor() {

    private val bridge = NativeAudioEngineBridge(libraryName = "unbound_audio")

    actual fun play(uri: String) = bridge.play(uri)
    actual fun pause() = bridge.pause()
    actual fun resume() = bridge.resume()
    actual fun stop() = bridge.stop()
    actual fun seekTo(positionMs: Long) = bridge.seekTo(positionMs)
    actual fun currentPositionMs(): Long = bridge.currentPositionMs()
    actual fun durationMs(): Long = bridge.durationMs()
    actual fun playbackState(): PlaybackState = bridge.playbackState()
    actual fun setPcmListener(listener: PcmListener?) = bridge.setPcmListener(listener)
    actual fun release() = bridge.release()
}
