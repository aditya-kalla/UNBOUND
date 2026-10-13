package com.unbound.audio

/**
 * Windows/macOS/Linux actual. Loads `unbound_audio.dll` / `libunbound_audio.dylib` /
 * `libunbound_audio.so` from the desktop app's native library path.
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
