package com.unbound.audio

import java.io.File
import java.util.concurrent.atomic.AtomicInteger

fun main() {
    println("=== UNBOUND Kotlin Desktop Smoke Test ===")

    val testFile = System.getProperty("unbound.testFile") ?: """C:\Windows\Media\Alarm01.wav"""
    println("Target audio file: $testFile")
    if (!File(testFile).exists()) {
        System.err.println("Error: Audio file not found at '$testFile'")
        System.exit(1)
    }

    println("[1/7] Creating AudioEngine...")
    val engine = AudioEngine()
    println("  AudioEngine created.")

    val chunksReceived = AtomicInteger(0)
    println("[2/7] Registering PCM listener...")
    engine.setPcmListener { pcm, sampleRate, channels ->
        val idx = chunksReceived.getAndIncrement()
        if (idx % 20 == 0 || idx < 3) {
            val firstSample = if (pcm.isNotEmpty()) pcm[0] else 0f
            println("  [PCM Listener] chunk #$idx: ${pcm.size} samples, ${sampleRate}Hz, ${channels}ch (first sample: ${"%.4f".format(firstSample)})")
        }
    }
    println("  PCM listener registered.")

    println("[3/7] Playing audio file...")
    engine.play(testFile)
    val duration = engine.durationMs()
    val initialState = engine.playbackState()
    println("  Playback started: state=$initialState, duration=${duration}ms")
    check(initialState == PlaybackState.PLAYING) { "Expected PLAYING but got $initialState" }

    println("[4/7] Monitoring playback for 1.5 seconds...")
    for (i in 0 until 15) {
        Thread.sleep(100)
        val pos = engine.currentPositionMs()
        val state = engine.playbackState()
        val chunks = chunksReceived.get()
        println("  Position: ${pos}ms / ${duration}ms | State: $state | PCM chunks received: $chunks")
    }

    println("[5/7] Testing Pause & Resume...")
    println("  Pausing...")
    engine.pause()
    val stateAtPause = engine.playbackState()
    val posAtPause = engine.currentPositionMs()
    println("  Paused state: $stateAtPause (pos: ${posAtPause}ms)")
    check(stateAtPause == PlaybackState.PAUSED) { "Expected PAUSED but got $stateAtPause" }
    Thread.sleep(500)
    val posAfterWait = engine.currentPositionMs()
    println("  Paused pos: ${posAtPause}ms -> After 500ms: ${posAfterWait}ms (delta: ${posAfterWait - posAtPause}ms)")

    println("  Resuming...")
    engine.resume()
    val stateAtResume = engine.playbackState()
    println("  Resumed state: $stateAtResume")
    check(stateAtResume == PlaybackState.PLAYING) { "Expected PLAYING but got $stateAtResume" }
    Thread.sleep(500)
    val posAfterResume = engine.currentPositionMs()
    println("  Resumed pos: ${posAfterResume}ms")

    println("[6/7] Testing Seek (to 200ms)...")
    engine.seekTo(200)
    Thread.sleep(500)
    val posAfterSeek = engine.currentPositionMs()
    val stateAfterSeek = engine.playbackState()
    println("  Position after seek: ${posAfterSeek}ms | State: $stateAfterSeek")

    println("[7/7] Stopping and releasing engine...")
    engine.stop()
    val stateAfterStop = engine.playbackState()
    println("  State after stop: $stateAfterStop")
    check(stateAfterStop == PlaybackState.IDLE) { "Expected IDLE but got $stateAfterStop" }
    engine.release()
    println("  Engine released cleanly.")

    val totalChunks = chunksReceived.get()
    println("\n=== Smoke Test Summary ===")
    println("Total PCM chunks received: $totalChunks")
    if (totalChunks > 0) {
        println("Status: SUCCESS! Audio engine decoded, streamed, and invoked callbacks properly from Kotlin.")
    } else {
        println("Status: WARNING - No PCM chunks received (check audio output device).")
    }
}
