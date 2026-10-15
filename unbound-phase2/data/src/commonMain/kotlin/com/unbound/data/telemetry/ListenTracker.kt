package com.unbound.data.telemetry

import com.unbound.data.model.*

interface TimeSource {
    fun nowMs(): Long
    /** Device UTC offset right now, in minutes (e.g. 330 for India). Stored with every event. */
    fun tzOffsetMinutes(): Int
}

/**
 * Turns player signals into the append-only listen ledger (START ... END per listen).
 *
 * `listenedMs` is MEDIA time actually played: pauses and seeks don't count, and a position that
 * jumps further than wall-clock time allows (a seek the caller forgot to report) is capped, so
 * the numbers can't be inflated. Single-threaded: call every method from the same dispatcher.
 */
class ListenTracker(
    private val time: TimeSource,
    private val newListenId: () -> String,
    private val emit: (PlaybackEventDraft) -> Unit,
) {
    private class Active(
        val listenId: String, val trackId: Long, val durationMs: Long,
        var lastPos: Long, var lastWall: Long, var playing: Boolean,
        var listenedMs: Long, val context: PlayContext, val device: OutputDevice,
        var shuffle: Boolean, var repeat: RepeatMode, var speed: Float,
    )

    private var active: Active? = null
    val currentListenId: String? get() = active?.listenId

    fun onTrackStarted(
        trackId: Long, durationMs: Long, reason: StartReason, context: PlayContext, device: OutputDevice,
        startPositionMs: Long = 0, shuffle: Boolean = false, repeat: RepeatMode = RepeatMode.OFF, speed: Float = 1f,
    ) {
        active?.let { finish(it, it.lastPos, EndReason.REPLACED) }   // never leave a START without an END
        val a = Active(newListenId(), trackId, durationMs, startPositionMs, time.nowMs(), true, 0, context, device, shuffle, repeat, speed)
        active = a
        emit(draft(a, EventType.START, startPositionMs, startReason = reason))
    }

    /** Call roughly every 250 ms - 1 s while playing (and any time you read the position). */
    fun onPosition(positionMs: Long) {
        val a = active ?: return
        if (a.playing) credit(a, positionMs) else { a.lastPos = positionMs; a.lastWall = time.nowMs() }
    }

    fun onPause(positionMs: Long) {
        val a = active ?: return
        if (a.playing) credit(a, positionMs)
        a.playing = false; a.lastPos = positionMs
    }

    fun onResume(positionMs: Long) {
        val a = active ?: return
        a.playing = true; a.lastPos = positionMs; a.lastWall = time.nowMs()
    }

    fun onSeek(fromMs: Long, toMs: Long) {
        val a = active ?: return
        if (a.playing) credit(a, fromMs)
        emit(draft(a, EventType.SEEK, fromMs, seekTo = toMs))
        a.lastPos = toMs; a.lastWall = time.nowMs()
    }

    fun onSettingsChanged(shuffle: Boolean, repeat: RepeatMode, speed: Float, device: OutputDevice? = null) {
        val a = active ?: return
        a.shuffle = shuffle; a.repeat = repeat; a.speed = speed
    }

    fun onTrackEnded(positionMs: Long, reason: EndReason) {
        val a = active ?: return
        finish(a, positionMs, reason)
        active = null
    }

    private fun credit(a: Active, pos: Long) {
        val now = time.nowMs()
        val media = pos - a.lastPos
        val wall = now - a.lastWall
        if (media > 0) {
            val allowed = (wall * a.speed).toLong() + SLACK_MS
            a.listenedMs += minOf(media, allowed)
        }
        a.lastPos = pos; a.lastWall = now
    }

    private fun finish(a: Active, pos: Long, reason: EndReason) {
        if (a.playing) credit(a, pos)
        emit(draft(a, EventType.END, pos, listened = a.listenedMs, endReason = reason))
    }

    private fun draft(
        a: Active, type: EventType, pos: Long, seekTo: Long? = null, listened: Long? = null,
        startReason: StartReason? = null, endReason: EndReason? = null,
    ) = PlaybackEventDraft(
        ts = time.nowMs(), tzOffsetMin = time.tzOffsetMinutes(), type = type, listenId = a.listenId, trackId = a.trackId,
        durationMs = a.durationMs, positionMs = pos, seekToMs = seekTo, listenedMs = listened,
        startReason = startReason, endReason = endReason, context = a.context, device = a.device,
        shuffle = a.shuffle, repeatMode = a.repeat, speed = a.speed,
    )

    private companion object { const val SLACK_MS = 750L }
}
