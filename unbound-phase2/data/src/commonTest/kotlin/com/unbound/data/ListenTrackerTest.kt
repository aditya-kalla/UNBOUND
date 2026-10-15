package com.unbound.data

import com.unbound.data.model.*
import com.unbound.data.telemetry.ListenTracker
import com.unbound.data.telemetry.TimeSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

private class FakeTime(var now: Long = 1_000_000L, var tz: Int = 330) : TimeSource {
    override fun nowMs() = now
    override fun tzOffsetMinutes() = tz
}

class ListenTrackerTest {
    private val time = FakeTime()
    private val events = ArrayList<PlaybackEventDraft>()
    private var ids = 0
    private val tracker = ListenTracker(time, { "L${ids++}" }, { events.add(it) })

    private fun start(track: Long = 1, duration: Long = 200_000, pos: Long = 0) =
        tracker.onTrackStarted(track, duration, StartReason.USER, PlayContext(ContextType.ALBUM, "9"), OutputDevice(OutputDeviceType.BLUETOOTH), pos)

    /** Plays [ms] of media time in 1 s ticks starting at [from]. */
    private fun playFor(from: Long, ms: Long) {
        var p = from
        while (p < from + ms) { time.now += 1000; p += 1000; tracker.onPosition(p) }
    }

    @Test fun fullListenCountsAllMediaTime() {
        start(); playFor(0, 200_000); tracker.onTrackEnded(200_000, EndReason.COMPLETED)
        assertEquals(listOf(EventType.START, EventType.END), events.map { it.type })
        assertEquals(200_000L, events.last().listenedMs)
        assertEquals(EndReason.COMPLETED, events.last().endReason)
        assertEquals(events[0].listenId, events[1].listenId)
        assertEquals(330, events[0].tzOffsetMin)
        assertEquals(ContextType.ALBUM, events[0].context.type)
    }

    @Test fun earlySkipRecordsShortListen() {
        start(); playFor(0, 10_000); tracker.onTrackEnded(10_000, EndReason.SKIP_NEXT)
        assertEquals(10_000L, events.last().listenedMs)
    }

    @Test fun seekingDoesNotCountSkippedMedia() {
        start(); playFor(0, 10_000)
        tracker.onSeek(10_000, 150_000)
        playFor(150_000, 10_000)
        tracker.onTrackEnded(160_000, EndReason.USER_STOP)
        assertEquals(20_000L, events.last().listenedMs)
        val seek = events.first { it.type == EventType.SEEK }
        assertEquals(10_000L, seek.positionMs); assertEquals(150_000L, seek.seekToMs)
    }

    @Test fun pausedTimeDoesNotCount() {
        start(); playFor(0, 5_000)
        tracker.onPause(5_000)
        time.now += 60_000
        tracker.onResume(5_000)
        playFor(5_000, 3_000)
        tracker.onTrackEnded(8_000, EndReason.USER_STOP)
        assertEquals(8_000L, events.last().listenedMs)
    }

    @Test fun startingAnotherTrackClosesThePreviousOne() {
        start(track = 1); playFor(0, 3_000)
        start(track = 2)
        assertEquals(listOf(EventType.START, EventType.END, EventType.START), events.map { it.type })
        assertEquals(EndReason.REPLACED, events[1].endReason)
        assertEquals(3_000L, events[1].listenedMs)
        assertEquals(1L, events[1].trackId); assertEquals(2L, events[2].trackId)
    }

    @Test fun unreportedPositionJumpIsCappedByWallClock() {
        start(); playFor(0, 1_000)
        time.now += 1000; tracker.onPosition(100_000)      // media jumped 99 s in 1 s of wall time
        tracker.onTrackEnded(100_000, EndReason.USER_STOP)
        assertTrue(events.last().listenedMs!! <= 1_000L + 1_000L + 750L, "listened=${events.last().listenedMs}")
    }

    @Test fun endWithoutActiveListenIsIgnored() {
        tracker.onTrackEnded(0, EndReason.USER_STOP)
        assertTrue(events.isEmpty()); assertNull(tracker.currentListenId)
    }
}
