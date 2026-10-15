package com.unbound.data.repo

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightTelemetryWriter
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.*
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

@OptIn(ExperimentalCoroutinesApi::class)
class TelemetryWriterTest {

    private fun sampleEvent(
        trackId: Long,
        type: EventType,
        ts: Long,
        listenId: String = "listen-1",
    ) = PlaybackEventDraft(
        ts = ts,
        tzOffsetMin = 0,
        type = type,
        listenId = listenId,
        trackId = trackId,
        durationMs = 200_000L,
        positionMs = 10_000L,
        seekToMs = if (type == EventType.SEEK) 50_000L else null,
        listenedMs = 10_000L,
        startReason = StartReason.USER,
        endReason = if (type == EventType.END) EndReason.COMPLETED else null,
        context = PlayContext.NONE,
        device = OutputDevice(OutputDeviceType.SPEAKER, "Desktop Speaker"),
        shuffle = false,
        repeatMode = RepeatMode.OFF,
        speed = 1.0f,
    )

    @Test
    fun testTelemetryWriterFlushAndOrder() = runTest {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val libWriter = SqlDelightLibraryWriter(db, driver)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        val f = FoundFile("file:///music/t1.mp3", 1000L, 1000L)
        val p = testTrack("Track 1", durationMs = 200_000L)
        libWriter.applyBatch(sourceId, ScanBatch(listOf(ParsedFile(f, p)), emptyList()))
        val trackId = db.libraryQueries.selectTrackIdByUri("file:///music/t1.mp3").executeAsOne()

        val testJob = kotlinx.coroutines.Job()
        val writerScope = kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Default + testJob)
        val writer = SqlDelightTelemetryWriter(db, writerScope, kotlinx.coroutines.Dispatchers.Default)

        try {
            // 1. Submit non-END events (START and SEEK) and verify flush() drains
            val e1 = sampleEvent(trackId, EventType.START, ts = 1000L)
            val e2 = sampleEvent(trackId, EventType.SEEK, ts = 2000L)

            writer.submit(e1)
            writer.submit(e2)

            writer.flush()

            val eventsInDb = db.eventsQueries.exportEvents().executeAsList()
            assertEquals(2, eventsInDb.size)
            // Verify order preserved
            assertEquals(1000L, eventsInDb[0].ts)
            assertEquals(EventType.START.name, eventsInDb[0].event_type)
            assertEquals(2000L, eventsInDb[1].ts)
            assertEquals(EventType.SEEK.name, eventsInDb[1].event_type)

            // 2. Submit END event - flushes promptly
            val e3 = sampleEvent(trackId, EventType.END, ts = 3000L)
            writer.submit(e3)
            writer.flush()

            val allEvents = db.eventsQueries.exportEvents().executeAsList()
            assertEquals(3, allEvents.size)
            assertEquals(3000L, allEvents[2].ts)
            assertEquals(EventType.END.name, allEvents[2].event_type)
        } finally {
            testJob.cancel()
        }
    }

    @Test
    fun testTelemetryStress20000Events() = runTest {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val libWriter = SqlDelightLibraryWriter(db, driver)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        val f = FoundFile("file:///music/t1.mp3", 1000L, 1000L)
        val p = testTrack("Track 1", durationMs = 200_000L)
        libWriter.applyBatch(sourceId, ScanBatch(listOf(ParsedFile(f, p)), emptyList()))
        val trackId = db.libraryQueries.selectTrackIdByUri("file:///music/t1.mp3").executeAsOne()

        val testJob = Job()
        val writerScope = CoroutineScope(Dispatchers.Default + testJob)
        val writer = SqlDelightTelemetryWriter(db, writerScope, Dispatchers.Default)

        try {
            val totalEvents = 20_000
            val numCoroutines = 8
            val eventsPerCoroutine = totalEvents / numCoroutines

            coroutineScope {
                val jobs = (0 until numCoroutines).map { coroutineIdx ->
                    launch(Dispatchers.Default) {
                        for (i in 0 until eventsPerCoroutine) {
                            val eventId = (coroutineIdx * eventsPerCoroutine + i).toLong()
                            writer.submit(sampleEvent(trackId, EventType.START, ts = eventId))
                            if (i % 250 == 0) {
                                writer.flush()
                            }
                        }
                    }
                }
                jobs.forEach { it.join() }
            }

            writer.flush()

            val rowCount = db.eventsQueries.exportEvents().executeAsList().size
            assertEquals(totalEvents, rowCount)
        } finally {
            testJob.cancel()
        }
    }
}
