package com.unbound.data.repo

import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightStatsRepository
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.runBlocking
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StatsParityTest {

    @Test
    fun statsParityScenarioMatchesPythonValidationExactly() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val statsRepo = SqlDelightStatsRepository(db)

        // 1. Create source and insert tracks Alpha and Beta
        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        val trackAlpha = ParsedTrack(
            title = "Alpha",
            artists = listOf("Artist A"),
            albumArtist = "Artist A",
            album = "Album A",
            genres = listOf("Rock"),
            trackNo = 1,
            discNo = 1,
            year = 2026,
            durationMs = 200_000L,
            bitrateKbps = 320,
            sampleRate = 44100,
            channels = 2,
            bitDepth = 16,
            codec = "mp3",
            isLossless = false,
            rgTrackGainDb = null,
            rgTrackPeak = null,
            rgAlbumGainDb = null,
            rgAlbumPeak = null,
            hasEmbeddedLyrics = false,
            artwork = null,
            fingerprint = "fp_alpha",
            metaVersion = 1,
        )

        val trackBeta = ParsedTrack(
            title = "Beta",
            artists = listOf("Artist B"),
            albumArtist = "Artist B",
            album = "Album B",
            genres = listOf("Jazz"),
            trackNo = 2,
            discNo = 1,
            year = 2026,
            durationMs = 200_000L,
            bitrateKbps = 320,
            sampleRate = 44100,
            channels = 2,
            bitDepth = 16,
            codec = "mp3",
            isLossless = false,
            rgTrackGainDb = null,
            rgTrackPeak = null,
            rgAlbumGainDb = null,
            rgAlbumPeak = null,
            hasEmbeddedLyrics = false,
            artwork = null,
            fingerprint = "fp_beta",
            metaVersion = 1,
        )

        writer.applyBatch(
            sourceId,
            ScanBatch(
                upserts = listOf(
                    ParsedFile(FoundFile("file:///music/alpha.mp3", 1000L, 1000L), trackAlpha),
                    ParsedFile(FoundFile("file:///music/beta.mp3", 1000L, 1000L), trackBeta),
                ),
                relinks = emptyList(),
            ),
        )

        val alphaId = db.libraryQueries.selectTrackIdByUri("file:///music/alpha.mp3").executeAsOne()
        val betaId = db.libraryQueries.selectTrackIdByUri("file:///music/beta.mp3").executeAsOne()

        // 2. Insert playback events
        val tzOffset = 330 // +330 min (India, IST)
        val march10_2040 = Instant.parse("2026-03-10T20:40:00Z").toEpochMilli()

        // Helper to insert a full listen (START + END)
        fun insertListen(
            listenId: String,
            trackId: Long,
            startTs: Long,
            endTs: Long,
            listenedMs: Long?,
            endReason: EndReason?,
        ) {
            db.eventsQueries.insertEvent(
                ts = startTs,
                tzOffsetMin = tzOffset.toLong(),
                eventType = "START",
                listenId = listenId,
                trackId = trackId,
                durationMs = 200_000L,
                positionMs = 0L,
                seekToMs = null,
                listenedMs = null,
                startReason = "USER",
                endReason = null,
                contextType = "ALBUM",
                contextId = "1",
                outputDeviceType = "SPEAKER",
                outputDeviceName = "Built-in",
                shuffle = 0L,
                repeatMode = "OFF",
                speed = 1.0,
            )
            if (endReason != null && listenedMs != null) {
                db.eventsQueries.insertEvent(
                    ts = endTs,
                    tzOffsetMin = tzOffset.toLong(),
                    eventType = "END",
                    listenId = listenId,
                    trackId = trackId,
                    durationMs = 200_000L,
                    positionMs = listenedMs,
                    seekToMs = null,
                    listenedMs = listenedMs,
                    startReason = null,
                    endReason = endReason.name,
                    contextType = "ALBUM",
                    contextId = "1",
                    outputDeviceType = "SPEAKER",
                    outputDeviceName = "Built-in",
                    shuffle = 0L,
                    repeatMode = "OFF",
                    speed = 1.0,
                )
            }
        }

        // Alpha x3, 5 minutes apart, fully listened (200 000 ms, COMPLETED)
        insertListen("l_a1", alphaId, march10_2040, march10_2040 + 200_000L, 200_000L, EndReason.COMPLETED)
        insertListen("l_a2", alphaId, march10_2040 + 300_000L, march10_2040 + 500_000L, 200_000L, EndReason.COMPLETED)
        insertListen("l_a3", alphaId, march10_2040 + 600_000L, march10_2040 + 800_000L, 200_000L, EndReason.COMPLETED)

        // Beta 1: START at 20:40 + 2 000 s (= 2 000 000 ms), END after 10 000 ms (SKIP_NEXT)
        val beta1Start = march10_2040 + 2_000_000L
        insertListen("l_b1", betaId, beta1Start, beta1Start + 10_000L, 10_000L, EndReason.SKIP_NEXT)

        // Beta 2: START at 2026-03-10 08:30 UTC with 180 000 ms listened (COMPLETED)
        val beta2Start = Instant.parse("2026-03-10T08:30:00Z").toEpochMilli()
        insertListen("l_b2", betaId, beta2Start, beta2Start + 180_000L, 180_000L, EndReason.COMPLETED)

        // Beta 3: START at 2026-03-10 09:00 UTC with NO end (orphan)
        val beta3Start = Instant.parse("2026-03-10T09:00:00Z").toEpochMilli()
        insertListen("l_b3", betaId, beta3Start, beta3Start, null, null)

        // Range = all of March 2026
        val range = TimeRange(
            fromTs = Instant.parse("2026-03-01T00:00:00Z").toEpochMilli(),
            toTs = Instant.parse("2026-04-01T00:00:00Z").toEpochMilli(),
        )

        // 3. Assertions
        val totals = statsRepo.totals(range)
        assertEquals(790_000L, totals.listenedMs, "Totals listenedMs should be 790 000")
        assertEquals(4, totals.plays, "Totals plays should be 4")
        assertEquals(1, totals.skips, "Totals skips should be 1")
        assertEquals(2, totals.uniqueTracks, "Totals unique_tracks should be 2")

        val topTracks = statsRepo.topTracks(range, limit = 10)
        assertEquals(2, topTracks.size)
        assertEquals(alphaId, topTracks[0].trackId)
        assertEquals(3, topTracks[0].plays)
        assertEquals(betaId, topTracks[1].trackId)
        assertEquals(1, topTracks[1].plays)

        val hourHist = statsRepo.hourHistogram(range).associate { it.hour to it.plays }
        assertEquals(mapOf(2 to 3, 14 to 1), hourHist, "hourHistogram should have 3 plays at hour 2 and 1 play at hour 14")

        val win1_3 = statsRepo.topTracksInHourWindow(range, startHour = 1, endHour = 3)
        assertEquals(1, win1_3.size)
        assertEquals(alphaId, win1_3[0].trackId)

        val win22_3 = statsRepo.topTracksInHourWindow(range, startHour = 22, endHour = 3)
        assertEquals(1, win22_3.size)
        assertEquals(alphaId, win22_3[0].trackId)

        val win3_5 = statsRepo.topTracksInHourWindow(range, startHour = 3, endHour = 5)
        assertTrue(win3_5.isEmpty(), "Window 3..5 should be empty")

        val looped = statsRepo.loopedTracks(range, minPlays = 3)
        assertEquals(1, looped.size)
        assertEquals(alphaId, looped[0].trackId)
        assertEquals("2026-03-11", looped[0].day)
        assertEquals(3, looped[0].plays)

        val skipped = statsRepo.mostSkipped(range, minStarts = 1)
        assertTrue(skipped.isNotEmpty())
        assertEquals(betaId, skipped[0].trackId)
        assertEquals(1, skipped[0].skips)
    }
}
