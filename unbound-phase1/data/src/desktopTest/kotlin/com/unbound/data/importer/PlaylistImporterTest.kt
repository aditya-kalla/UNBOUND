package com.unbound.data.importer

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.FoundFile
import com.unbound.data.model.ParsedTrack
import com.unbound.data.repo.impl.SqlDelightLibraryRepository
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightPlaylistRepository
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PlaylistImporterTest {

    @Test
    fun testCsvImportMatchingAndPlaylistCreation() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val libraryRepo = SqlDelightLibraryRepository(db, driver)
        val playlistRepo = SqlDelightPlaylistRepository(db)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        // Populate local library
        val tracks = listOf(
            testTrack(
                title = "Time",
                artists = listOf("Pink Floyd"),
                album = "The Dark Side of the Moon",
                durationMs = 413_000L,
            ),
            testTrack(
                title = "Comfortably Numb",
                artists = listOf("Pink Floyd"),
                album = "The Wall",
                durationMs = 382_000L,
            ),
            testTrack(
                title = "Get Lucky",
                artists = listOf("Daft Punk", "Pharrell Williams"),
                album = "Random Access Memories",
                durationMs = 248_000L,
            ),
        )

        val batch = tracks.mapIndexed { idx, t ->
            val f = FoundFile("file:///music/t$idx.mp3", 1000L, 1000L)
            ParsedFile(f, t)
        }
        writer.applyBatch(sourceId, ScanBatch(batch, emptyList()))

        val importer = PlaylistImporter(libraryRepo, playlistRepo)

        // Fixture CSV in Exportify format with quoted fields, commas, noise suffixes
        val csv = """
            "Track Name","Artist Name(s)","Album Name","Duration (ms)","Added At"
            "Time","Pink Floyd","The Dark Side of the Moon","413500","2023-01-01"
            "Comfortably Numb - Remastered 2011","Pink Floyd","The Wall","381900","2023-01-01"
            "Get Lucky (feat. Pharrell Williams)","Daft Punk, Pharrell Williams","Random Access Memories","248000","2023-01-01"
            "Unknown Song In The Sky","Alien Artist","Mystery Album","180000","2023-01-01"
        """.trimIndent()

        val result = importer.import("Pink Floyd & More", csv)

        // 3 matched, 1 unmatched
        assertEquals(3, result.matched.size)
        assertEquals(1, result.unmatched.size)

        assertEquals("Time", result.matched[0].track.title)
        assertTrue(result.matched[0].score >= 80)

        assertEquals("Comfortably Numb", result.matched[1].track.title)
        assertTrue(result.matched[1].score >= 80)

        assertEquals("Get Lucky", result.matched[2].track.title)
        assertTrue(result.matched[2].score >= 80)

        assertEquals("Unknown Song In The Sky", result.unmatched[0].csv.trackName)

        // Verify playlist items in DB match CSV order
        val playlistEntries = playlistRepo.observeEntries(result.playlistId).first()
        assertEquals(3, playlistEntries.size)
        assertEquals("Time", playlistEntries[0].track.title)
        assertEquals("Comfortably Numb", playlistEntries[1].track.title)
        assertEquals("Get Lucky", playlistEntries[2].track.title)
    }
}
