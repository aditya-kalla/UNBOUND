package com.unbound.data.repo

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.impl.SqlDelightLibraryRepository
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightPlaylistRepository
import com.unbound.data.scan.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlin.test.*

class ScannerTest {

    private class FakeMediaFileSource(var files: List<FoundFile>) : MediaFileSource {
        var shouldThrow = false
        override suspend fun list(source: ScanSourceInfo): List<FoundFile> {
            if (shouldThrow) throw IllegalStateException("Drive unmounted")
            return files
        }
    }

    private class FakeMetadataReader : MetadataReader {
        val tracks = mutableMapOf<String, ParsedTrack?>()
        override suspend fun read(file: FoundFile): ParsedTrack? = tracks[file.uri]
    }

    private fun sampleTrack(
        title: String,
        artist: String = "Test Artist",
        album: String = "Test Album",
        fingerprint: String? = "fp-$title",
        durationMs: Long = 180_000L,
    ) = testTrack(
        title = title,
        artists = listOf(artist),
        albumArtist = artist,
        album = album,
        durationMs = durationMs,
        fingerprint = fingerprint,
    )

    @Test
    fun testScannerScenarios() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val libraryRepo = SqlDelightLibraryRepository(db, driver)
        val playlistRepo = SqlDelightPlaylistRepository(db)

        val sourceId = libraryRepo.addSource("file:///music", ScanSourceKind.FOLDER, "Music")
        val srcInfo = ScanSourceInfo(sourceId, "file:///music", ScanSourceKind.FOLDER, "Music")
        val file1 = FoundFile("file:///music/song1.mp3", size = 1000L, mtime = 1000L)
        val file2 = FoundFile("file:///music/song2.mp3", size = 2000L, mtime = 1000L)

        val fakeSource = FakeMediaFileSource(listOf(file1, file2))
        val fakeReader = FakeMetadataReader().apply {
            tracks[file1.uri] = sampleTrack("Song One", fingerprint = "fp-song1")
            tracks[file2.uri] = sampleTrack("Song Two", fingerprint = "fp-song2")
        }

        var clock = 1_000_000L
        val scanner = LibraryScanner(
            source = fakeSource,
            reader = fakeReader,
            writer = writer,
            ioDispatcher = Dispatchers.Unconfined,
            nowMs = { clock += 1000; clock },
            concurrency = 2,
            batchSize = 10,
        )

        // 1. New files: scan imports file1 and file2
        val progress1 = scanner.scan(srcInfo).toList()
        val done1 = progress1.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(2, done1.summary.added)
        assertEquals(0, done1.summary.updated)
        assertEquals(0, done1.summary.markedMissing)

        val tracksInitial = libraryRepo.observeTracks(TrackSort.TITLE, includeMissing = true).first()
        assertEquals(2, tracksInitial.size)
        val track1Id = tracksInitial.first { it.title == "Song One" }.id
        val track2Id = tracksInitial.first { it.title == "Song Two" }.id

        // 2. Changed file: file1 mtime changes, metadata updated
        val file1Updated = FoundFile("file:///music/song1.mp3", size = 1050L, mtime = 2000L)
        fakeSource.files = listOf(file1Updated, file2)
        fakeReader.tracks[file1.uri] = sampleTrack("Song One Remaster", fingerprint = "fp-song1")

        val progress2 = scanner.scan(srcInfo).toList()
        val done2 = progress2.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(0, done2.summary.added)
        assertEquals(1, done2.summary.updated)
        assertEquals(1, done2.summary.unchanged)

        val track1AfterUpdate = libraryRepo.getTrack(track1Id)
        assertNotNull(track1AfterUpdate)
        assertEquals("Song One Remaster", track1AfterUpdate.row.title)

        // 3. Moved file keeps like, playlist entry, and listening history (relink)
        // Set like on track1
        libraryRepo.setLiked(track1Id, true)
        assertTrue(libraryRepo.observeIsLiked(track1Id).first())

        // Add track1 to a playlist
        val playlistId = playlistRepo.create("Favorites")
        playlistRepo.addTracks(playlistId, listOf(track1Id))
        val playlistItemsBefore = playlistRepo.observeEntries(playlistId).first()
        assertEquals(1, playlistItemsBefore.size)
        assertEquals(track1Id, playlistItemsBefore[0].track.id)

        // Record a listening event for track1
        db.eventsQueries.insertEvent(
            ts = clock,
            tzOffsetMin = 0L,
            eventType = "START",
            listenId = "session-1",
            trackId = track1Id,
            durationMs = 180_000L,
            positionMs = 0L,
            seekToMs = null,
            listenedMs = null,
            startReason = "USER",
            endReason = null,
            contextType = "PLAYLIST",
            contextId = playlistId.toString(),
            outputDeviceType = "SPEAKER",
            outputDeviceName = "Desktop",
            shuffle = 0L,
            repeatMode = "OFF",
            speed = 1.0,
        )
        val eventsBefore = db.eventsQueries.exportEvents().executeAsList().filter { it.track_id == track1Id }
        assertEquals(1, eventsBefore.size)

        // Now move file1 to song1_moved.mp3 with same fingerprint "fp-song1"
        val file1Moved = FoundFile("file:///music/song1_moved.mp3", size = 1050L, mtime = 2000L)
        fakeSource.files = listOf(file1Moved, file2)
        fakeReader.tracks[file1Moved.uri] = sampleTrack("Song One Remaster", fingerprint = "fp-song1")

        val progress3 = scanner.scan(srcInfo).toList()
        val done3 = progress3.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(1, done3.summary.relinked)
        assertEquals(0, done3.summary.markedMissing)

        // Verify track ID is preserved and URI is updated
        val track1AfterMove = libraryRepo.getTrack(track1Id)
        assertNotNull(track1AfterMove)
        assertEquals("file:///music/song1_moved.mp3", track1AfterMove.row.uri)
        assertFalse(track1AfterMove.row.isMissing)

        // Verify like is preserved
        assertTrue(libraryRepo.observeIsLiked(track1Id).first())

        // Verify playlist item still references track1Id
        val playlistItemsAfter = playlistRepo.observeEntries(playlistId).first()
        assertEquals(1, playlistItemsAfter.size)
        assertEquals(track1Id, playlistItemsAfter[0].track.id)

        // Verify listening history still references track1Id
        val eventsAfter = db.eventsQueries.exportEvents().executeAsList().filter { it.track_id == track1Id }
        assertEquals(1, eventsAfter.size)

        // 4. Vanished file -> marked missing, rows kept
        fakeSource.files = listOf(file1Moved) // file2 vanished
        val progress4 = scanner.scan(srcInfo).toList()
        val done4 = progress4.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(1, done4.summary.markedMissing)

        val track2Missing = libraryRepo.getTrack(track2Id)
        assertNotNull(track2Missing)
        assertTrue(track2Missing.row.isMissing)

        // 5. Reappearing file resurrected
        fakeSource.files = listOf(file1Moved, file2) // file2 reappears
        val progress5 = scanner.scan(srcInfo).toList()
        val done5 = progress5.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(1, done5.summary.updated) // resurrected files are in diff.changed, so updated++
        assertEquals(0, done5.summary.added)

        val track2Resurrected = libraryRepo.getTrack(track2Id)
        assertNotNull(track2Resurrected)
        assertFalse(track2Resurrected.row.isMissing)

        // 6. Empty listing of a populated source -> Failed and nothing marked missing
        fakeSource.files = emptyList()
        val progress6 = scanner.scan(srcInfo).toList()
        val failed = progress6.filterIsInstance<ScanProgress.Failed>().single()
        assertTrue(failed.message.contains("returned no files"))

        // Verify neither track was marked missing
        assertFalse(libraryRepo.getTrack(track1Id)!!.row.isMissing)
        assertFalse(libraryRepo.getTrack(track2Id)!!.row.isMissing)

        // 7. Unreadable file counted
        val unreadableFile = FoundFile("file:///music/corrupt.mp3", size = 500L, mtime = 1000L)
        fakeSource.files = listOf(file1Moved, file2, unreadableFile)
        fakeReader.tracks[unreadableFile.uri] = null // corrupt / unreadable

        val progress7 = scanner.scan(srcInfo).toList()
        val done7 = progress7.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(1, done7.summary.unreadable)

        // 8. 1 000-file batch test
        val batchFiles = (1..1000).map { i ->
            val uri = "file:///music/batch_$i.mp3"
            fakeReader.tracks[uri] = sampleTrack("Batch Song $i", artist = "Batch Artist", album = "Batch Album")
            FoundFile(uri, size = 1000L + i, mtime = 5000L)
        }
        fakeSource.files = batchFiles
        val progress8 = scanner.scan(srcInfo).toList()
        val done8 = progress8.filterIsInstance<ScanProgress.Done>().single()
        assertEquals(1000, done8.summary.added)

        val counts = libraryRepo.observeCounts().first()
        assertEquals(1000, counts.tracks)
        assertEquals(2, counts.missing) // track1 and track2 are now missing since batch didn't include them
    }

    @Test
    fun testAddSourceIdempotentAndResurrectsMissingTracks() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val libraryRepo = SqlDelightLibraryRepository(db, driver)

        val uri = "file:///music_resurrect"
        val id1 = libraryRepo.addSource(uri, ScanSourceKind.FOLDER, "Music Folder")
        assertTrue(id1 > 0)

        // Insert a track under this source
        val file = FoundFile("$uri/song.mp3", 1000L, 1000L)
        val parsed = sampleTrack("Song to Resurrect")
        writer.applyBatch(id1, ScanBatch(listOf(ParsedFile(file, parsed)), emptyList()))

        val track = libraryRepo.observeTracks(TrackSort.TITLE, includeMissing = true).first().single()
        assertFalse(track.isMissing)

        // Mark tracks missing for this source
        db.libraryQueries.markTracksMissingForSource(id1)
        val trackMissing = libraryRepo.getTrack(track.id)!!
        assertTrue(trackMissing.row.isMissing)

        // Re-add the same source folder
        val id2 = libraryRepo.addSource(uri, ScanSourceKind.FOLDER, "Music Folder")
        assertEquals(id1, id2, "Adding the same source folder twice must return the same id")

        // Tracks must be resurrected
        val trackResurrected = libraryRepo.getTrack(track.id)!!
        assertFalse(trackResurrected.row.isMissing, "Re-adding source folder must resurrect missing tracks")
    }
}
