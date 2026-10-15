package com.unbound.data.repo

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.FoundFile
import com.unbound.data.model.ParsedTrack
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightPlaylistRepository
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlin.test.*

class PlaylistTest {

    @Test
    fun testPlaylistMoveRenumberPinLimitAndFolderDelete() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val playlistRepo = SqlDelightPlaylistRepository(db)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        // Insert 3 tracks
        val trackIds = (1..3).map { i ->
            val f = FoundFile("file:///music/p$i.mp3", 1000L, 1000L)
            val p = testTrack(title = "PTrack $i", artists = listOf("PArtist"), durationMs = 150_000L)
            writer.applyBatch(sourceId, ScanBatch(listOf(ParsedFile(f, p)), emptyList()))
            db.libraryQueries.selectTrackIdByUri("file:///music/p$i.mp3").executeAsOne()
        }

        // 1. Create playlist and add tracks
        val playlistId = playlistRepo.create("Roadtrip", "Best driving songs")
        playlistRepo.addTracks(playlistId, trackIds)

        var entries = playlistRepo.observeEntries(playlistId).first()
        assertEquals(3, entries.size)
        assertEquals(trackIds, entries.map { it.track.id })

        // Force a collision by manually setting positions adjacent: 100, 101, 102
        db.playlistQueries.setItemPosition(100L, entries[0].itemId)
        db.playlistQueries.setItemPosition(101L, entries[1].itemId)
        db.playlistQueries.setItemPosition(102L, entries[2].itemId)

        // Moving entries[2] between entries[0] (pos 100) and entries[1] (pos 101):
        // PositionAllocator.between(100, 101) == null. Renumbering must kick in!
        playlistRepo.moveEntry(playlistId, itemId = entries[2].itemId, afterItemId = entries[0].itemId)

        val entriesAfterMove = playlistRepo.observeEntries(playlistId).first()
        assertEquals(3, entriesAfterMove.size)
        // Order must now be: trackIds[0], trackIds[2], trackIds[1]
        assertEquals(listOf(trackIds[0], trackIds[2], trackIds[1]), entriesAfterMove.map { it.track.id })

        // Verify gaps were restored after renumbering
        val rawItems = db.playlistQueries.playlistItems(playlistId).executeAsList()
        assertEquals(1024L, rawItems[0].position)
        assertEquals(2048L, rawItems[1].position)
        assertEquals(3072L, rawItems[2].position)

        // 2. Pin limit: MAX_PINNED = 4
        val p1 = playlistId
        val p2 = playlistRepo.create("P2")
        val p3 = playlistRepo.create("P3")
        val p4 = playlistRepo.create("P4")
        val p5 = playlistRepo.create("P5")

        assertTrue(playlistRepo.setPinned(p1, true))
        assertTrue(playlistRepo.setPinned(p2, true))
        assertTrue(playlistRepo.setPinned(p3, true))
        assertTrue(playlistRepo.setPinned(p4, true))

        // 5th pinned must return false
        assertFalse(playlistRepo.setPinned(p5, true))

        // Unpin p1
        assertTrue(playlistRepo.setPinned(p1, false))

        // Now pinning p5 should succeed
        assertTrue(playlistRepo.setPinned(p5, true))

        // 3. Folder deletion moves playlists to root (folder_id becomes null)
        val folderId = playlistRepo.createFolder("Rock Classics")
        playlistRepo.moveToFolder(p2, folderId)

        val p2SummaryWithFolder = playlistRepo.observePlaylists().first().first { it.id == p2 }
        assertEquals(folderId, p2SummaryWithFolder.folderId)

        // Delete folder
        playlistRepo.deleteFolder(folderId)

        val p2SummaryAfterFolderDelete = playlistRepo.observePlaylists().first().first { it.id == p2 }
        assertNull(p2SummaryAfterFolderDelete.folderId) // moved to root
    }
}
