package com.unbound.data.repo

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.repo.impl.SqlDelightQueueRepository
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class QueueTest {

    @Test
    fun testQueuePlayNextShuffleAndPersistence() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val queueRepo = SqlDelightQueueRepository(db)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        // Insert tracks T1..T6
        for (i in 1..6) {
            val f = FoundFile("file:///music/t$i.mp3", 1000L, 1000L)
            val p = testTrack(title = "Track $i", artists = listOf("Artist"), durationMs = 120_000L)
            writer.applyBatch(sourceId, ScanBatch(listOf(ParsedFile(f, p)), emptyList()))
        }
        val trackIds = db.libraryQueries.tracksByTitle().executeAsList().map { it.id }
        assertEquals(6, trackIds.size)

        // 1. replaceQueue with shuffle=false
        queueRepo.replaceQueue(
            trackIds = trackIds,
            startItemIndex = 0,
            context = PlayContext(ContextType.ALBUM, "1"),
            shuffle = false,
        )

        val queue1 = queueRepo.observeQueue().first()
        assertEquals(6, queue1.size)
        assertEquals(trackIds, queue1.map { it.track.id })

        val state1 = queueRepo.getState()
        assertEquals(queue1[0].itemId, state1.currentItemId)

        // 2. playNext ordering:
        // Current item is queue1[0] (T1).
        // Call playNext(listOf(trackIds[4])) (T5) -> should be right after T1.
        queueRepo.playNext(listOf(trackIds[4]))
        val queue2 = queueRepo.observeQueue().first()
        assertEquals(7, queue2.size)
        assertEquals(trackIds[0], queue2[0].track.id)
        assertEquals(trackIds[4], queue2[1].track.id) // inserted after current
        assertEquals(QueueOrigin.USER, queue2[1].origin)

        // Call playNext(listOf(trackIds[5])) (T6) -> should insert after earlier USER items (after T5), but before T2!
        queueRepo.playNext(listOf(trackIds[5]))
        val queue3 = queueRepo.observeQueue().first()
        assertEquals(8, queue3.size)
        assertEquals(trackIds[0], queue3[0].track.id) // T1
        assertEquals(trackIds[4], queue3[1].track.id) // T5 (earlier USER)
        assertEquals(trackIds[5], queue3[2].track.id) // T6 (later USER)
        assertEquals(trackIds[1], queue3[3].track.id) // T2
        assertEquals(trackIds[2], queue3[4].track.id) // T3
        assertEquals(trackIds[3], queue3[5].track.id) // T4

        // 3. Shuffle on/off restores original order
        // Reset queue to [T1, T2, T3, T4, T5, T6]
        queueRepo.replaceQueue(
            trackIds = trackIds,
            startItemIndex = 1, // T2 is playing
            context = PlayContext.NONE,
            shuffle = false,
        )
        val initialQueue = queueRepo.observeQueue().first()
        val currentItem = initialQueue[1] // T2
        queueRepo.saveState(state1.copy(currentItemId = currentItem.itemId))

        // Turn shuffle ON
        queueRepo.setShuffle(true)
        val shuffledQueue = queueRepo.observeQueue().first()
        assertEquals(6, shuffledQueue.size)
        // Items up to current (index 0 and 1) must remain untouched
        assertEquals(initialQueue[0].track.id, shuffledQueue[0].track.id)
        assertEquals(initialQueue[1].track.id, shuffledQueue[1].track.id)

        // Turn shuffle OFF -> items after current must restore original_position order
        queueRepo.setShuffle(false)
        val restoredQueue = queueRepo.observeQueue().first()
        assertEquals(initialQueue.map { it.track.id }, restoredQueue.map { it.track.id })

        // 4. Persistence round-trip and speed clamp
        val stateToSave = PlayerStateSnapshot(
            currentItemId = currentItem.itemId,
            positionMs = 45_000L,
            shuffle = true,
            repeatMode = RepeatMode.ALL,
            speed = 3.5f, // should clamp to 2.0
            context = PlayContext(ContextType.PLAYLIST, "42"),
        )
        queueRepo.saveState(stateToSave)
        val loadedState = queueRepo.getState()
        assertEquals(currentItem.itemId, loadedState.currentItemId)
        assertEquals(45_000L, loadedState.positionMs)
        assertTrue(loadedState.shuffle)
        assertEquals(RepeatMode.ALL, loadedState.repeatMode)
        assertEquals(2.0f, loadedState.speed) // clamped to 2.0
        assertEquals(ContextType.PLAYLIST, loadedState.context.type)
        assertEquals("42", loadedState.context.id)

        // Test lower speed clamp
        queueRepo.saveState(loadedState.copy(speed = 0.1f))
        val loadedStateLow = queueRepo.getState()
        assertEquals(0.5f, loadedStateLow.speed) // clamped to 0.5
    }

    @Test
    fun testReplaceQueuePreservesStartIndexAndRepeatMode() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val queueRepo = SqlDelightQueueRepository(db)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        // Insert 5 tracks
        for (i in 1..5) {
            val f = FoundFile("file:///music/t$i.mp3", 1000L, 1000L)
            val p = testTrack(title = "Track $i", artists = listOf("Artist"), durationMs = 120_000L)
            writer.applyBatch(sourceId, ScanBatch(listOf(ParsedFile(f, p)), emptyList()))
        }
        val trackIds = db.libraryQueries.tracksByTitle().executeAsList().map { it.id }
        assertEquals(5, trackIds.size)

        // Set repeat mode to ALL and speed to 1.5
        queueRepo.saveState(
            PlayerStateSnapshot(
                currentItemId = null,
                positionMs = 0L,
                shuffle = false,
                repeatMode = RepeatMode.ALL,
                speed = 1.5f,
                context = PlayContext.NONE,
            )
        )

        // Replace queue with 5 tracks, start index 3 (4th track), shuffle off
        queueRepo.replaceQueue(
            trackIds = trackIds,
            startItemIndex = 3,
            context = PlayContext(ContextType.ALBUM, "1"),
            shuffle = false,
        )

        val queue = queueRepo.observeQueue().first()
        assertEquals(5, queue.size)
        // Queue order unchanged
        assertEquals(trackIds, queue.map { it.track.id })

        val state = queueRepo.getState()
        // Current item is the 4th track (index 3)
        assertEquals(queue[3].itemId, state.currentItemId)
        assertEquals(trackIds[3], queue[3].track.id)
        // Repeat mode and speed survive
        assertEquals(RepeatMode.ALL, state.repeatMode)
        assertEquals(1.5f, state.speed)
    }
}
