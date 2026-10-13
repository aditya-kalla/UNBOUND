package com.unbound.data.repo.impl

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToList
import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.QueueRepository
import com.unbound.data.util.PositionAllocator
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import kotlin.random.Random

class SqlDelightQueueRepository(
    private val db: UnboundDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
) : QueueRepository {

    override fun observeQueue(): Flow<List<QueueEntry>> {
        return db.queueQueries.queueRowsWithAlbum().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                QueueEntry(
                    itemId = row.item_id,
                    track = TrackRow(
                        id = row.track_id,
                        title = row.title,
                        artist = row.artist_display,
                        albumId = row.album_id,
                        albumTitle = row.album_title,
                        durationMs = row.duration_ms,
                        artworkId = row.artwork_id,
                        uri = row.uri,
                        isMissing = row.is_missing == 1L,
                    ),
                    origin = QueueOrigin.valueOf(row.origin),
                )
            }
        }.flowOn(dispatcher)
    }

    override suspend fun getState(): PlayerStateSnapshot = withContext(dispatcher) {
        val row = db.queueQueries.selectPlayerState().executeAsOneOrNull()
        if (row == null) {
            PlayerStateSnapshot(
                currentItemId = null,
                positionMs = 0L,
                shuffle = false,
                repeatMode = RepeatMode.OFF,
                speed = 1.0f,
                context = PlayContext.NONE,
            )
        } else {
            val ctxType = row.context_type?.let { runCatching { ContextType.valueOf(it) }.getOrNull() }
            PlayerStateSnapshot(
                currentItemId = row.current_item_id,
                positionMs = row.position_ms,
                shuffle = row.shuffle == 1L,
                repeatMode = runCatching { RepeatMode.valueOf(row.repeat_mode) }.getOrDefault(RepeatMode.OFF),
                speed = row.speed.toFloat(),
                context = PlayContext(ctxType, row.context_id),
            )
        }
    }

    override suspend fun saveState(state: PlayerStateSnapshot): Unit = withContext(dispatcher) {
        val clampedSpeed = state.speed.coerceIn(0.5f, 2.0f).toDouble()
        db.queueQueries.savePlayerState(
            currentItemId = state.currentItemId,
            positionMs = state.positionMs,
            shuffle = if (state.shuffle) 1L else 0L,
            repeatMode = state.repeatMode.name,
            speed = clampedSpeed,
            contextType = state.context.type?.name,
            contextId = state.context.id,
            now = nowMs(),
        )
    }

    override suspend fun replaceQueue(
        trackIds: List<Long>,
        startItemIndex: Int,
        context: PlayContext,
        shuffle: Boolean,
    ): Unit = withContext(dispatcher) {
        db.transaction {
            db.queueQueries.clearQueue()
            if (trackIds.isEmpty()) {
                val clampedSpeed = 1.0
                db.queueQueries.savePlayerState(
                    currentItemId = null,
                    positionMs = 0L,
                    shuffle = if (shuffle) 1L else 0L,
                    repeatMode = RepeatMode.OFF.name,
                    speed = clampedSpeed,
                    contextType = context.type?.name,
                    contextId = context.id,
                    now = nowMs(),
                )
                return@transaction
            }

            val validStart = startItemIndex.coerceIn(0, trackIds.size - 1)
            val orderedIndices = if (shuffle) {
                val others = (0 until trackIds.size).filter { it != validStart }.toMutableList()
                // Fisher-Yates shuffle over the other items
                for (i in others.size - 1 downTo 1) {
                    val j = Random.nextInt(i + 1)
                    val tmp = others[i]
                    others[i] = others[j]
                    others[j] = tmp
                }
                listOf(validStart) + others
            } else {
                (0 until trackIds.size).toList()
            }

            var firstInsertedId: Long? = null
            for ((posIndex, originalIndex) in orderedIndices.withIndex()) {
                val trackId = trackIds[originalIndex]
                val pos = (posIndex + 1) * PositionAllocator.GAP
                val origPos = (originalIndex + 1) * PositionAllocator.GAP
                db.queueQueries.insertQueueItem(trackId, pos, origPos, QueueOrigin.CONTEXT.name)
                if (posIndex == 0) {
                    firstInsertedId = db.playlistQueries.lastInsertedId().executeAsOne()
                }
            }

            db.queueQueries.savePlayerState(
                currentItemId = firstInsertedId,
                positionMs = 0L,
                shuffle = if (shuffle) 1L else 0L,
                repeatMode = RepeatMode.OFF.name,
                speed = 1.0,
                contextType = context.type?.name,
                contextId = context.id,
                now = nowMs(),
            )
        }
    }

    override suspend fun playNext(trackIds: List<Long>): Unit = withContext(dispatcher) {
        if (trackIds.isEmpty()) return@withContext
        db.transaction {
            val items = db.queueQueries.allQueueItems().executeAsList()
            val state = db.queueQueries.selectPlayerState().executeAsOneOrNull()
            val currentItemId = state?.current_item_id

            val currentIdx = items.indexOfFirst { it.id == currentItemId }
            var targetIdx = if (currentIdx >= 0) currentIdx else -1
            while (targetIdx + 1 < items.size && items[targetIdx + 1].origin == QueueOrigin.USER.name) {
                targetIdx++
            }

            val prevPos = if (targetIdx >= 0) items[targetIdx].position else null
            val nextPos = if (targetIdx + 1 < items.size) items[targetIdx + 1].position else null

            val n = trackIds.size
            val canFitGapped = prevPos != null && nextPos != null && (nextPos - prevPos) > (n + 1) * 2

            if (canFitGapped) {
                val step = (nextPos!! - prevPos) / (n + 1)
                for ((i, trackId) in trackIds.withIndex()) {
                    val pos = prevPos + step * (i + 1)
                    db.queueQueries.insertQueueItem(trackId, pos, pos, QueueOrigin.USER.name)
                }
            } else {
                // Insert items into sequence and renumber
                val head = if (targetIdx >= 0) items.take(targetIdx + 1) else emptyList()
                val tail = if (targetIdx + 1 < items.size) items.drop(targetIdx + 1) else emptyList()

                // Insert the new items
                val newInserted = mutableListOf<Long>()
                for (trackId in trackIds) {
                    db.queueQueries.insertQueueItem(trackId, 0L, 0L, QueueOrigin.USER.name)
                    newInserted.add(db.playlistQueries.lastInsertedId().executeAsOne())
                }

                val allIds = head.map { it.id } + newInserted + tail.map { it.id }
                for ((idx, id) in allIds.withIndex()) {
                    val pos = (idx + 1) * PositionAllocator.GAP
                    db.queueQueries.setQueuePosition(pos, id)
                }
            }
        }
    }

    override suspend fun addToEnd(trackIds: List<Long>, origin: QueueOrigin): Unit = withContext(dispatcher) {
        if (trackIds.isEmpty()) return@withContext
        db.transaction {
            var maxPos = db.queueQueries.maxQueuePosition().executeAsOneOrNull()?.MAX
            for (trackId in trackIds) {
                val nextPos = PositionAllocator.append(maxPos)
                db.queueQueries.insertQueueItem(trackId, nextPos, nextPos, origin.name)
                maxPos = nextPos
            }
        }
    }

    override suspend fun move(itemId: Long, afterItemId: Long?): Unit = withContext(dispatcher) {
        db.transaction {
            val items = db.queueQueries.allQueueItems().executeAsList()
            val others = items.filter { it.id != itemId }

            val prevPos: Long?
            val nextPos: Long?
            if (afterItemId == null) {
                prevPos = null
                nextPos = others.firstOrNull()?.position
            } else {
                val afterIdx = others.indexOfFirst { it.id == afterItemId }
                prevPos = if (afterIdx >= 0) others[afterIdx].position else null
                nextPos = if (afterIdx >= 0 && afterIdx + 1 < others.size) others[afterIdx + 1].position else null
            }

            val newPos = PositionAllocator.between(prevPos, nextPos)
            if (newPos != null) {
                db.queueQueries.setQueuePosition(newPos, itemId)
            } else {
                // Renumber
                val newOrder = if (afterItemId == null) {
                    listOf(itemId) + others.map { it.id }
                } else {
                    val list = others.map { it.id }.toMutableList()
                    val afterIdx = others.indexOfFirst { it.id == afterItemId }
                    if (afterIdx >= 0) list.add(afterIdx + 1, itemId) else list.add(itemId)
                    list
                }
                for ((index, id) in newOrder.withIndex()) {
                    val pos = (index + 1) * PositionAllocator.GAP
                    db.queueQueries.setQueuePosition(pos, id)
                }
            }
        }
    }

    override suspend fun remove(itemId: Long): Unit = withContext(dispatcher) {
        db.queueQueries.deleteQueueItem(itemId)
    }

    override suspend fun clear(): Unit = withContext(dispatcher) {
        db.queueQueries.clearQueue()
    }

    override suspend fun setShuffle(enabled: Boolean): Unit = withContext(dispatcher) {
        db.transaction {
            val items = db.queueQueries.allQueueItems().executeAsList()
            val state = db.queueQueries.selectPlayerState().executeAsOneOrNull()
            val currentItemId = state?.current_item_id

            val currentIdx = items.indexOfFirst { it.id == currentItemId }
            val splitIdx = if (currentIdx >= 0) currentIdx else -1

            val head = items.filterIndexed { index, _ -> index <= splitIdx }
            val tail = items.filterIndexed { index, _ -> index > splitIdx }

            val reorderedTail = if (enabled) {
                tail.shuffled()
            } else {
                tail.sortedBy { it.original_position }
            }

            val fullList = head + reorderedTail
            for ((index, item) in fullList.withIndex()) {
                val pos = (index + 1) * PositionAllocator.GAP
                db.queueQueries.setQueuePosition(pos, item.id)
            }

            if (state != null) {
                db.queueQueries.savePlayerState(
                    currentItemId = state.current_item_id,
                    positionMs = state.position_ms,
                    shuffle = if (enabled) 1L else 0L,
                    repeatMode = state.repeat_mode,
                    speed = state.speed,
                    contextType = state.context_type,
                    contextId = state.context_id,
                    now = nowMs(),
                )
            }
        }
    }
}
