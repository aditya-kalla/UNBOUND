package com.unbound.data.repo.impl

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToList
import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.PlaylistRepository
import com.unbound.data.util.PositionAllocator
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext

class SqlDelightPlaylistRepository(
    private val db: UnboundDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
) : PlaylistRepository {

    companion object {
        const val MAX_PINNED = 4
    }

    override fun observePlaylists(): Flow<List<PlaylistSummary>> {
        return db.playlistQueries.allPlaylists().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                PlaylistSummary(
                    id = row.id,
                    name = row.name,
                    description = row.description,
                    folderId = row.folder_id,
                    isPinned = row.is_pinned == 1L,
                    itemCount = row.item_count.toInt(),
                    updatedAt = row.updated_at,
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeFolders(): Flow<List<PlaylistFolder>> {
        return db.playlistQueries.allFolders().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                PlaylistFolder(
                    id = row.id,
                    name = row.name,
                    parentId = row.parent_id,
                    sortOrder = row.sort_order.toInt(),
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeEntries(playlistId: Long): Flow<List<PlaylistEntry>> {
        return db.playlistQueries.playlistItemsWithAlbum(playlistId).asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                PlaylistEntry(
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
                )
            }
        }.flowOn(dispatcher)
    }

    override suspend fun create(name: String, description: String, folderId: Long?): Long = withContext(dispatcher) {
        db.transactionWithResult {
            val now = nowMs()
            db.playlistQueries.insertPlaylist(name, description, folderId, now)
            db.playlistQueries.lastInsertedId().executeAsOne()
        }
    }

    override suspend fun rename(id: Long, name: String, description: String): Unit = withContext(dispatcher) {
        db.playlistQueries.renamePlaylist(name, description, nowMs(), id)
    }

    override suspend fun delete(id: Long): Unit = withContext(dispatcher) {
        db.playlistQueries.deletePlaylist(id)
    }

    override suspend fun addTracks(playlistId: Long, trackIds: List<Long>): Unit = withContext(dispatcher) {
        db.transaction {
            var currentPos = db.playlistQueries.maxItemPosition(playlistId).executeAsOneOrNull()?.MAX
            val now = nowMs()
            for (trackId in trackIds) {
                val nextPos = PositionAllocator.append(currentPos)
                db.playlistQueries.insertItem(playlistId, trackId, nextPos, now)
                currentPos = nextPos
            }
            db.playlistQueries.updatePlaylistUpdated(now, playlistId)
        }
    }

    override suspend fun removeEntry(itemId: Long): Unit = withContext(dispatcher) {
        db.playlistQueries.deleteItem(itemId)
    }

    override suspend fun moveEntry(playlistId: Long, itemId: Long, afterItemId: Long?): Unit = withContext(dispatcher) {
        db.transaction {
            val orderedItems = db.playlistQueries.playlistItems(playlistId).executeAsList()
            val others = orderedItems.filter { it.item_id != itemId }

            val prevPos: Long?
            val nextPos: Long?
            if (afterItemId == null) {
                prevPos = null
                nextPos = others.firstOrNull()?.position
            } else {
                val afterIdx = others.indexOfFirst { it.item_id == afterItemId }
                prevPos = if (afterIdx >= 0) others[afterIdx].position else null
                nextPos = if (afterIdx >= 0 && afterIdx + 1 < others.size) others[afterIdx + 1].position else null
            }

            val newPos = PositionAllocator.between(prevPos, nextPos)
            val now = nowMs()
            if (newPos != null) {
                db.playlistQueries.setItemPosition(newPos, itemId)
                db.playlistQueries.updatePlaylistUpdated(now, playlistId)
            } else {
                // Gap exhausted: renumber inside one transaction
                val newOrder = if (afterItemId == null) {
                    listOf(itemId) + others.map { it.item_id }
                } else {
                    val list = others.map { it.item_id }.toMutableList()
                    val afterIdx = others.indexOfFirst { it.item_id == afterItemId }
                    if (afterIdx >= 0) list.add(afterIdx + 1, itemId) else list.add(itemId)
                    list
                }
                for ((index, id) in newOrder.withIndex()) {
                    val pos = (index + 1) * PositionAllocator.GAP
                    db.playlistQueries.setItemPosition(pos, id)
                }
                db.playlistQueries.updatePlaylistUpdated(now, playlistId)
            }
        }
    }

    override suspend fun setPinned(id: Long, pinned: Boolean): Boolean = withContext(dispatcher) {
        db.transactionWithResult {
            if (pinned) {
                val currentPinned = db.playlistQueries.pinnedCount().executeAsOne()
                if (currentPinned >= MAX_PINNED) {
                    return@transactionWithResult false
                }
                db.playlistQueries.setPinned(pinned = 1L, pinOrder = currentPinned, id = id)
                true
            } else {
                db.playlistQueries.setPinned(pinned = 0L, pinOrder = 0L, id = id)
                true
            }
        }
    }

    override suspend fun createFolder(name: String, parentId: Long?): Long = withContext(dispatcher) {
        db.transactionWithResult {
            val now = nowMs()
            db.playlistQueries.insertFolder(name, parentId, sortOrder = 0L, now = now)
            db.playlistQueries.lastInsertedId().executeAsOne()
        }
    }

    override suspend fun deleteFolder(id: Long): Unit = withContext(dispatcher) {
        db.playlistQueries.deleteFolder(id)
    }

    override suspend fun moveToFolder(playlistId: Long, folderId: Long?): Unit = withContext(dispatcher) {
        db.playlistQueries.movePlaylistToFolder(folderId, nowMs(), playlistId)
    }
}
