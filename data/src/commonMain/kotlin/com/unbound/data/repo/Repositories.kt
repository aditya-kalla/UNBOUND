package com.unbound.data.repo

import com.unbound.data.model.*
import kotlinx.coroutines.flow.Flow

/** Contracts the UI/player layers depend on. Implementations (SQLDelight-backed) live in the data module. */
interface LibraryRepository {
    fun observeTracks(sort: TrackSort = TrackSort.TITLE, includeMissing: Boolean = false): Flow<List<TrackRow>>
    fun observeAlbums(): Flow<List<AlbumRow>>
    fun observeArtists(): Flow<List<ArtistRow>>
    fun observeGenres(): Flow<List<GenreRow>>
    fun observeAlbumTracks(albumId: Long): Flow<List<TrackRow>>       // disc, track order
    fun observeArtistTracks(artistId: Long): Flow<List<TrackRow>>
    fun observeCounts(): Flow<LibraryCounts>
    suspend fun getTrack(id: Long): TrackDetail?

    /** FTS5 prefix search (diacritic-insensitive); falls back to LIKE if FTS5 is unavailable. */
    suspend fun search(query: String, limit: Int = 50): List<TrackRow>
    suspend fun addSearchHistory(query: String)
    fun observeRecentSearches(limit: Int = 10): Flow<List<String>>
    suspend fun clearSearchHistory()

    fun observeLiked(): Flow<List<TrackRow>>
    fun observeIsLiked(trackId: Long): Flow<Boolean>
    suspend fun setLiked(trackId: Long, liked: Boolean)

    fun observeRecentlyPlayed(limit: Int = 50): Flow<List<TrackRow>>

    fun observeSources(): Flow<List<ScanSourceInfo>>
    suspend fun addSource(uri: String, kind: ScanSourceKind, displayName: String): Long
    /** Removes the source; its tracks are flagged missing (not deleted) so playlists/likes/history survive a re-add. */
    suspend fun removeSource(id: Long)
}

interface PlaylistRepository {
    fun observePlaylists(): Flow<List<PlaylistSummary>>                // pinned first
    fun observeFolders(): Flow<List<PlaylistFolder>>
    fun observeEntries(playlistId: Long): Flow<List<PlaylistEntry>>
    suspend fun create(name: String, description: String = "", folderId: Long? = null): Long
    suspend fun rename(id: Long, name: String, description: String)
    suspend fun delete(id: Long)
    suspend fun addTracks(playlistId: Long, trackIds: List<Long>)      // appended in order; duplicates allowed
    suspend fun removeEntry(itemId: Long)
    /** Moves [itemId] so it sits after [afterItemId] (null = to the top). One row update; renumbers only if a gap is exhausted. */
    suspend fun moveEntry(playlistId: Long, itemId: Long, afterItemId: Long?)
    suspend fun setPinned(id: Long, pinned: Boolean): Boolean          // false if the pin limit is reached
    suspend fun createFolder(name: String, parentId: Long? = null): Long
    suspend fun deleteFolder(id: Long)
    suspend fun moveToFolder(playlistId: Long, folderId: Long?)
}

interface QueueRepository {
    fun observeQueue(): Flow<List<QueueEntry>>
    suspend fun getState(): PlayerStateSnapshot
    suspend fun saveState(state: PlayerStateSnapshot)                  // call on change + every ~5 s while playing
    /** Replaces the whole queue (play an album/playlist). [startItemIndex] becomes the current item. */
    suspend fun replaceQueue(trackIds: List<Long>, startItemIndex: Int, context: PlayContext, shuffle: Boolean)
    suspend fun playNext(trackIds: List<Long>)                         // origin USER, right after the current item (after earlier USER items)
    suspend fun addToEnd(trackIds: List<Long>, origin: QueueOrigin = QueueOrigin.USER)
    suspend fun move(itemId: Long, afterItemId: Long?)
    suspend fun remove(itemId: Long)
    suspend fun clear()
    suspend fun setShuffle(enabled: Boolean)                           // reshuffle upcoming items / restore original_position order
}

interface StatsRepository {
    suspend fun totals(range: TimeRange): ListeningTotals
    suspend fun topTracks(range: TimeRange, limit: Int = 20): List<TopTrack>
    suspend fun topArtists(range: TimeRange, limit: Int = 20): List<TopArtist>
    suspend fun topGenres(range: TimeRange, limit: Int = 10): List<TopGenre>
    suspend fun hourHistogram(range: TimeRange): List<HourBucket>      // local hour 0..23, missing hours omitted
    suspend fun topTracksInHourWindow(range: TimeRange, startHour: Int, endHour: Int, limit: Int = 10): List<TopTrack>
    suspend fun weekdayHistogram(range: TimeRange): List<WeekdayBucket>
    suspend fun dailyMinutes(range: TimeRange): List<DayMinutes>
    suspend fun mostSkipped(range: TimeRange, minStarts: Int = 3, limit: Int = 10): List<SkippedTrack>
    suspend fun loopedTracks(range: TimeRange, minPlays: Int = 3, limit: Int = 10): List<LoopedTrack>
    suspend fun deviceBreakdown(range: TimeRange): List<DeviceBucket>
}

/** Non-blocking sink for ListenTracker output. Must never block the audio/UI path. */
interface TelemetryWriter {
    fun submit(event: PlaybackEventDraft)       // enqueue; batch-insert off-thread
    suspend fun flush()                         // END events should be flushed promptly; also flush on app background
}
