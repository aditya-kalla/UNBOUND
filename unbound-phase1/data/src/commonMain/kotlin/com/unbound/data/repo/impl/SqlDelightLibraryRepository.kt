package com.unbound.data.repo.impl

import app.cash.sqldelight.coroutines.asFlow
import app.cash.sqldelight.coroutines.mapToList
import app.cash.sqldelight.coroutines.mapToOne
import app.cash.sqldelight.db.SqlDriver
import com.unbound.data.db.FtsIndexHelper
import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.LibraryRepository
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext

class SqlDelightLibraryRepository(
    private val db: UnboundDatabase,
    private val driver: SqlDriver,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
) : LibraryRepository {

    override fun observeTracks(sort: TrackSort, includeMissing: Boolean): Flow<List<TrackRow>> {
        val inc = if (includeMissing) 1L else 0L
        val mapper = { id: Long, title: String, artist_display: String, album_id: Long?, album_title: String?, duration_ms: Long, artwork_id: Long?, uri: String, is_missing: Long ->
            TrackRow(
                id = id,
                title = title,
                artist = artist_display,
                albumId = album_id,
                albumTitle = album_title,
                durationMs = duration_ms,
                artworkId = artwork_id,
                uri = uri,
                isMissing = is_missing == 1L,
            )
        }
        val query = when (sort) {
            TrackSort.TITLE -> db.libraryQueries.tracksWithAlbumByTitle(inc, mapper)
            TrackSort.ARTIST -> db.libraryQueries.tracksWithAlbumByArtist(inc, mapper)
            TrackSort.ALBUM -> db.libraryQueries.tracksWithAlbumByAlbum(inc, mapper)
            TrackSort.DATE_ADDED -> db.libraryQueries.tracksWithAlbumByDateAdded(inc, mapper)
            TrackSort.DURATION -> db.libraryQueries.tracksWithAlbumByDuration(inc, mapper)
        }
        return query.asFlow().mapToList(dispatcher).flowOn(dispatcher)
    }

    override fun observeAlbums(): Flow<List<AlbumRow>> {
        return db.libraryQueries.allAlbums().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                AlbumRow(
                    id = row.id,
                    title = row.title,
                    albumArtist = row.album_artist,
                    year = row.year?.toInt(),
                    artworkId = row.artwork_id,
                    trackCount = row.track_count.toInt(),
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeArtists(): Flow<List<ArtistRow>> {
        return db.libraryQueries.allArtists().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                ArtistRow(
                    id = row.id,
                    name = row.name,
                    trackCount = row.track_count.toInt(),
                    albumCount = row.album_count.toInt(),
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeGenres(): Flow<List<GenreRow>> {
        return db.libraryQueries.allGenres().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                GenreRow(
                    genre = row.genre,
                    trackCount = row.track_count.toInt(),
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeAlbumTracks(albumId: Long): Flow<List<TrackRow>> {
        return db.libraryQueries.albumTracks(albumId).asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                TrackRow(
                    id = row.id,
                    title = row.title,
                    artist = row.artist_display,
                    albumId = row.album_id,
                    albumTitle = row.album_title,
                    durationMs = row.duration_ms,
                    artworkId = row.artwork_id,
                    uri = row.uri,
                    isMissing = row.is_missing == 1L,
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeArtistTracks(artistId: Long): Flow<List<TrackRow>> {
        return db.libraryQueries.artistTracks(artistId).asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                TrackRow(
                    id = row.id,
                    title = row.title,
                    artist = row.artist_display,
                    albumId = row.album_id,
                    albumTitle = row.album_title,
                    durationMs = row.duration_ms,
                    artworkId = row.artwork_id,
                    uri = row.uri,
                    isMissing = row.is_missing == 1L,
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeCounts(): Flow<LibraryCounts> {
        return db.libraryQueries.libraryCounts().asFlow().mapToOne(dispatcher).map { row ->
            LibraryCounts(
                tracks = row.tracks.toInt(),
                albums = row.albums.toInt(),
                artists = row.artists.toInt(),
                missing = row.missing.toInt(),
            )
        }.flowOn(dispatcher)
    }

    override suspend fun getTrack(id: Long): TrackDetail? = withContext(dispatcher) {
        val row = db.libraryQueries.trackDetailById(id).executeAsOneOrNull() ?: return@withContext null
        val genres = db.libraryQueries.trackGenresById(id).executeAsList()
        val trackRow = TrackRow(
            id = row.id,
            title = row.title,
            artist = row.artist_display,
            albumId = row.album_id,
            albumTitle = row.album_title,
            durationMs = row.duration_ms,
            artworkId = row.artwork_id,
            uri = row.uri,
            isMissing = row.is_missing == 1L,
        )
        TrackDetail(
            row = trackRow,
            genres = genres,
            year = row.year?.toInt(),
            trackNo = row.track_no?.toInt(),
            discNo = row.disc_no?.toInt(),
            codec = row.codec,
            bitrateKbps = row.bitrate_kbps?.toInt(),
            sampleRate = row.sample_rate?.toInt(),
            bitDepth = row.bit_depth?.toInt(),
            channels = row.channels?.toInt(),
            isLossless = row.is_lossless == 1L,
            rgTrackGainDb = row.rg_track_gain_db,
            rgTrackPeak = row.rg_track_peak,
            rgAlbumGainDb = row.rg_album_gain_db,
            rgAlbumPeak = row.rg_album_peak,
            loudnessLufs = row.loudness_lufs,
            hasEmbeddedLyrics = row.has_embedded_lyrics == 1L,
            lrcUri = row.lrc_uri,
        )
    }

    override suspend fun search(query: String, limit: Int): List<TrackRow> = withContext(dispatcher) {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) return@withContext emptyList()

        // 1. Attempt FTS5 search with sanitized query
        val tokens = trimmed.split(Regex("\\s+")).filter { it.isNotEmpty() }
        val ftsQuery = tokens.mapIndexed { idx, token ->
            val clean = token.replace("\"", "\"\"")
            if (idx == tokens.lastIndex) "\"$clean\"*" else "\"$clean\""
        }.joinToString(" ")

        val ftsIds = FtsIndexHelper.searchFts(driver, ftsQuery, limit.toLong())
        if (ftsIds.isNotEmpty()) {
            val rows = db.libraryQueries.tracksByIds(ftsIds).executeAsList().map { row ->
                TrackRow(
                    id = row.id,
                    title = row.title,
                    artist = row.artist_display,
                    albumId = row.album_id,
                    albumTitle = row.album_title,
                    durationMs = row.duration_ms,
                    artworkId = row.artwork_id,
                    uri = row.uri,
                    isMissing = row.is_missing == 1L,
                )
            }.associateBy { it.id }
            return@withContext ftsIds.mapNotNull { rows[it] }.filter { !it.isMissing }
        }

        // 2. Fallback to LIKE query
        db.libraryQueries.searchLikeWithAlbum(trimmed, limit.toLong()).executeAsList().map { row ->
            TrackRow(
                id = row.id,
                title = row.title,
                artist = row.artist_display,
                albumId = row.album_id,
                albumTitle = row.album_title,
                durationMs = row.duration_ms,
                artworkId = row.artwork_id,
                uri = row.uri,
                isMissing = row.is_missing == 1L,
            )
        }
    }

    override suspend fun addSearchHistory(query: String): Unit = withContext(dispatcher) {
        if (query.isNotBlank()) {
            db.playlistQueries.addSearchHistory(query.trim(), nowMs())
        }
    }

    override fun observeRecentSearches(limit: Int): Flow<List<String>> {
        return db.playlistQueries.recentSearches(limit.toLong()).asFlow().mapToList(dispatcher).map { list ->
            list.map { it.query }
        }.flowOn(dispatcher)
    }

    override suspend fun clearSearchHistory(): Unit = withContext(dispatcher) {
        db.playlistQueries.clearSearchHistory()
    }

    override fun observeLiked(): Flow<List<TrackRow>> {
        return db.playlistQueries.likedTracksWithAlbum().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                TrackRow(
                    id = row.id,
                    title = row.title,
                    artist = row.artist_display,
                    albumId = row.album_id,
                    albumTitle = row.album_title,
                    durationMs = row.duration_ms,
                    artworkId = row.artwork_id,
                    uri = row.uri,
                    isMissing = row.is_missing == 1L,
                )
            }
        }.flowOn(dispatcher)
    }

    override fun observeIsLiked(trackId: Long): Flow<Boolean> {
        return db.playlistQueries.isLiked(trackId).asFlow().mapToOne(dispatcher).map { it }.flowOn(dispatcher)
    }

    override suspend fun setLiked(trackId: Long, liked: Boolean): Unit = withContext(dispatcher) {
        if (liked) {
            db.playlistQueries.like(trackId, nowMs())
        } else {
            db.playlistQueries.unlike(trackId)
        }
    }

    override fun observeRecentlyPlayed(limit: Int): Flow<List<TrackRow>> {
        return db.statsQueries.recentlyPlayed(limit.toLong()).asFlow().mapToList(dispatcher).map { list ->
            if (list.isEmpty()) return@map emptyList<TrackRow>()
            val ids = list.mapNotNull { it.track_id }
            val rows = db.libraryQueries.tracksByIds(ids).executeAsList().map { row ->
                TrackRow(
                    id = row.id,
                    title = row.title,
                    artist = row.artist_display,
                    albumId = row.album_id,
                    albumTitle = row.album_title,
                    durationMs = row.duration_ms,
                    artworkId = row.artwork_id,
                    uri = row.uri,
                    isMissing = row.is_missing == 1L,
                )
            }.associateBy { it.id }
            ids.mapNotNull { rows[it] }.filter { !it.isMissing }
        }.flowOn(dispatcher)
    }

    override fun observeSources(): Flow<List<ScanSourceInfo>> {
        return db.libraryQueries.sourceList().asFlow().mapToList(dispatcher).map { list ->
            list.map { row ->
                ScanSourceInfo(
                    id = row.id,
                    uri = row.uri,
                    kind = ScanSourceKind.valueOf(row.kind),
                    displayName = row.display_name,
                    enabled = row.enabled == 1L,
                    lastScanAt = row.last_scan_at,
                )
            }
        }.flowOn(dispatcher)
    }

    override suspend fun addSource(uri: String, kind: ScanSourceKind, displayName: String): Long = withContext(dispatcher) {
        db.transactionWithResult {
            db.libraryQueries.insertSource(uri, kind.name, displayName)
            db.playlistQueries.lastInsertedId().executeAsOne()
        }
    }

    override suspend fun removeSource(id: Long): Unit = withContext(dispatcher) {
        db.transaction {
            db.libraryQueries.markTracksMissingForSource(id)
            db.libraryQueries.deleteSource(id)
        }
    }
}
