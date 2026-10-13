package com.unbound.data.repo.impl

import app.cash.sqldelight.db.SqlDriver
import com.unbound.data.db.FtsIndexHelper
import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.FoundFile
import com.unbound.data.model.KnownFile
import com.unbound.data.model.ParsedTrack
import com.unbound.data.scan.LibraryWriter
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.Relink
import com.unbound.data.scan.ScanBatch
import com.unbound.data.util.ArtistSplitter
import com.unbound.data.util.matchKey
import com.unbound.data.util.sortKey

class SqlDelightLibraryWriter(
    private val db: UnboundDatabase,
    private val driver: SqlDriver,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
) : LibraryWriter {

    override suspend fun scanIndex(sourceId: Long): List<KnownFile> {
        return db.libraryQueries.selectScanIndex(sourceId).executeAsList().map { row ->
            KnownFile(
                id = row.id,
                uri = row.uri,
                size = row.file_size,
                mtime = row.file_mtime,
                fingerprint = row.fingerprint,
                isMissing = row.is_missing == 1L,
                metaVersion = row.meta_version.toInt(),
            )
        }
    }

    override suspend fun applyBatch(sourceId: Long, batch: ScanBatch) {
        db.transaction {
            val now = nowMs()
            for (item in batch.upserts) {
                writeTrack(sourceId, item.file, item.parsed, now, existingTrackId = null)
            }
            for (relink in batch.relinks) {
                writeTrack(sourceId, relink.file, relink.parsed, now, existingTrackId = relink.trackId)
            }
        }
    }

    private fun writeTrack(
        sourceId: Long,
        file: FoundFile,
        parsed: ParsedTrack,
        now: Long,
        existingTrackId: Long?,
    ) {
        // 1. Artwork
        val artworkId = if (parsed.artwork != null) {
            db.libraryQueries.insertArtworkIgnore(
                hash = parsed.artwork.hash,
                path = parsed.artwork.path,
                width = parsed.artwork.width.toLong(),
                height = parsed.artwork.height.toLong(),
            )
            db.libraryQueries.selectArtworkIdByHash(parsed.artwork.hash).executeAsOneOrNull()
        } else null

        // 2. Artists
        val (mains, featured) = when {
            parsed.artists.isEmpty() -> listOf("Unknown Artist") to emptyList()
            parsed.artists.size == 1 -> {
                val (m, f) = ArtistSplitter.split(parsed.artists[0])
                (if (m.isEmpty()) listOf("Unknown Artist") else m) to f
            }
            else -> parsed.artists to emptyList()
        }

        val artistDisplay = if (parsed.artists.isNotEmpty()) {
            parsed.artists.joinToString(", ")
        } else {
            "Unknown Artist"
        }

        val artistEntries = mutableListOf<Triple<Long, String, Int>>() // artistId, role, position
        var pos = 0
        for (artistName in mains) {
            val nameSort = sortKey(artistName)
            val nameKey = matchKey(artistName)
            db.libraryQueries.insertArtistIgnore(artistName, nameSort, nameKey)
            val artistId = db.libraryQueries.selectArtistIdByKey(nameKey).executeAsOne()
            artistEntries.add(Triple(artistId, "MAIN", pos++))
        }
        for (artistName in featured) {
            val nameSort = sortKey(artistName)
            val nameKey = matchKey(artistName)
            db.libraryQueries.insertArtistIgnore(artistName, nameSort, nameKey)
            val artistId = db.libraryQueries.selectArtistIdByKey(nameKey).executeAsOne()
            artistEntries.add(Triple(artistId, "FEATURED", pos++))
        }

        // 3. Album
        val rawAlbum = parsed.album?.takeIf { it.isNotBlank() } ?: "Unknown Album"
        val firstArtist = mains.firstOrNull() ?: "Unknown Artist"
        val rawAlbumArtist = parsed.albumArtist?.takeIf { it.isNotBlank() } ?: firstArtist
        val albumKey = "${matchKey(rawAlbum)}|${matchKey(rawAlbumArtist)}"
        val albumArtistKey = matchKey(rawAlbumArtist)
        db.libraryQueries.insertArtistIgnore(rawAlbumArtist, sortKey(rawAlbumArtist), albumArtistKey)
        val albumArtistId = db.libraryQueries.selectArtistIdByKey(albumArtistKey).executeAsOneOrNull()
        db.libraryQueries.insertAlbumIgnore(
            title = rawAlbum,
            titleSort = sortKey(rawAlbum),
            albumArtistId = albumArtistId,
            albumArtistDisplay = rawAlbumArtist,
            year = parsed.year?.toLong(),
            artworkId = artworkId,
            albumKey = albumKey,
        )
        val albumId = db.libraryQueries.selectAlbumIdByKey(albumKey).executeAsOneOrNull()

        // 4. Title fallback
        val fallbackTitle = file.uri
            .substringAfterLast('/')
            .substringAfterLast('\\')
            .substringBeforeLast('.')
            .takeIf { it.isNotBlank() } ?: "Unknown Track"
        val title = parsed.title?.takeIf { it.isNotBlank() } ?: fallbackTitle
        val titleSort = sortKey(title)

        // 5. Genres: split on ';' only, never '/'
        val allGenres = parsed.genres
            .flatMap { it.split(';') }
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .distinct()

        // 6. Upsert or Relink Track
        val trackId: Long
        if (existingTrackId != null) {
            trackId = existingTrackId
            db.libraryQueries.relinkTrackMetadata(
                sourceId = sourceId,
                uri = file.uri,
                fileSize = file.size,
                fileMtime = file.mtime,
                fingerprint = parsed.fingerprint,
                title = title,
                titleSort = titleSort,
                artistDisplay = artistDisplay,
                albumId = albumId,
                trackNo = parsed.trackNo?.toLong(),
                discNo = parsed.discNo?.toLong(),
                year = parsed.year?.toLong(),
                durationMs = parsed.durationMs,
                bitrateKbps = parsed.bitrateKbps?.toLong(),
                sampleRate = parsed.sampleRate?.toLong(),
                channels = parsed.channels?.toLong(),
                bitDepth = parsed.bitDepth?.toLong(),
                codec = parsed.codec,
                isLossless = if (parsed.isLossless) 1L else 0L,
                rgTrackGain = parsed.rgTrackGainDb,
                rgTrackPeak = parsed.rgTrackPeak,
                rgAlbumGain = parsed.rgAlbumGainDb,
                rgAlbumPeak = parsed.rgAlbumPeak,
                hasLyrics = if (parsed.hasEmbeddedLyrics) 1L else 0L,
                lrcUri = null,
                artworkId = artworkId,
                now = now,
                metaVersion = parsed.metaVersion.toLong(),
                id = trackId,
            )
        } else {
            db.libraryQueries.upsertTrack(
                sourceId = sourceId,
                uri = file.uri,
                fileSize = file.size,
                fileMtime = file.mtime,
                fingerprint = parsed.fingerprint,
                title = title,
                titleSort = titleSort,
                artistDisplay = artistDisplay,
                albumId = albumId,
                trackNo = parsed.trackNo?.toLong(),
                discNo = parsed.discNo?.toLong(),
                year = parsed.year?.toLong(),
                durationMs = parsed.durationMs,
                bitrateKbps = parsed.bitrateKbps?.toLong(),
                sampleRate = parsed.sampleRate?.toLong(),
                channels = parsed.channels?.toLong(),
                bitDepth = parsed.bitDepth?.toLong(),
                codec = parsed.codec,
                isLossless = if (parsed.isLossless) 1L else 0L,
                rgTrackGain = parsed.rgTrackGainDb,
                rgTrackPeak = parsed.rgTrackPeak,
                rgAlbumGain = parsed.rgAlbumGainDb,
                rgAlbumPeak = parsed.rgAlbumPeak,
                hasLyrics = if (parsed.hasEmbeddedLyrics) 1L else 0L,
                lrcUri = null,
                artworkId = artworkId,
                now = now,
                metaVersion = parsed.metaVersion.toLong(),
            )
            trackId = db.libraryQueries.selectTrackIdByUri(file.uri).executeAsOne()
        }

        // 7. Rewrite track_artist and track_genre
        db.libraryQueries.deleteTrackArtists(trackId)
        for ((artistId, role, position) in artistEntries) {
            db.libraryQueries.insertTrackArtist(trackId, artistId, role, position.toLong())
        }

        db.libraryQueries.deleteTrackGenres(trackId)
        for (g in allGenres) {
            db.libraryQueries.insertTrackGenre(trackId, g)
        }

        // 8. Rewrite track_search (FTS5)
        FtsIndexHelper.insertOrUpdate(
            driver = driver,
            trackId = trackId,
            title = title,
            artists = artistDisplay,
            album = rawAlbum,
            genres = allGenres.joinToString(" "),
        )
    }

    override suspend fun markMissing(trackIds: List<Long>) {
        db.transaction {
            for (id in trackIds) {
                db.libraryQueries.markMissing(id)
            }
        }
    }

    override suspend fun finishScan(sourceId: Long, scannedAt: Long) {
        db.transaction {
            db.libraryQueries.pruneEmptyAlbums()
            db.libraryQueries.pruneEmptyArtists()
            db.libraryQueries.pruneUnusedArtwork()
            db.libraryQueries.touchSource(scannedAt, sourceId)
        }
    }
}
