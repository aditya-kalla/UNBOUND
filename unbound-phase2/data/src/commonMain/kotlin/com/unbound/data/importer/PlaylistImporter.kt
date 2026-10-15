package com.unbound.data.importer

import com.unbound.data.model.TrackRow
import com.unbound.data.repo.LibraryRepository
import com.unbound.data.repo.PlaylistRepository
import com.unbound.data.util.matchKey
import kotlin.math.abs

data class CsvTrackRow(
    val trackName: String,
    val artistName: String,
    val albumName: String,
    val durationMs: Long,
)

data class MatchedTrack(
    val csv: CsvTrackRow,
    val track: TrackRow,
    val score: Int,
)

data class UnmatchedRow(
    val csv: CsvTrackRow,
    val reason: String,
)

data class ImportResult(
    val playlistId: Long,
    val matched: List<MatchedTrack>,
    val unmatched: List<UnmatchedRow>,
)

class PlaylistImporter(
    private val libraryRepo: LibraryRepository,
    private val playlistRepo: PlaylistRepository,
) {
    companion object {
        /** Minimum score required to accept a match (scale 0..100). */
        const val MATCH_THRESHOLD = 50

        private val NOISE_SUFFIX_REGEX = Regex(
            """\s*(?:-\s*|\(\s*|\[\s*)(?:remastered|remaster|live|deluxe|radio edit|bonus track|feat\.?|ft\.?).*""",
            RegexOption.IGNORE_CASE,
        )

        fun cleanTitle(title: String): String {
            return title.replace(NOISE_SUFFIX_REGEX, "").trim()
        }

        private fun splitArtists(artistStr: String): Set<String> {
            return artistStr.split(Regex("[,;/&]|\\s+(?:feat\\.?|ft\\.?|featuring)\\s+", RegexOption.IGNORE_CASE))
                .map { matchKey(it) }
                .filter { it.isNotEmpty() }
                .toSet()
        }

        /** Simple RFC 4180 CSV parser supporting quotes, commas, and escaped quotes. */
        fun parseCsv(csvText: String): List<List<String>> {
            val rows = mutableListOf<List<String>>()
            val currentField = StringBuilder()
            val currentRow = mutableListOf<String>()
            var inQuotes = false
            var i = 0

            while (i < csvText.length) {
                val c = csvText[i]
                when {
                    c == '"' -> {
                        if (inQuotes && i + 1 < csvText.length && csvText[i + 1] == '"') {
                            currentField.append('"')
                            i++ // skip escaped quote
                        } else {
                            inQuotes = !inQuotes
                        }
                    }
                    c == ',' && !inQuotes -> {
                        currentRow.add(currentField.toString())
                        currentField.clear()
                    }
                    (c == '\n' || c == '\r') && !inQuotes -> {
                        if (c == '\r' && i + 1 < csvText.length && csvText[i + 1] == '\n') {
                            i++
                        }
                        currentRow.add(currentField.toString())
                        currentField.clear()
                        if (currentRow.any { it.isNotBlank() }) {
                            rows.add(currentRow.toList())
                        }
                        currentRow.clear()
                    }
                    else -> {
                        currentField.append(c)
                    }
                }
                i++
            }

            if (currentField.isNotEmpty() || currentRow.isNotEmpty()) {
                currentRow.add(currentField.toString())
                if (currentRow.any { it.isNotBlank() }) {
                    rows.add(currentRow.toList())
                }
            }

            return rows
        }
    }

    suspend fun import(playlistName: String, csvContent: String): ImportResult {
        val rows = parseCsv(csvContent)
        if (rows.isEmpty()) {
            val id = playlistRepo.create(playlistName, "Imported playlist (empty)")
            return ImportResult(id, emptyList(), emptyList())
        }

        val header = rows.first().map { it.trim().lowercase() }
        val titleCol = header.indexOfFirst { it == "track name" }
        val artistCol = header.indexOfFirst { it == "artist name(s)" || it == "artist name" || it == "artist" }
        val albumCol = header.indexOfFirst { it == "album name" || it == "album" }
        val durationCol = header.indexOfFirst { it.startsWith("duration") }

        if (titleCol < 0) {
            val id = playlistRepo.create(playlistName, "Imported playlist")
            return ImportResult(id, emptyList(), listOf(UnmatchedRow(CsvTrackRow("", "", "", 0L), "Missing 'Track Name' column in CSV")))
        }

        val matched = mutableListOf<MatchedTrack>()
        val unmatched = mutableListOf<UnmatchedRow>()

        for (row in rows.drop(1)) {
            val title = row.getOrNull(titleCol)?.trim().orEmpty()
            val artist = if (artistCol >= 0) row.getOrNull(artistCol)?.trim().orEmpty() else ""
            val album = if (albumCol >= 0) row.getOrNull(albumCol)?.trim().orEmpty() else ""
            val durationStr = if (durationCol >= 0) row.getOrNull(durationCol)?.trim().orEmpty() else ""
            val durationMs = durationStr.toLongOrNull() ?: 0L

            val csvRow = CsvTrackRow(title, artist, album, durationMs)
            if (title.isBlank()) {
                unmatched.add(UnmatchedRow(csvRow, "Empty track name"))
                continue
            }

            // Clean title for search
            val cleaned = cleanTitle(title)
            val candidates = mutableListOf<TrackRow>()
            candidates.addAll(libraryRepo.search(cleaned, limit = 20))
            if (cleaned != title) {
                candidates.addAll(libraryRepo.search(title, limit = 20))
            }
            val distinctCandidates = candidates.distinctBy { it.id }

            if (distinctCandidates.isEmpty()) {
                unmatched.add(UnmatchedRow(csvRow, "No candidates found in local library"))
                continue
            }

            var bestTrack: TrackRow? = null
            var bestScore = 0

            val targetCleanTitleKey = matchKey(cleaned)
            val targetArtists = splitArtists(artist)
            val targetAlbumKey = matchKey(album)

            for (cand in distinctCandidates) {
                var score = 0
                val candCleanTitleKey = matchKey(cleanTitle(cand.title))

                // Title score (max 50)
                if (targetCleanTitleKey == candCleanTitleKey) {
                    score += 50
                } else if (targetCleanTitleKey in candCleanTitleKey || candCleanTitleKey in targetCleanTitleKey) {
                    score += 30
                }

                // Artist score (max 25)
                val candArtists = splitArtists(cand.artist)
                if (targetArtists.any { it in candArtists }) {
                    score += 25
                }

                // Duration score (max 15)
                if (csvRow.durationMs > 0 && cand.durationMs > 0) {
                    val diff = abs(cand.durationMs - csvRow.durationMs)
                    if (diff <= 3000L) {
                        score += 15
                    } else if (diff <= 10000L) {
                        score += 5
                    }
                }

                // Album score (max 10)
                if (targetAlbumKey.isNotEmpty() && cand.albumTitle != null) {
                    if (matchKey(cand.albumTitle) == targetAlbumKey) {
                        score += 10
                    }
                }

                if (score > bestScore) {
                    bestScore = score
                    bestTrack = cand
                }
            }

            if (bestTrack != null && bestScore >= MATCH_THRESHOLD) {
                matched.add(MatchedTrack(csvRow, bestTrack, bestScore))
            } else {
                unmatched.add(UnmatchedRow(csvRow, "Best match score ($bestScore) below threshold ($MATCH_THRESHOLD)"))
            }
        }

        val playlistId = playlistRepo.create(playlistName, "Imported from CSV")
        if (matched.isNotEmpty()) {
            playlistRepo.addTracks(playlistId, matched.map { it.track.id })
        }

        return ImportResult(playlistId, matched, unmatched)
    }
}
