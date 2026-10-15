package com.unbound.data.benchmark

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.impl.SqlDelightLibraryRepository
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.scan.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertTrue

class LibraryBenchmarkTest {

    private class SyntheticSource(val fileList: List<FoundFile>) : MediaFileSource {
        override suspend fun list(source: ScanSourceInfo): List<FoundFile> = fileList
    }

    private class SyntheticReader(val trackList: List<ParsedTrack>) : MetadataReader {
        override suspend fun read(file: FoundFile): ParsedTrack? {
            val idx = file.uri.substringAfter("track_").substringBefore(".mp3").toInt()
            return trackList[idx]
        }
    }

    @Test
    fun benchmark50kLibrary() = runBlocking {
        println("=== STARTING 50,000-TRACK PERFORMANCE BENCHMARK ===")
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val libraryRepo = SqlDelightLibraryRepository(db, driver)

        val sourceId = libraryRepo.addSource("file:///benchmark_music", ScanSourceKind.FOLDER, "Benchmark")
        val sourceInfo = ScanSourceInfo(sourceId, "file:///benchmark_music", ScanSourceKind.FOLDER, "Benchmark")

        val totalTracks = 50_000
        println("Generating $totalTracks synthetic tracks...")
        val files = ArrayList<FoundFile>(totalTracks)
        val tracks = ArrayList<ParsedTrack>(totalTracks)

        val genresList = listOf("Rock", "Pop", "Jazz", "Classical", "Electronic", "Hip Hop", "Metal", "Ambient")

        for (i in 0 until totalTracks) {
            val artistIdx = i % 2500
            val albumIdx = i % 5000
            val genre = genresList[i % genresList.size]
            val uri = "file:///benchmark_music/track_$i.mp3"

            files.add(FoundFile(uri, size = 3_000_000L + i, mtime = 1_700_000_000L))
            tracks.add(
                testTrack(
                    title = "Track $i Synthetic Song",
                    artists = listOf("Artist $artistIdx"),
                    albumArtist = "Artist $artistIdx",
                    album = "Album $albumIdx Collection",
                    genres = listOf(genre),
                    trackNo = (i % 10) + 1,
                    discNo = 1,
                    year = 2000 + (i % 25),
                    durationMs = 180_000L + (i % 120_000),
                    fingerprint = "fp-$i",
                )
            )
        }

        val fakeSource = SyntheticSource(files)
        val fakeReader = SyntheticReader(tracks)

        var clock = 1_000_000L
        val scanner = LibraryScanner(
            source = fakeSource,
            reader = fakeReader,
            writer = writer,
            ioDispatcher = Dispatchers.Default,
            nowMs = { clock += 1000; clock },
            concurrency = 4,
            batchSize = 1000,
        )

        // 1. Time for full scan write
        println("Executing full scan write of $totalTracks tracks...")
        val fullScanMs = measureTimeMillis {
            val progress = scanner.scan(sourceInfo).toList()
            val done = progress.filterIsInstance<ScanProgress.Done>().single()
            println("Full scan completed: added=${done.summary.added}, durationMs=${done.summary.durationMs}")
        }
        val writeRate = (totalTracks.toDouble() / (fullScanMs / 1000.0)).toInt()
        println(">>> BENCHMARK RESULT: Full scan write time: ${fullScanMs} ms ($writeRate tracks/sec)")

        // 2. Time for unchanged rescan (target <= 3 s)
        println("Executing unchanged rescan of $totalTracks tracks...")
        val rescanMs = measureTimeMillis {
            val progress = scanner.scan(sourceInfo).toList()
            val done = progress.filterIsInstance<ScanProgress.Done>().single()
            println("Rescan completed: unchanged=${done.summary.unchanged}, added=${done.summary.added}, updated=${done.summary.updated}")
            assertTrue(done.summary.unchanged == totalTracks)
        }
        println(">>> BENCHMARK RESULT: Unchanged rescan time: ${rescanMs} ms (Target <= 3000 ms)")
        assertTrue(rescanMs <= 3000, "Unchanged rescan took ${rescanMs}ms, exceeding target of 3000ms")

        // 3. First page of all sorts (< 50 ms)
        println("Executing first page for all sorts (paged query with LIMIT 50)...")
        for (sort in TrackSort.values()) {
            // Warmup
            libraryRepo.tracksPage(sort, limit = 50, offset = 0)
            val pagedTimes = LongArray(5)
            for (attempt in 0 until 5) {
                pagedTimes[attempt] = measureTimeMillis {
                    val page = libraryRepo.tracksPage(sort, limit = 50, offset = 0)
                    assertTrue(page.size == 50)
                }
            }
            val firstPageMs = pagedTimes.average()
            println(">>> BENCHMARK RESULT: tracksPage($sort) (paged first 50 rows): ${firstPageMs} ms (runs: ${pagedTimes.joinToString()}) (Target < 50 ms)")
            assertTrue(firstPageMs < 50.0, "Paged first page for $sort took ${firstPageMs}ms, exceeding target of 50ms")
        }

        // 4. FTS search (< 50 ms)
        println("Executing FTS search...")
        // Warmup
        libraryRepo.search("Track 250", limit = 50)

        val ftsTimes = LongArray(5)
        for (attempt in 0 until 5) {
            ftsTimes[attempt] = measureTimeMillis {
                val results = libraryRepo.search("Track 250", limit = 50)
                assertTrue(results.isNotEmpty())
            }
        }
        val ftsAvgMs = ftsTimes.average()
        println(">>> BENCHMARK RESULT: FTS search time: ${ftsAvgMs} ms (runs: ${ftsTimes.joinToString()}) (Target < 50 ms)")

        // 5. EXPLAIN QUERY PLAN analysis on list and stats queries
        println("\n=== EXPLAIN QUERY PLAN AUDIT ===")
        val queriesToExplain = listOf(
            "tracksWithAlbumByTitlePage" to "EXPLAIN QUERY PLAN SELECT t.id, t.title, t.artist_display, t.album_id, a.title AS album_title, t.duration_ms, t.artwork_id, t.uri, t.is_missing FROM track t LEFT JOIN album a ON a.id = t.album_id WHERE t.is_missing = 0 ORDER BY t.title_sort, t.id LIMIT 50 OFFSET 0;",
            "tracksWithAlbumByArtistPage" to "EXPLAIN QUERY PLAN SELECT t.id, t.title, t.artist_display, t.album_id, a.title AS album_title, t.duration_ms, t.artwork_id, t.uri, t.is_missing FROM track t LEFT JOIN album a ON a.id = t.album_id WHERE t.is_missing = 0 ORDER BY t.artist_display COLLATE NOCASE, t.title_sort, t.id LIMIT 50 OFFSET 0;",
            "tracksWithAlbumByAlbumPage" to "EXPLAIN QUERY PLAN SELECT t.id, t.title, t.artist_display, t.album_id, a.title AS album_title, t.duration_ms, t.artwork_id, t.uri, t.is_missing FROM album a CROSS JOIN track t ON t.album_id = a.id WHERE t.is_missing = 0 ORDER BY a.title_sort, a.id, t.disc_no, t.track_no, t.id LIMIT 50 OFFSET 0;",
            "tracksWithAlbumByDateAddedPage" to "EXPLAIN QUERY PLAN SELECT t.id, t.title, t.artist_display, t.album_id, a.title AS album_title, t.duration_ms, t.artwork_id, t.uri, t.is_missing FROM track t LEFT JOIN album a ON a.id = t.album_id WHERE t.is_missing = 0 ORDER BY t.added_at DESC, t.id DESC LIMIT 50 OFFSET 0;",
            "tracksWithAlbumByDurationPage" to "EXPLAIN QUERY PLAN SELECT t.id, t.title, t.artist_display, t.album_id, a.title AS album_title, t.duration_ms, t.artwork_id, t.uri, t.is_missing FROM track t LEFT JOIN album a ON a.id = t.album_id WHERE t.is_missing = 0 ORDER BY t.duration_ms DESC, t.id DESC LIMIT 50 OFFSET 0;",
            "allAlbums" to "EXPLAIN QUERY PLAN SELECT a.id, a.title, a.album_artist_display AS album_artist, a.year, a.artwork_id, (SELECT COUNT(*) FROM track t WHERE t.album_id = a.id AND t.is_missing = 0) AS track_count FROM album a WHERE (SELECT COUNT(*) FROM track t WHERE t.album_id = a.id AND t.is_missing = 0) > 0 ORDER BY a.title_sort, a.id LIMIT 50;",
            "allArtists" to "EXPLAIN QUERY PLAN SELECT a.id, a.name, (SELECT COUNT(DISTINCT ta.track_id) FROM track_artist ta JOIN track t ON t.id = ta.track_id WHERE ta.artist_id = a.id AND t.is_missing = 0) AS track_count FROM artist a ORDER BY a.name_sort, a.id LIMIT 50;",
            "allGenres" to "EXPLAIN QUERY PLAN SELECT tg.genre, COUNT(DISTINCT t.id) AS track_count FROM track_genre tg JOIN track t ON t.id = tg.track_id WHERE t.is_missing = 0 GROUP BY tg.genre ORDER BY tg.genre COLLATE NOCASE;",
            "searchFts" to "EXPLAIN QUERY PLAN SELECT rowid FROM track_search WHERE track_search MATCH '\"track\"*' ORDER BY bm25(track_search) LIMIT 50;",
            "statsTotals" to "EXPLAIN QUERY PLAN SELECT COALESCE(SUM(listened_ms), 0), COALESCE(SUM(counted), 0), COALESCE(SUM(skipped), 0), COUNT(DISTINCT CASE WHEN counted = 1 THEN track_id END) FROM listen WHERE started_at >= 1000 AND started_at < 2000;",
            "statsTopTracks" to "EXPLAIN QUERY PLAN SELECT l.track_id, t.title, t.artist_display, COUNT(*) AS plays, SUM(l.listened_ms) AS listened_ms FROM listen l JOIN track t ON t.id = l.track_id WHERE l.counted = 1 AND l.started_at >= 1000 AND l.started_at < 2000 GROUP BY l.track_id ORDER BY plays DESC LIMIT 20;",
            "statsHourHistogram" to "EXPLAIN QUERY PLAN SELECT ((((l.started_at / 1000) + l.tz_offset_min * 60) % 86400) + 86400) % 86400 / 3600 AS hour, COUNT(*), COALESCE(SUM(l.listened_ms), 0) FROM listen l WHERE l.counted = 1 AND l.started_at >= 1000 AND l.started_at < 2000 GROUP BY hour ORDER BY hour;"
        )

        for ((name, sql) in queriesToExplain) {
            println("--- Query Plan: $name ---")
            val plan = mutableListOf<String>()
            driver.executeQuery(
                null,
                sql,
                mapper = { cursor ->
                    while (cursor.next().value) {
                        val detail = cursor.getString(3) ?: cursor.getString(2) ?: ""
                        plan.add(detail)
                    }
                    app.cash.sqldelight.db.QueryResult.Unit
                },
                parameters = 0
            )
            for (line in plan) {
                println("  $line")
                if (name.startsWith("tracksWithAlbum") && name.endsWith("Page")) {
                    assertTrue(!line.contains("TEMP B-TREE", ignoreCase = true), "Query $name must not use temp B-tree: $line")
                }
            }
        }
        println("=== BENCHMARK COMPLETED SUCCESSFULLY ===")
    }
}
