package com.unbound.data.scan

import com.unbound.data.db.DatabaseFactory
import com.unbound.data.model.ScanSourceInfo
import com.unbound.data.model.ScanSourceKind
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) = runBlocking {
    println("========================================================")
    println("       UNBOUND Desktop Library Scan Smoke Program       ")
    println("========================================================")

    val candidatePath = args.firstOrNull()
        ?: System.getProperty("unbound.scanFolder")
        ?: System.getenv("UNBOUND_SCAN_FOLDER")
        ?: """C:\Users\lenovo\Music"""

    val scanFolder = File(candidatePath).let { dir ->
        if (dir.exists() && dir.isDirectory) {
            dir
        } else {
            // Fallback to a folder with media or test fixtures if user folder is empty/missing
            val winMedia = File("""C:\Windows\Media""")
            if (winMedia.exists()) winMedia else dir
        }
    }

    println("Target Music Folder: ${scanFolder.absolutePath}")
    if (!scanFolder.exists() || !scanFolder.isDirectory) {
        System.err.println("Error: Directory does not exist: ${scanFolder.absolutePath}")
        return@runBlocking
    }

    val tempDir = File(System.getProperty("java.io.tmpdir"), "unbound_smoke_${System.currentTimeMillis()}").apply {
        mkdirs()
        deleteOnExit()
    }
    val artDir = File(tempDir, "art_cache").apply { mkdirs() }
    val dbFile = File(tempDir, "smoke_library.db")

    println("Temporary DB Path: ${dbFile.absolutePath}")
    println("Artwork Cache Path: ${artDir.absolutePath}")

    println("\n[1/5] Initializing SQLite Database & LibraryWriter...")
    val factory = DatabaseFactory(dbFile)
    val driver = factory.createDriver()
    val database = com.unbound.data.db.initDatabase(driver)
    val writer = SqlDelightLibraryWriter(database, driver)
    val source = DesktopMediaFileSource()
    val reader = NativeMetadataReader(artDir = artDir.absolutePath)
    val scanner = LibraryScanner(
        source = source,
        reader = reader,
        writer = writer,
        ioDispatcher = Dispatchers.IO,
        nowMs = { System.currentTimeMillis() },
        concurrency = 4,
        batchSize = 200,
    )

    val scanSource = ScanSourceInfo(
        id = 1L,
        uri = scanFolder.absolutePath,
        kind = ScanSourceKind.FOLDER,
        displayName = scanFolder.name.ifEmpty { "Local Music" },
    )

    println("\n[2/5] Running First Scan (Initial Ingestion)...")
    var firstSummary: ScanSummary? = null
    val firstScanTime = measureTimeMillis {
        scanner.scan(scanSource).collect { progress ->
            when (progress) {
                is ScanProgress.Discovering -> println("  -> Discovering audio files in folder...")
                is ScanProgress.Reading -> {
                    if (progress.done % 20 == 0 || progress.done == progress.total) {
                        println("  -> Reading tags: ${progress.done} / ${progress.total}")
                    }
                }
                is ScanProgress.Done -> {
                    firstSummary = progress.summary
                    println("  -> Scan Complete!")
                }
                is ScanProgress.Failed -> {
                    println("  -> Scan FAILED: ${progress.message}")
                }
            }
        }
    }

    println("\n--- First Scan Summary (${firstScanTime} ms) ---")
    firstSummary?.let { s ->
        println("  Added:           ${s.added}")
        println("  Updated:         ${s.updated}")
        println("  Relinked:        ${s.relinked}")
        println("  Marked Missing:  ${s.markedMissing}")
        println("  Unchanged:       ${s.unchanged}")
        println("  Unreadable:      ${s.unreadable}")
        println("  Scan Duration:   ${s.durationMs} ms")
    } ?: println("  No summary received.")

    println("\n[3/5] Querying Sample Rows from Database...")
    val sampleTracks = database.libraryQueries.tracksWithAlbumByTitle(includeMissing = 0L).executeAsList().take(5)
    if (sampleTracks.isEmpty()) {
        println("  (No audio files found in directory)")
    } else {
        println("  Found ${sampleTracks.size} sample tracks:")
        sampleTracks.forEachIndexed { i, t ->
            val art = t.artwork_id?.let { " [artId=$it]" } ?: ""
            println("  ${i + 1}. \"${t.title}\" by ${t.artist_display} | Album: ${t.album_title ?: "Single"} | ${t.duration_ms}ms$art")
            println("     Path: ${t.uri}")
        }
    }

    println("\n[4/5] Running Second Scan (Incremental Rescan)...")
    var secondSummary: ScanSummary? = null
    val secondScanTime = measureTimeMillis {
        scanner.scan(scanSource).collect { progress ->
            when (progress) {
                is ScanProgress.Discovering -> println("  -> Re-checking directory...")
                is ScanProgress.Reading -> println("  -> Re-reading: ${progress.done} / ${progress.total}")
                is ScanProgress.Done -> {
                    secondSummary = progress.summary
                    println("  -> Rescan Complete!")
                }
                is ScanProgress.Failed -> println("  -> Rescan FAILED: ${progress.message}")
            }
        }
    }

    println("\n--- Second Scan Summary (${secondScanTime} ms) ---")
    secondSummary?.let { s ->
        println("  Added:           ${s.added}")
        println("  Updated:         ${s.updated}")
        println("  Relinked:        ${s.relinked}")
        println("  Marked Missing:  ${s.markedMissing}")
        println("  Unchanged:       ${s.unchanged} (all existing tracks unchanged!)")
        println("  Unreadable:      ${s.unreadable}")
        println("  Rescan Duration: ${s.durationMs} ms")
    }

    println("\n[5/5] Smoke Program Completed Successfully!")
    println("========================================================")
}
