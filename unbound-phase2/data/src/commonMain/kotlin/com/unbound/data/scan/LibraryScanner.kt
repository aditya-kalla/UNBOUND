package com.unbound.data.scan

import com.unbound.data.model.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit

/** Platform-specific discovery: desktop = directory walk, Android = MediaStore / SAF tree, iOS = media library. */
interface MediaFileSource {
    /** All audio files under [source]. Must THROW if the source is unavailable (unmounted drive, revoked permission). */
    suspend fun list(source: ScanSourceInfo): List<FoundFile>
}

/** Wraps the native (Rust) tag reader. Returns null for unreadable / non-audio files. */
interface MetadataReader {
    suspend fun read(file: FoundFile): ParsedTrack?
}

data class ParsedFile(val file: FoundFile, val parsed: ParsedTrack)
data class Relink(val trackId: Long, val file: FoundFile, val parsed: ParsedTrack)
data class ScanBatch(val upserts: List<ParsedFile>, val relinks: List<Relink>)

/** Database side of a scan. Each call must be ONE transaction. */
interface LibraryWriter {
    suspend fun scanIndex(sourceId: Long): List<KnownFile>
    suspend fun applyBatch(sourceId: Long, batch: ScanBatch)
    suspend fun markMissing(trackIds: List<Long>)
    /** Prune empty albums/artists/artwork and stamp the source's last_scan_at. */
    suspend fun finishScan(sourceId: Long, scannedAt: Long)
}

data class ScanSummary(
    val added: Int, val updated: Int, val relinked: Int, val markedMissing: Int,
    val unreadable: Int, val unchanged: Int, val durationMs: Long,
)

sealed interface ScanProgress {
    data object Discovering : ScanProgress
    data class Reading(val done: Int, val total: Int) : ScanProgress
    data class Done(val summary: ScanSummary) : ScanProgress
    data class Failed(val message: String) : ScanProgress
}

class LibraryScanner(
    private val source: MediaFileSource,
    private val reader: MetadataReader,
    private val writer: LibraryWriter,
    private val ioDispatcher: CoroutineDispatcher,
    private val nowMs: () -> Long,
    private val concurrency: Int = 4,
    private val batchSize: Int = 200,
    private val currentMetaVersion: Int = 1,
) {
    fun scan(src: ScanSourceInfo): Flow<ScanProgress> = flow {
        val started = nowMs()
        emit(ScanProgress.Discovering)

        val found = try {
            source.list(src)
        } catch (e: CancellationException) { throw e }
        catch (e: Throwable) { emit(ScanProgress.Failed("Could not read ${src.displayName}: ${e.message}")); return@flow }

        val known = writer.scanIndex(src.id)
        // Safety net: an empty listing for a source that had files means "unavailable", never "user deleted everything".
        if (found.isEmpty() && known.any { !it.isMissing }) {
            emit(ScanProgress.Failed("${src.displayName} returned no files; library left untouched")); return@flow
        }

        val diff = computeScanDiff(known, found, currentMetaVersion)
        val toRead = diff.added + diff.changed.map { it.found }
        val relinkPool = HashMap<String, KnownFile>()                       // fingerprint -> vanished track (move detection)
        for (m in diff.missing) m.fingerprint?.let { relinkPool.putIfAbsent(it, m) }
        val changedByUri = diff.changed.associate { it.found.uri to it.known }
        val relinked = HashSet<Long>()
        val gate = Semaphore(concurrency)
        var added = 0; var updated = 0; var unreadable = 0; var done = 0

        for (chunk in toRead.chunked(batchSize)) {
            val parsed = coroutineScope {
                chunk.map { f -> async(ioDispatcher) { gate.withPermit { f to reader.read(f) } } }.awaitAll()
            }
            val upserts = ArrayList<ParsedFile>(); val relinks = ArrayList<Relink>()
            for ((f, p) in parsed) {
                if (p == null) { unreadable++; continue }
                val moved = if (f.uri !in changedByUri) p.fingerprint?.let { relinkPool[it] } else null
                if (moved != null && relinked.add(moved.id)) { relinks.add(Relink(moved.id, f, p)); relinkPool.remove(p.fingerprint) }
                else { upserts.add(ParsedFile(f, p)); if (f.uri in changedByUri) updated++ else added++ }
            }
            writer.applyBatch(src.id, ScanBatch(upserts, relinks))
            done += chunk.size
            emit(ScanProgress.Reading(done, toRead.size))
        }

        val gone = diff.missing.filter { it.id !in relinked }
        if (gone.isNotEmpty()) writer.markMissing(gone.map { it.id })
        writer.finishScan(src.id, nowMs())
        emit(ScanProgress.Done(ScanSummary(added, updated, relinked.size, gone.size, unreadable, diff.unchangedCount, nowMs() - started)))
    }
}
