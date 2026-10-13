package com.unbound.data.scan

import com.unbound.data.model.FoundFile
import com.unbound.data.model.KnownFile

data class ChangedFile(val known: KnownFile, val found: FoundFile)

data class ScanDiff(
    val added: List<FoundFile>,
    val changed: List<ChangedFile>,
    val unchangedCount: Int,
    /** In the database (and not already flagged) but no longer found. Candidates for move-detection, else marked missing. */
    val missing: List<KnownFile>,
)

/**
 * Pure comparison of what is on disk vs what the library knows. A file is "changed" (needs
 * re-reading) when size or mtime differ, when it was flagged missing but reappeared, or when it
 * was read by an older metadata reader ([currentMetaVersion]).
 */
fun computeScanDiff(known: List<KnownFile>, found: List<FoundFile>, currentMetaVersion: Int): ScanDiff {
    val knownByUri = known.associateBy { it.uri }
    val foundUris = HashSet<String>(found.size * 2)
    val added = ArrayList<FoundFile>()
    val changed = ArrayList<ChangedFile>()
    var unchanged = 0
    for (f in found) {
        foundUris.add(f.uri)
        val k = knownByUri[f.uri]
        when {
            k == null -> added.add(f)
            k.isMissing || k.size != f.size || k.mtime != f.mtime || k.metaVersion < currentMetaVersion -> changed.add(ChangedFile(k, f))
            else -> unchanged++
        }
    }
    val missing = known.filter { !it.isMissing && it.uri !in foundUris }
    return ScanDiff(added, changed, unchanged, missing)
}
