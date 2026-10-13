package com.unbound.data

import com.unbound.data.model.FoundFile
import com.unbound.data.model.KnownFile
import com.unbound.data.scan.computeScanDiff
import kotlin.test.Test
import kotlin.test.assertEquals

class ScanDiffTest {
    private fun known(id: Long, uri: String, size: Long = 10, mtime: Long = 5, missing: Boolean = false, v: Int = 1) =
        KnownFile(id, uri, size, mtime, "fp$id", missing, v)

    @Test fun classifiesAddedChangedUnchangedMissing() {
        val k = listOf(known(1, "/a"), known(2, "/b"), known(3, "/c"))
        val f = listOf(FoundFile("/a", 10, 5), FoundFile("/b", 11, 5), FoundFile("/d", 1, 1))
        val d = computeScanDiff(k, f, 1)
        assertEquals(listOf("/d"), d.added.map { it.uri })
        assertEquals(listOf("/b"), d.changed.map { it.found.uri })
        assertEquals(1, d.unchangedCount)
        assertEquals(listOf(3L), d.missing.map { it.id })
    }

    @Test fun reappearedMissingFileIsReRead() {
        val d = computeScanDiff(listOf(known(1, "/a", missing = true)), listOf(FoundFile("/a", 10, 5)), 1)
        assertEquals(1, d.changed.size)
    }

    @Test fun outdatedReaderVersionTriggersRescan() {
        val d = computeScanDiff(listOf(known(1, "/a", v = 1)), listOf(FoundFile("/a", 10, 5)), 2)
        assertEquals(1, d.changed.size)
    }

    @Test fun alreadyMissingFilesAreNotReportedMissingAgain() {
        val d = computeScanDiff(listOf(known(1, "/gone", missing = true)), emptyList(), 1)
        assertEquals(0, d.missing.size)
    }
}
