package com.unbound.data.scan

import com.unbound.data.model.ScanSourceInfo
import com.unbound.data.model.ScanSourceKind
import kotlinx.coroutines.runBlocking
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DesktopMediaFileSourceTest {

    @Test
    fun testLoopProtectionAndUnreadableDirectoryHandling() = runBlocking {
        val tempDir = Files.createTempDirectory("unbound_source_test").toFile()
        try {
            val musicDir = File(tempDir, "music").apply { mkdirs() }
            val song1 = File(musicDir, "song1.mp3").apply { writeBytes(ByteArray(100)) }

            val subDir = File(musicDir, "sub").apply { mkdirs() }
            val song2 = File(subDir, "song2.flac").apply { writeBytes(ByteArray(200)) }

            // Try creating a symbolic link loop: sub/loop -> musicDir
            val loopLink = File(subDir, "loop")
            var symlinkCreated = false
            try {
                Files.createSymbolicLink(loopLink.toPath(), musicDir.toPath())
                symlinkCreated = true
            } catch (_: Throwable) {
                // Windows might require elevated privileges for symlinks
            }

            val source = DesktopMediaFileSource()
            val info = ScanSourceInfo(1L, musicDir.absolutePath, ScanSourceKind.FOLDER, "Test Music")

            val files = source.list(info)
            val fileNames = files.map { File(it.uri).name }.sorted()

            // Even with loop present, it must terminate and find exactly the real files without duplicate infinite recursion
            assertEquals(listOf("song1.mp3", "song2.flac"), fileNames)

            // Test unreadable subfolder: create subfolder and make it unreadable if possible
            val unreadableDir = File(musicDir, "locked").apply { mkdirs() }
            val songInLocked = File(unreadableDir, "locked_song.mp3").apply { writeBytes(ByteArray(50)) }
            val readBlocked = unreadableDir.setReadable(false)

            val source2 = DesktopMediaFileSource()
            val files2 = source2.list(info)
            // Scan must not crash; it either visits or skips and counts
            assertTrue(files2.isNotEmpty())
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
