package com.unbound.data.scan

import com.unbound.data.model.FoundFile
import com.unbound.data.model.ScanSourceInfo
import java.io.File

/**
 * Desktop media file discovery: recursive directory walk for supported audio formats.
 * Skips hidden files and directories, guards against symlink/junction loops by tracking
 * canonical paths, and skips/counts unreadable subfolders without aborting the scan.
 */
class DesktopMediaFileSource(
    private val onUnreadableFolder: ((File) -> Unit)? = null,
) : MediaFileSource {

    private val supportedExtensions = setOf(
        "mp3", "flac", "wav", "m4a", "aac", "ogg", "opus", "wma", "aiff", "ape", "wv"
    )

    var unreadableFolderCount: Int = 0
        private set

    override suspend fun list(source: ScanSourceInfo): List<FoundFile> {
        val root = File(source.uri)
        if (!root.exists() || !root.isDirectory || !root.canRead()) {
            throw IllegalArgumentException("Source root does not exist or is not readable: ${source.uri}")
        }

        val found = mutableListOf<FoundFile>()
        val visitedDirs = mutableSetOf<String>()
        unreadableFolderCount = 0
        walk(root, found, visitedDirs)
        return found
    }

    private fun walk(dir: File, out: MutableList<FoundFile>, visitedDirs: MutableSet<String>) {
        val canonicalPath = try {
            dir.toPath().toRealPath().toString()
        } catch (_: Exception) {
            try { dir.canonicalPath } catch (_: Exception) { dir.absolutePath }
        }
        if (!visitedDirs.add(canonicalPath)) {
            // Already visited this real/canonical directory — break cycle / loop!
            return
        }

        val files = dir.listFiles()
        if (files == null) {
            // Unreadable subfolder: skip and count it, don't abort the whole scan
            unreadableFolderCount++
            onUnreadableFolder?.invoke(dir)
            return
        }

        for (f in files) {
            if (f.isHidden || f.name.startsWith(".")) {
                continue
            }
            if (f.isDirectory) {
                walk(f, out, visitedDirs)
            } else if (f.isFile) {
                val ext = f.extension.lowercase()
                if (ext in supportedExtensions) {
                    out.add(
                        FoundFile(
                            uri = f.absolutePath,
                            size = f.length(),
                            mtime = f.lastModified(),
                        )
                    )
                }
            }
        }
    }
}
