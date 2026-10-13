package com.unbound.data.scan

import com.unbound.data.model.FoundFile
import com.unbound.data.model.ScanSourceInfo
import java.io.File

/**
 * Desktop media file discovery: recursive directory walk for supported audio formats.
 * Skips hidden files and directories.
 */
class DesktopMediaFileSource : MediaFileSource {

    private val supportedExtensions = setOf(
        "mp3", "flac", "wav", "m4a", "aac", "ogg", "opus", "wma", "aiff", "ape", "wv"
    )

    override suspend fun list(source: ScanSourceInfo): List<FoundFile> {
        val root = File(source.uri)
        if (!root.exists() || !root.isDirectory || !root.canRead()) {
            throw IllegalArgumentException("Source root does not exist or is not readable: ${source.uri}")
        }

        val found = mutableListOf<FoundFile>()
        walk(root, found)
        return found
    }

    private fun walk(dir: File, out: MutableList<FoundFile>) {
        val files = dir.listFiles() ?: return
        for (f in files) {
            if (f.isHidden || f.name.startsWith(".")) {
                continue
            }
            if (f.isDirectory) {
                walk(f, out)
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
