package com.unbound.data.scan

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.unbound.data.model.FoundFile
import com.unbound.data.model.ParsedArtwork
import com.unbound.data.model.ParsedTrack
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Wraps the native Rust tag reader (lofty + blake3 + image) through JNA.
 * Shared between Android and Desktop via the `jvmCommonMain` source set.
 */
class NativeMetadataReader(
    libraryName: String = "unbound_audio",
    private val artDir: String? = null,
) : MetadataReader {

    private interface RustMetaLib : Library {
        fun engine_abi_version(): Int
        fun meta_read_json(path: String, artDir: String?): Pointer?
        fun meta_free_string(ptr: Pointer)
    }

    init {
        // Rust expects UTF-8 paths; JNA defaults to the OS charset (breaks non-English song titles on Windows).
        System.setProperty("jna.encoding", "UTF-8")
    }

    private val lib: RustMetaLib = Native.load(libraryName, RustMetaLib::class.java).also { loaded ->
        val version = loaded.engine_abi_version()
        require(version == EXPECTED_ABI_VERSION) {
            "Stale native library: expected ABI version $EXPECTED_ABI_VERSION but loaded $version. " +
                "Replace the DLL/so/dylib with a matching build."
        }
    }

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    @Serializable
    private data class JsonArtwork(
        val hash: String,
        val path: String,
        val width: Int,
        val height: Int,
    )

    @Serializable
    private data class JsonMetaResponse(
        val ok: Boolean,
        val error: String? = null,
        val title: String? = null,
        val artists: List<String> = emptyList(),
        val album_artist: String? = null,
        val album: String? = null,
        val genres: List<String> = emptyList(),
        val track_no: Int? = null,
        val disc_no: Int? = null,
        val year: Int? = null,
        val duration_ms: Long = 0,
        val bitrate_kbps: Int? = null,
        val sample_rate: Int? = null,
        val channels: Int? = null,
        val bit_depth: Int? = null,
        val codec: String? = null,
        val lossless: Boolean = false,
        val rg_track_gain_db: Double? = null,
        val rg_track_peak: Double? = null,
        val rg_album_gain_db: Double? = null,
        val rg_album_peak: Double? = null,
        val has_embedded_lyrics: Boolean = false,
        val artwork: JsonArtwork? = null,
        val fingerprint: String? = null,
        val meta_version: Int = 1,
    )

    override suspend fun read(file: FoundFile): ParsedTrack? {
        val ptr = lib.meta_read_json(file.uri, artDir) ?: return null
        val jsonStr = try {
            ptr.getString(0, "UTF-8")
        } finally {
            lib.meta_free_string(ptr)
        }

        return try {
            val resp = json.decodeFromString<JsonMetaResponse>(jsonStr)
            if (!resp.ok) return null
            ParsedTrack(
                title = resp.title,
                artists = resp.artists,
                albumArtist = resp.album_artist,
                album = resp.album,
                genres = resp.genres,
                trackNo = resp.track_no,
                discNo = resp.disc_no,
                year = resp.year,
                durationMs = resp.duration_ms,
                bitrateKbps = resp.bitrate_kbps,
                sampleRate = resp.sample_rate,
                channels = resp.channels,
                bitDepth = resp.bit_depth,
                codec = resp.codec,
                isLossless = resp.lossless,
                rgTrackGainDb = resp.rg_track_gain_db,
                rgTrackPeak = resp.rg_track_peak,
                rgAlbumGainDb = resp.rg_album_gain_db,
                rgAlbumPeak = resp.rg_album_peak,
                hasEmbeddedLyrics = resp.has_embedded_lyrics,
                artwork = resp.artwork?.let {
                    ParsedArtwork(
                        hash = it.hash,
                        path = it.path,
                        width = it.width,
                        height = it.height,
                    )
                },
                fingerprint = resp.fingerprint,
                metaVersion = resp.meta_version,
            )
        } catch (e: Exception) {
            null
        }
    }

    companion object {
        const val EXPECTED_ABI_VERSION = 3
    }
}
