package com.unbound.data.model

// ---------------------------------------------------------------- enums (stored as TEXT; names are the contract)
enum class StartReason { USER, AUTO_NEXT, AUTOPLAY, QUEUE, RESUME_APP, PREVIOUS }
enum class EndReason { COMPLETED, SKIP_NEXT, SKIP_PREV, USER_STOP, REPLACED, APP_CLOSED, ERROR }
enum class ContextType { ALBUM, PLAYLIST, ARTIST, LIBRARY, LIKED, SEARCH, VIBE_SEARCH, AUTOPLAY, QUEUE }
enum class OutputDeviceType { SPEAKER, WIRED, BLUETOOTH, USB_DAC, CAR, HDMI, UNKNOWN }
enum class RepeatMode { OFF, ALL, ONE }
enum class QueueOrigin { CONTEXT, USER, AUTOPLAY }
enum class ScanSourceKind { FOLDER, MEDIASTORE }
enum class EventType { START, END, SEEK }
enum class TrackSort { TITLE, ARTIST, ALBUM, DATE_ADDED, DURATION }

// ---------------------------------------------------------------- telemetry
data class PlayContext(val type: ContextType?, val id: String?) { companion object { val NONE = PlayContext(null, null) } }
data class OutputDevice(val type: OutputDeviceType, val name: String? = null) { companion object { val UNKNOWN = OutputDevice(OutputDeviceType.UNKNOWN) } }

/** One row destined for `playback_event`. Facts only; "play"/"skip" are derived by the `listen` view. */
data class PlaybackEventDraft(
    val ts: Long,
    val tzOffsetMin: Int,
    val type: EventType,
    val listenId: String,
    val trackId: Long,
    val durationMs: Long,
    val positionMs: Long,
    val seekToMs: Long? = null,
    val listenedMs: Long? = null,
    val startReason: StartReason? = null,
    val endReason: EndReason? = null,
    val context: PlayContext = PlayContext.NONE,
    val device: OutputDevice = OutputDevice.UNKNOWN,
    val shuffle: Boolean = false,
    val repeatMode: RepeatMode = RepeatMode.OFF,
    val speed: Float = 1f,
)

// ---------------------------------------------------------------- scanning
data class ScanSourceInfo(val id: Long, val uri: String, val kind: ScanSourceKind, val displayName: String, val enabled: Boolean = true, val lastScanAt: Long? = null)
data class FoundFile(val uri: String, val size: Long, val mtime: Long)
data class KnownFile(val id: Long, val uri: String, val size: Long, val mtime: Long, val fingerprint: String?, val isMissing: Boolean, val metaVersion: Int)

data class ParsedArtwork(val hash: String, val path: String, val width: Int, val height: Int)

/** What the native metadata reader returns for one audio file. Raw tag values: fallbacks (e.g. "Unknown Artist") are applied when writing. */
data class ParsedTrack(
    val title: String?,
    val artists: List<String>,
    val albumArtist: String?,
    val album: String?,
    val genres: List<String>,
    val trackNo: Int?,
    val discNo: Int?,
    val year: Int?,
    val durationMs: Long,
    val bitrateKbps: Int?,
    val sampleRate: Int?,
    val channels: Int?,
    val bitDepth: Int?,
    val codec: String?,
    val isLossless: Boolean,
    val rgTrackGainDb: Double?,
    val rgTrackPeak: Double?,
    val rgAlbumGainDb: Double?,
    val rgAlbumPeak: Double?,
    val hasEmbeddedLyrics: Boolean,
    val artwork: ParsedArtwork?,
    val fingerprint: String?,
    val metaVersion: Int,
)

// ---------------------------------------------------------------- read models
data class TrackRow(
    val id: Long, val title: String, val artist: String, val albumId: Long?, val albumTitle: String?,
    val durationMs: Long, val artworkId: Long?, val uri: String, val isMissing: Boolean,
)
data class AlbumRow(val id: Long, val title: String, val albumArtist: String, val year: Int?, val artworkId: Long?, val trackCount: Int)
data class ArtistRow(val id: Long, val name: String, val trackCount: Int, val albumCount: Int)
data class GenreRow(val genre: String, val trackCount: Int)
data class LibraryCounts(val tracks: Int, val albums: Int, val artists: Int, val missing: Int)

data class TrackDetail(
    val row: TrackRow, val genres: List<String>, val year: Int?, val trackNo: Int?, val discNo: Int?,
    val codec: String?, val bitrateKbps: Int?, val sampleRate: Int?, val bitDepth: Int?, val channels: Int?, val isLossless: Boolean,
    val rgTrackGainDb: Double?, val rgTrackPeak: Double?, val rgAlbumGainDb: Double?, val rgAlbumPeak: Double?,
    val loudnessLufs: Double?, val hasEmbeddedLyrics: Boolean, val lrcUri: String?,
)

data class PlaylistFolder(val id: Long, val name: String, val parentId: Long?, val sortOrder: Int)
data class PlaylistSummary(val id: Long, val name: String, val description: String, val folderId: Long?, val isPinned: Boolean, val itemCount: Int, val updatedAt: Long)
data class PlaylistEntry(val itemId: Long, val track: TrackRow)

data class QueueEntry(val itemId: Long, val track: TrackRow, val origin: QueueOrigin)
data class PlayerStateSnapshot(
    val currentItemId: Long?, val positionMs: Long, val shuffle: Boolean, val repeatMode: RepeatMode,
    val speed: Float, val context: PlayContext,
)

// ---------------------------------------------------------------- stats
data class TimeRange(val fromTs: Long, val toTs: Long)   // [from, to) UTC epoch ms
data class ListeningTotals(val listenedMs: Long, val plays: Int, val skips: Int, val uniqueTracks: Int)
data class TopTrack(val trackId: Long, val title: String, val artist: String, val plays: Int, val listenedMs: Long)
data class TopArtist(val artistId: Long, val name: String, val plays: Int, val listenedMs: Long)
data class TopGenre(val genre: String, val plays: Int, val listenedMs: Long)
data class HourBucket(val hour: Int, val plays: Int, val listenedMs: Long)
data class WeekdayBucket(val weekday: Int, val plays: Int, val listenedMs: Long)   // 0 = Sunday
data class DayMinutes(val day: String, val listenedMs: Long, val plays: Int)       // day = yyyy-MM-dd (listener's local date)
data class SkippedTrack(val trackId: Long, val title: String, val artist: String, val starts: Int, val skips: Int, val skipRate: Double)
data class LoopedTrack(val trackId: Long, val title: String, val artist: String, val day: String, val plays: Int)
data class DeviceBucket(val deviceType: String, val plays: Int, val listenedMs: Long)
