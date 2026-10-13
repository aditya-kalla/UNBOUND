package com.unbound.data.repo.impl

import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.*
import com.unbound.data.repo.StatsRepository
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SqlDelightStatsRepository(
    private val db: UnboundDatabase,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
) : StatsRepository {

    override suspend fun totals(range: TimeRange): ListeningTotals = withContext(dispatcher) {
        val row = db.statsQueries.totals(range.fromTs, range.toTs).executeAsOne()
        ListeningTotals(
            listenedMs = row.listened_ms.toLong(),
            plays = row.plays.toInt(),
            skips = row.skips.toInt(),
            uniqueTracks = row.unique_tracks.toInt(),
        )
    }

    override suspend fun topTracks(range: TimeRange, limit: Int): List<TopTrack> = withContext(dispatcher) {
        db.statsQueries.topTracks(range.fromTs, range.toTs, limit.toLong()).executeAsList().map { row ->
            TopTrack(
                trackId = row.track_id ?: 0L,
                title = row.title,
                artist = row.artist_display,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms?.toLong() ?: 0L,
            )
        }
    }

    override suspend fun topArtists(range: TimeRange, limit: Int): List<TopArtist> = withContext(dispatcher) {
        db.statsQueries.topArtists(range.fromTs, range.toTs, limit.toLong()).executeAsList().map { row ->
            TopArtist(
                artistId = row.artist_id,
                name = row.name,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms?.toLong() ?: 0L,
            )
        }
    }

    override suspend fun topGenres(range: TimeRange, limit: Int): List<TopGenre> = withContext(dispatcher) {
        db.statsQueries.topGenres(range.fromTs, range.toTs, limit.toLong()).executeAsList().map { row ->
            TopGenre(
                genre = row.genre,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms?.toLong() ?: 0L,
            )
        }
    }

    override suspend fun hourHistogram(range: TimeRange): List<HourBucket> = withContext(dispatcher) {
        db.statsQueries.hourHistogram(range.fromTs, range.toTs).executeAsList().map { row ->
            HourBucket(
                hour = (row.hour ?: 0L).toInt(),
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms.toLong(),
            )
        }
    }

    override suspend fun topTracksInHourWindow(
        range: TimeRange,
        startHour: Int,
        endHour: Int,
        limit: Int,
    ): List<TopTrack> = withContext(dispatcher) {
        val wrap = if (startHour <= endHour) 0L else 1L
        db.statsQueries.topTracksInHourWindow(
            fromTs = range.fromTs,
            toTs = range.toTs,
            wrap = wrap,
            startHour = startHour.toLong(),
            endHour = endHour.toLong(),
            limit = limit.toLong(),
        ).executeAsList().map { row ->
            TopTrack(
                trackId = row.track_id ?: 0L,
                title = row.title,
                artist = row.artist_display,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms?.toLong() ?: 0L,
            )
        }
    }

    override suspend fun weekdayHistogram(range: TimeRange): List<WeekdayBucket> = withContext(dispatcher) {
        db.statsQueries.weekdayHistogram(range.fromTs, range.toTs).executeAsList().map { row ->
            WeekdayBucket(
                weekday = row.weekday?.toInt() ?: 0,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms.toLong(),
            )
        }
    }

    override suspend fun dailyMinutes(range: TimeRange): List<DayMinutes> = withContext(dispatcher) {
        db.statsQueries.dailyMinutes(range.fromTs, range.toTs).executeAsList().map { row ->
            DayMinutes(
                day = row.day ?: "",
                listenedMs = row.listened_ms.toLong(),
                plays = row.plays.toInt(),
            )
        }
    }

    override suspend fun mostSkipped(range: TimeRange, minStarts: Int, limit: Int): List<SkippedTrack> = withContext(dispatcher) {
        db.statsQueries.mostSkipped(range.fromTs, range.toTs, minStarts.toLong(), limit.toLong()).executeAsList().map { row ->
            SkippedTrack(
                trackId = row.track_id ?: 0L,
                title = row.title,
                artist = row.artist_display,
                starts = row.starts.toInt(),
                skips = (row.skips ?: 0L).toInt(),
                skipRate = row.skip_rate ?: 0.0,
            )
        }
    }

    override suspend fun loopedTracks(range: TimeRange, minPlays: Int, limit: Int): List<LoopedTrack> = withContext(dispatcher) {
        db.statsQueries.loopedTracks(range.fromTs, range.toTs, minPlays.toLong(), limit.toLong()).executeAsList().map { row ->
            LoopedTrack(
                trackId = row.track_id ?: 0L,
                title = row.title,
                artist = row.artist_display,
                day = row.day ?: "",
                plays = row.plays.toInt(),
            )
        }
    }

    override suspend fun deviceBreakdown(range: TimeRange): List<DeviceBucket> = withContext(dispatcher) {
        db.statsQueries.deviceBreakdown(range.fromTs, range.toTs).executeAsList().map { row ->
            DeviceBucket(
                deviceType = row.device_type,
                plays = row.plays.toInt(),
                listenedMs = row.listened_ms.toLong(),
            )
        }
    }
}
