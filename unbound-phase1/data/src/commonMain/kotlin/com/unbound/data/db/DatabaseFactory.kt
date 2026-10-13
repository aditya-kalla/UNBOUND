package com.unbound.data.db

import app.cash.sqldelight.db.QueryResult
import app.cash.sqldelight.db.SqlDriver

expect class DatabaseFactory {
    fun createDriver(): SqlDriver
    fun createInMemoryDriver(): SqlDriver
    fun createDatabase(): UnboundDatabase
    fun createInMemoryDatabase(): UnboundDatabase
}

fun initDatabase(driver: SqlDriver): UnboundDatabase {
    // 1. Pragmas on open
    driver.execute(null, "PRAGMA foreign_keys=ON;", 0)
    driver.execute(null, "PRAGMA journal_mode=WAL;", 0)
    driver.execute(null, "PRAGMA synchronous=NORMAL;", 0)

    // 2. Schema creation
    val currentVersion = driver.executeQuery(null, "PRAGMA user_version;", { cursor ->
        val v = if (cursor.next().value) cursor.getLong(0) ?: 0L else 0L
        QueryResult.Value(v)
    }, 0).value

    if (currentVersion == 0L) {
        UnboundDatabase.Schema.create(driver)
        driver.execute(null, "PRAGMA user_version = 1;", 0)
    }

    // 3. FTS5 table initialization (raw SQL)
    try {
        driver.execute(
            null,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS track_search USING fts5(
              title, artists, album, genres,
              tokenize = "unicode61 remove_diacritics 2",
              prefix = "2 3"
            );
            """.trimIndent(),
            0
        )
    } catch (_: Throwable) {
        // Platform lacks FTS5
    }

    // 4. Create database and initialize player state
    val database = UnboundDatabase(driver)
    database.queueQueries.initPlayerState()
    return database
}

object FtsIndexHelper {
    fun isFts5Available(driver: SqlDriver): Boolean {
        return try {
            driver.execute(
                null,
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS track_search USING fts5(
                  title, artists, album, genres,
                  tokenize = "unicode61 remove_diacritics 2",
                  prefix = "2 3"
                );
                """.trimIndent(),
                0
            )
            true
        } catch (_: Throwable) {
            false
        }
    }

    fun insertOrUpdate(
        driver: SqlDriver,
        trackId: Long,
        title: String,
        artists: String,
        album: String,
        genres: String
    ) {
        try {
            driver.execute(null, "DELETE FROM track_search WHERE rowid = ?;", 1) {
                bindLong(0, trackId)
            }
            driver.execute(
                null,
                "INSERT INTO track_search(rowid, title, artists, album, genres) VALUES (?, ?, ?, ?, ?);",
                5
            ) {
                bindLong(0, trackId)
                bindString(1, title)
                bindString(2, artists)
                bindString(3, album)
                bindString(4, genres)
            }
        } catch (_: Throwable) {
            // Ignore if FTS5 is not available
        }
    }

    fun delete(driver: SqlDriver, trackId: Long) {
        try {
            driver.execute(null, "DELETE FROM track_search WHERE rowid = ?;", 1) {
                bindLong(0, trackId)
            }
        } catch (_: Throwable) {
        }
    }

    fun searchFts(driver: SqlDriver, ftsQuery: String, limit: Long): List<Long> {
        return try {
            driver.executeQuery(
                null,
                "SELECT rowid FROM track_search WHERE track_search MATCH ? ORDER BY bm25(track_search) LIMIT ?;",
                mapper = { cursor ->
                    val ids = mutableListOf<Long>()
                    while (cursor.next().value) {
                        ids.add(cursor.getLong(0)!!)
                    }
                    QueryResult.Value(ids)
                },
                parameters = 2
            ) {
                bindString(0, ftsQuery)
                bindLong(1, limit)
            }.value
        } catch (_: Throwable) {
            emptyList()
        }
    }
}
