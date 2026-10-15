package com.unbound.data.db

import kotlin.test.Test
import kotlin.test.assertTrue

class DesktopDatabaseTest {
    @Test
    fun sqliteVersionIsAtLeast3_38AndFts5Works() {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)

        // 1. Assert sqlite_version() >= 3.38
        val version = driver.executeQuery(null, "SELECT sqlite_version()", { cursor ->
            cursor.next()
            app.cash.sqldelight.db.QueryResult.Value(cursor.getString(0)!!)
        }, 0).value

        println("Desktop SQLite version: $version")
        val parts = version.split(".").map { it.toIntOrNull() ?: 0 }
        val major = parts.getOrElse(0) { 0 }
        val minor = parts.getOrElse(1) { 0 }
        assertTrue(major > 3 || (major == 3 && minor >= 38), "SQLite version should be >= 3.38, was $version")

        // 2. Assert FTS5 works
        assertTrue(FtsIndexHelper.isFts5Available(driver), "FTS5 should be available on desktop driver")

        // Test inserting and querying FTS5 with diacritics and prefix
        FtsIndexHelper.insertOrUpdate(driver, 1L, "Renaissance", "Beyoncé", "Renaissance", "Dance;Pop")
        val results = FtsIndexHelper.searchFts(driver, "\"beyonce\"*", 10)
        assertTrue(results.contains(1L), "FTS5 search should find track by artist ignoring diacritics")
    }
}
