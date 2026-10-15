package com.unbound.data.db

import app.cash.sqldelight.db.SqlDriver
import app.cash.sqldelight.driver.jdbc.sqlite.JdbcSqliteDriver
import java.io.File

actual class DatabaseFactory(
    private val dbFile: File? = null
) {
    actual fun createDriver(): SqlDriver {
        val url = if (dbFile != null) {
            dbFile.parentFile?.mkdirs()
            "jdbc:sqlite:${dbFile.absolutePath}"
        } else {
            JdbcSqliteDriver.IN_MEMORY
        }
        return JdbcSqliteDriver(url)
    }

    actual fun createInMemoryDriver(): SqlDriver {
        return JdbcSqliteDriver(JdbcSqliteDriver.IN_MEMORY)
    }

    actual fun createDatabase(): UnboundDatabase {
        return initDatabase(createDriver())
    }

    actual fun createInMemoryDatabase(): UnboundDatabase {
        return initDatabase(createInMemoryDriver())
    }
}
