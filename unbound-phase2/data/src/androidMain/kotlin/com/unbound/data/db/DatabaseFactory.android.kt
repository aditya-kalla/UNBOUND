package com.unbound.data.db

import android.content.Context
import app.cash.sqldelight.db.SqlDriver
import app.cash.sqldelight.driver.android.AndroidSqliteDriver

actual class DatabaseFactory(
    private val context: Context? = null
) {
    actual fun createDriver(): SqlDriver {
        val ctx = checkNotNull(context) { "AndroidSqliteDriver requires a non-null Context" }
        return AndroidSqliteDriver(
            schema = UnboundDatabase.Schema,
            context = ctx,
            name = "unbound.db"
        )
    }

    actual fun createInMemoryDriver(): SqlDriver {
        val ctx = checkNotNull(context) { "AndroidSqliteDriver requires a non-null Context" }
        return AndroidSqliteDriver(
            schema = UnboundDatabase.Schema,
            context = ctx,
            name = null
        )
    }

    actual fun createDatabase(): UnboundDatabase {
        return initDatabase(createDriver())
    }

    actual fun createInMemoryDatabase(): UnboundDatabase {
        return initDatabase(createInMemoryDriver())
    }
}
