package com.unbound.data.repo

import com.unbound.data.testTrack
import com.unbound.data.db.DatabaseFactory
import com.unbound.data.db.initDatabase
import com.unbound.data.model.FoundFile
import com.unbound.data.model.ParsedTrack
import com.unbound.data.repo.impl.SqlDelightLibraryRepository
import com.unbound.data.repo.impl.SqlDelightLibraryWriter
import com.unbound.data.scan.ParsedFile
import com.unbound.data.scan.ScanBatch
import kotlinx.coroutines.runBlocking
import kotlin.test.*

class FtsTest {

    @Test
    fun testFtsDiacriticsPrefixAndSpecialCharacters() = runBlocking {
        val factory = DatabaseFactory()
        val driver = factory.createInMemoryDriver()
        val db = initDatabase(driver)
        val writer = SqlDelightLibraryWriter(db, driver)
        val libraryRepo = SqlDelightLibraryRepository(db, driver)

        db.libraryQueries.insertSource("folder:/music", "FOLDER", "Music")
        val sourceId = db.playlistQueries.lastInsertedId().executeAsOne()

        val track1 = testTrack(
            title = "Café del Mar",
            artists = listOf("Beyoncé"),
            album = "Renâissance",
            genres = listOf("Electronic", "Chillout"),
            durationMs = 210_000L,
        )
        val track2 = testTrack(
            title = "Bohemian Rhapsody",
            artists = listOf("Queen"),
            album = "A Night at the Opera",
            genres = listOf("Rock"),
            durationMs = 354_000L,
        )

        writer.applyBatch(
            sourceId,
            ScanBatch(
                upserts = listOf(
                    ParsedFile(FoundFile("file:///music/cafe.mp3", 1000L, 1000L), track1),
                    ParsedFile(FoundFile("file:///music/queen.mp3", 2000L, 1000L), track2),
                ),
                relinks = emptyList(),
            )
        )

        // 1. Diacritics matching (unicode61 remove_diacritics 2)
        val cafeResult = libraryRepo.search("cafe")
        assertEquals(1, cafeResult.size)
        assertEquals("Café del Mar", cafeResult[0].title)

        // 2. Prefix matching (bey -> Beyoncé)
        val beyResult = libraryRepo.search("bey")
        assertEquals(1, beyResult.size)
        assertEquals("Café del Mar", beyResult[0].title)

        val renaisResult = libraryRepo.search("renais")
        assertEquals(1, renaisResult.size)
        assertEquals("Café del Mar", renaisResult[0].title)

        // 3. Special characters must not crash
        val quoteOnly = libraryRepo.search("\"")
        assertNotNull(quoteOnly)

        val asteriskOnly = libraryRepo.search("*")
        assertNotNull(asteriskOnly)

        val quotePhrase = libraryRepo.search("\"Café\"")
        assertEquals(1, quotePhrase.size)

        val ftsKeywords = libraryRepo.search("AND OR NOT NEAR")
        assertNotNull(ftsKeywords)

        val parens = libraryRepo.search("(()))((***\"\"\"")
        assertNotNull(parens)

        // 4. Empty and whitespace queries return empty list
        assertTrue(libraryRepo.search("").isEmpty())
        assertTrue(libraryRepo.search("   ").isEmpty())
    }
}
