package com.unbound.data

import com.unbound.data.util.ArtistSplitter
import com.unbound.data.util.PositionAllocator
import com.unbound.data.util.matchKey
import com.unbound.data.util.sortKey
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class TextAndPositionTest {
    @Test fun sortKeyStripsLeadingArticlesOnly() {
        assertEquals("beatles", sortKey("The Beatles"))
        assertEquals("a tribe called quest".removePrefix("a "), sortKey("A Tribe Called Quest"))
        assertEquals("the", sortKey("The"))                       // a title that IS the article is kept
        assertEquals("theory of a deadman", sortKey("Theory of a Deadman"))
    }

    @Test fun matchKeyCollapsesCaseAndWhitespace() {
        assertEquals("abba", matchKey("  ABBA "))
        assertEquals("simon & garfunkel", matchKey("Simon  &   Garfunkel"))
    }

    @Test fun artistSplitterIsConservative() {
        assertEquals(listOf("AC/DC") to emptyList(), ArtistSplitter.split("AC/DC"))
        assertEquals(listOf("Simon & Garfunkel") to emptyList(), ArtistSplitter.split("Simon & Garfunkel"))
        assertEquals(listOf("Tyler, The Creator") to emptyList(), ArtistSplitter.split("Tyler, The Creator"))
        assertEquals(listOf("A", "B") to emptyList(), ArtistSplitter.split("A; B"))
        assertEquals(listOf("Drake") to listOf("Rihanna"), ArtistSplitter.split("Drake feat. Rihanna"))
        assertEquals(listOf("Drake") to listOf("Rihanna", "Kanye"), ArtistSplitter.split("Drake ft. Rihanna; Kanye"))
    }

    @Test fun positionAllocator() {
        assertEquals(1024L, PositionAllocator.append(null))
        assertEquals(3072L, PositionAllocator.append(2048))
        assertEquals(1024L, PositionAllocator.between(null, null))
        assertEquals(0L, PositionAllocator.between(null, 1024))
        assertEquals(2048L + 1024, PositionAllocator.between(2048, null))
        assertEquals(1536L, PositionAllocator.between(1024, 2048))
        assertNull(PositionAllocator.between(1024, 1025))
        assertNull(PositionAllocator.between(5, 5))
    }
}
