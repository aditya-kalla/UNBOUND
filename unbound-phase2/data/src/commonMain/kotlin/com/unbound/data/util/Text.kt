package com.unbound.data.util

private val ARTICLES = listOf("the ", "a ", "an ")

/** Collapses whitespace + lowercases: the identity key for artists/albums ("ABBA" == "Abba"). */
fun matchKey(s: String): String = s.trim().lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")

/** Sort key: matchKey without a leading article, so "The Beatles" files under B. */
fun sortKey(s: String): String {
    val k = matchKey(s)
    for (a in ARTICLES) if (k.startsWith(a) && k.length > a.length) return k.removePrefix(a)
    return k
}

/**
 * Splits a single "Artist" tag into individual artists. Deliberately CONSERVATIVE: it only splits
 * on ';' and on "feat./ft./featuring". It never splits on '/', '&' or ',' because real names contain
 * them (AC/DC, Simon & Garfunkel, Tyler, The Creator). Multi-value tags (ID3v2.4 / Vorbis) arrive
 * from the reader already as a list and bypass this.
 */
object ArtistSplitter {
    private val featRegex = Regex("\\s+(?:feat\\.?|ft\\.?|featuring)\\s+", RegexOption.IGNORE_CASE)

    /** @return pair(main artists, featured artists) */
    fun split(raw: String): Pair<List<String>, List<String>> {
        val featParts = featRegex.split(raw.trim(), limit = 2)
        val mains = featParts[0].split(';').map { it.trim() }.filter { it.isNotEmpty() }
        val featured = if (featParts.size > 1) featParts[1].split(';').map { it.trim().trimEnd(')', ']') }.filter { it.isNotEmpty() } else emptyList()
        return mains to featured
    }
}
