package com.unbound.data.util

/**
 * Gapped integer ordering for playlists and the queue. Items sit 1024 apart, so inserting or
 * dragging an item between two neighbours is a single-row update. When a gap is used up
 * [between] returns null and the repository must renumber (read ids in order, rewrite
 * 1024, 2048, ... inside one transaction).
 */
object PositionAllocator {
    const val GAP = 1024L

    fun append(maxPosition: Long?): Long = (maxPosition ?: 0L) + GAP

    /** Position strictly between [prev] and [next]; either may be null (list start/end). Null = renumber needed. */
    fun between(prev: Long?, next: Long?): Long? = when {
        prev == null && next == null -> GAP
        prev == null -> next!! - GAP
        next == null -> prev + GAP
        next - prev >= 2 -> prev + (next - prev) / 2
        else -> null
    }
}
