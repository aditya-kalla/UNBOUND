package com.unbound.data.repo.impl

import com.unbound.data.db.UnboundDatabase
import com.unbound.data.model.EventType
import com.unbound.data.model.PlaybackEventDraft
import com.unbound.data.repo.TelemetryWriter
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import kotlinx.coroutines.selects.onTimeout
import kotlinx.coroutines.selects.select

class SqlDelightTelemetryWriter(
    private val db: UnboundDatabase,
    scope: CoroutineScope,
    dispatcher: CoroutineDispatcher = Dispatchers.Default,
) : TelemetryWriter {

    private sealed interface Command {
        data class Event(val draft: PlaybackEventDraft) : Command
        data class Flush(val ack: CompletableDeferred<Unit>) : Command
    }

    private val channel = Channel<Command>(Channel.UNLIMITED)

    init {
        scope.launch(dispatcher) {
            val buffer = mutableListOf<PlaybackEventDraft>()
            var lastFlushTime = System.currentTimeMillis()

            while (isActive) {
                val timeoutMs = if (buffer.isNotEmpty()) {
                    val elapsed = System.currentTimeMillis() - lastFlushTime
                    maxOf(0L, 2000L - elapsed)
                } else {
                    Long.MAX_VALUE
                }

                var channelClosed = false
                val cmd = if (timeoutMs == Long.MAX_VALUE) {
                    val result = channel.receiveCatching()
                    if (result.isClosed) {
                        channelClosed = true
                        null
                    } else {
                        result.getOrNull()
                    }
                } else {
                    @OptIn(ExperimentalCoroutinesApi::class)
                    select<Command?> {
                        channel.onReceiveCatching { result ->
                            if (result.isClosed) {
                                channelClosed = true
                                null
                            } else {
                                result.getOrNull()
                            }
                        }
                        onTimeout(timeoutMs) {
                            null
                        }
                    }
                }

                if (channelClosed && cmd == null) {
                    break
                }

                when (cmd) {
                    is Command.Event -> {
                        buffer.add(cmd.draft)
                        val isEnd = cmd.draft.type == EventType.END
                        val isFull = buffer.size >= 50
                        if (isEnd || isFull) {
                            writeBuffer(buffer)
                            lastFlushTime = System.currentTimeMillis()
                        }
                    }
                    is Command.Flush -> {
                        if (buffer.isNotEmpty()) {
                            writeBuffer(buffer)
                            lastFlushTime = System.currentTimeMillis()
                        }
                        cmd.ack.complete(Unit)
                    }
                    null -> {
                        if (buffer.isNotEmpty()) {
                            writeBuffer(buffer)
                            lastFlushTime = System.currentTimeMillis()
                        }
                    }
                }
            }

            if (buffer.isNotEmpty()) {
                writeBuffer(buffer)
            }
        }
    }

    override fun submit(event: PlaybackEventDraft) {
        channel.trySend(Command.Event(event))
    }

    override suspend fun flush() {
        val ack = CompletableDeferred<Unit>()
        channel.send(Command.Flush(ack))
        ack.await()
    }

    private fun writeBuffer(buffer: MutableList<PlaybackEventDraft>) {
        if (buffer.isEmpty()) return
        val batch = buffer.toList()
        buffer.clear()
        db.transaction {
            for (draft in batch) {
                db.eventsQueries.insertEvent(
                    ts = draft.ts,
                    tzOffsetMin = draft.tzOffsetMin.toLong(),
                    eventType = draft.type.name,
                    listenId = draft.listenId,
                    trackId = draft.trackId,
                    durationMs = draft.durationMs,
                    positionMs = draft.positionMs,
                    seekToMs = draft.seekToMs,
                    listenedMs = draft.listenedMs,
                    startReason = draft.startReason?.name,
                    endReason = draft.endReason?.name,
                    contextType = draft.context.type?.name,
                    contextId = draft.context.id,
                    outputDeviceType = draft.device.type.name,
                    outputDeviceName = draft.device.name,
                    shuffle = if (draft.shuffle) 1L else 0L,
                    repeatMode = draft.repeatMode.name,
                    speed = draft.speed.toDouble(),
                )
            }
        }
    }
}
