# UNBOUND Phase 2 - Data & Telemetry Layer

## What this phase delivers
The local library (tracks/albums/artists/artwork, incremental scanning with move detection), playlists (nested folders, pinning, liked songs, search history), the persistent queue + player state, the append-only **Sonic Journal ledger** with ready-made analytics queries, full-text search, and reserved tables for Phase 4 (embeddings, stems).

## Decisions (and why) - veto any of these

**D1. SQLite (via SQLDelight) is the single database. DuckDB is deferred behind `StatsRepository`.**
This deviates from the blueprint, deliberately:
- DuckDB's own docs treat [Android as a build-it-yourself target](https://duckdb.org/docs/current/dev/building/android.html); there is no maintained Kotlin Multiplatform driver, and iOS would need its own native build. That is weeks of native-build work before a single feature exists.
- Scale: a heavy listener produces roughly 10k-150k event rows per year. Every Sonic Journal query is a GROUP BY over that, and SQLite answers them in milliseconds (all of `Stats.sq` is validated, see below). DuckDB's strength - columnar scans over hundreds of millions of rows - never applies on one person's phone.
- Playlists, queue and likes are small transactional writes, which is SQLite's strength, not DuckDB's.
- The door stays open: the UI only talks to `StatsRepository`. If analytics ever outgrow SQLite, add a DuckDB implementation on desktop, or export `playback_event` to Parquet, without touching the UI.

**D2. One bundled SQLite everywhere.** The SQLite that ships inside old Android versions is old and varies by device (window functions and FTS5 are not guaranteed). Bundle one modern SQLite on every platform (>= 3.38) so queries behave identically. Prompt 1 task B verifies this on desktop and Android.

**D3. Stable track identity.** Tracks are *soft-deleted* (`is_missing`), never dropped by a scan. A moved or renamed file is recognised by its `fingerprint` (size + hash of first/last 64 KiB) and **relinked**, so likes, playlists and listening history survive. Safety nets: an empty listing of a source that had files is treated as "drive unavailable", never "user deleted everything"; `meta_version` forces a rescan when the tag reader improves.

**D4. Tags are read once, in Rust (`lofty`), for all four platforms.** Platform code only *lists* files (directory walk / MediaStore / SAF / iOS media library). One reader = identical results everywhere. Artwork is extracted, thumbnailed and deduplicated by hash (one cover per album, not one per track).

**D5. The ledger stores facts, not interpretations.** `playback_event` is append-only: START and END rows (plus SEEK) sharing a `listen_id`. END carries `listened_ms` = media time actually played (pauses and seeks excluded, speed-safe). What counts as a "play" (>= 30 s or half the track) or a "skip" (<30 s ended by next/prev) lives in the `listen` *view*, so thresholds can change later without rewriting history. A killed app leaves a START with no END; the view keeps it as an uncounted listen. Every row stores the device UTC offset at that moment, so "your 2 AM tracks" stays right across travel and DST.

**D6. Gapped integer ordering** (1024 apart) for playlists and the queue: drag-reorder is one row update. When a gap runs out the repository renumbers inside one transaction. (I originally wrote renumbering as a single self-referencing UPDATE; testing showed SQLite can read rows it already changed, so it was replaced - good reason to keep tests close to the SQL.)

**D7. Search = FTS5** (prefix + diacritic-insensitive: "beyonce" finds "Beyonce"), with an automatic LIKE fallback if a platform's SQLite lacks FTS5.

**D8. Reserved tables** `track_embedding` (Vibe Search / Neural Autoplay) and `stem_cache` (Acoustic X-Ray) exist from v1 so Phase 4 never needs a risky migration on a big user database. Vectors are BLOBs loaded into memory for cosine search (brute force over 50k tracks takes milliseconds).

**D9. The user owns the data.** Everything is on-device; `deleteAllEvents`, `deleteEventsBefore` and `exportEvents` exist for a Settings > Privacy screen.

**D10. Migrations.** Schema v1 is final-by-discipline: every later change is a numbered `.sqm` migration with `verifyMigrations` on. Never edit a shipped table in place.

## Modules
```
:shared  (Phase 1)  audio engine bridge            :data  (Phase 2)  database + repositories + scanner + tracker
rust-audio-engine   cpal/symphonia/rubato          data/src/commonMain/sqldelight/...  Library, Playlist, Queue, Events, Stats, Embedding
                    + meta_read_json (Prompt 2)    data/src/commonMain/kotlin/...      model, util, telemetry, scan, repo (interfaces)
```

## Data flows
```
SCAN   MediaFileSource.list -> computeScanDiff(known, found) -> read tags (Rust, 4 in parallel)
       -> fingerprint matches a vanished track? relink : upsert -> one transaction per 200 files
       -> markMissing(rest) -> prune empty albums/artists/art -> stamp last_scan_at

PLAY   engine position/state ->  ListenTracker  -> PlaybackEventDraft -> TelemetryWriter (batched, off-thread)
                                                                      -> playback_event -> listen view -> StatsRepository
```

## Baseline-feature coverage in this phase
Liked Songs, playlists (nested folders, pinning, reorder), recently played, search + search history, persistent queue (play-next vs context, shuffle that restores original order), listening stats (top tracks/artists/genres, hour/weekday/daily, skip rate, hyper-fixation loops, device breakdown, circadian "2 AM" windows), album/artist/genre browsing, ReplayGain fields for normalization, lyrics sidecar/embedded flags, playlist import from CSV (Exportify format, matched against your *local* files).
**Not in the legal build:** the batch download manager (Community flavor only) and Spotify OAuth import (CSV import covers migration without any Spotify API).

## Known gaps and risks
1. **The engine has no end-of-track signal.** The queue manager can't auto-advance without it. Prompt 2 adds `engine_state` / `engine_duration_ms` (ABI v3).
2. **`panic = "abort"` in the Rust release profile** means a malformed tag could kill the whole app. Prompt 2 switches to unwind + `catch_unwind` on every FFI function.
3. **Android file access (scoped storage):** native code can't open arbitrary content URIs by path. A file-descriptor variant of the tag reader is a Phase 3 item; desktop and MediaStore paths work now.
4. **SQLDelight + FTS5:** `CREATE VIRTUAL TABLE ... fts5` must be accepted by SQLDelight 2.0.2 and by Android's SQLite. Prompt 1 task B/C has explicit fallbacks.
5. **Event volume:** unbounded growth is fine for years, but Settings should offer "keep last N months".

## Verified so far (by me, in this session)
- Whole schema + all 80 SQL statements run on real SQLite 3.45 with FTS5; scenario assertions pass (counted/skipped/orphan listens, 2 AM window including midnight wrap, listener-local dates, upsert-on-conflict, relink, FTS diacritics+prefix). Re-run any time: `python tools/validate_sql.py data/src/commonMain/sqldelight/com/unbound/data/db`.
- Pure-Kotlin core (`ListenTracker`, `computeScanDiff`, text/position utils) compiled with kotlinc 2.0.21: **15/15 tests pass.**
- **Not verified:** SQLDelight code generation, the Gradle build, anything using coroutines (`LibraryScanner`, repositories), Android. Those are exactly what the prompts assign.

## Phase 2 is done when
Prompt 1 and 2 reports show: module builds on desktop; all tests green (including the stats-parity scenario); 50k-track synthetic library: unchanged rescan <= 3 s, first list page < 50 ms, FTS search < 50 ms; a moved file keeps its likes/playlists/history; the engine reports end-of-track; a corrupt file returns an error instead of crashing.
