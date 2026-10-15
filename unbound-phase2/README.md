# UNBOUND — Phase 2: Data & Telemetry Layer

This module implements the local music library database, track metadata scanner, playback telemetry ledger (Sonic Journal), persistent player queue, playlists, and full-text search (FTS5) using SQLite and Cash App SQLDelight.

## Architecture & Documentation
- `PHASE2_ARCHITECTURE.md`: Detailed architectural decisions, data flows, and ledger design.
- `START_HERE.md`: Quick-start instructions for Phase 2.
- `ANTIGRAVITY_PROMPT_1_DATA_MODULE.md`: Specification for SQLite schema & SQLDelight repositories.
- `ANTIGRAVITY_PROMPT_2_RUST_METADATA.md`: Specification for native metadata reader & FFI bindings.

## Directory Layout
- `data/`: Kotlin Multiplatform module (Android + Desktop JVM).
  - `src/commonMain/sqldelight/com/unbound/data/db/`: SQLDelight queries (`Library.sq`, `Playlist.sq`, `Queue.sq`, `Events.sq`, `Stats.sq`, `Embedding.sq`).
  - `src/commonMain/kotlin/`: Core interfaces, repositories, models, `ListenTracker`, `LibraryScanner`.
  - `src/jvmCommonMain/kotlin/`: Shared JVM/Desktop native bindings (`NativeMetadataReader`).
  - `src/desktopMain/kotlin/`: Desktop driver factory, recursive file walker, `ScanSmokeProgram`.
  - `src/desktopTest/kotlin/`: Desktop unit tests, benchmarks, and SQLite parity suites.
- `tools/`: Validation tooling (`validate_sql.py`).
- `unbound_audio.dll`: Native Windows binary providing tag extraction via `lofty` and Blake3 fingerprinting.

## Building & Testing

```powershell
# Run the complete test suite
.\gradlew.bat desktopTest

# Run the SQL validation script
python tools\validate_sql.py data\src\commonMain\sqldelight\com\unbound\data\db

# Run the real folder scanner smoke test
.\gradlew.bat :data:scanSmoke -Dunbound.scanFolder="C:\Users\lenovo\Music"
```
