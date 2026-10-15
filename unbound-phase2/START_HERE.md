# UNBOUND Phase 2 - how to use this zip

1. This directory (`unbound-phase2`) contains the complete Data & Telemetry module.
2. The database layer is located in `data/` with schema in `data/src/commonMain/sqldelight/com/unbound/data/db`.
3. (Optional sanity check, needs Python) from this folder:
   `python tools\validate_sql.py data\src\commonMain\sqldelight\com\unbound\data\db`
4. Open `ANTIGRAVITY_PROMPT_1_DATA_MODULE.md`, paste everything below the line into Antigravity. Wait for its report.
5. Then do the same with `ANTIGRAVITY_PROMPT_2_RUST_METADATA.md` (it needs Prompt 1 merged first).
6. Bring me both reports. I review them in one pass, then we start the design phase (references) or Phase 3 - your call.

Read `PHASE2_ARCHITECTURE.md` first if you want the reasoning (it explains why the database is SQLite and not DuckDB - that is the one decision where I deviated from your blueprint, and you can veto it).
