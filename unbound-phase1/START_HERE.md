# UNBOUND Phase 2 - how to use this zip

1. Unzip somewhere temporary (not over your project).
2. Copy the whole `data/` folder and `tools/` folder into `...\unbound-phase1\` (the project root). Don't overwrite your existing Gradle files - Antigravity edits those (step 4).
3. (Optional sanity check, needs Python) from the project root:
   `python tools\validate_sql.py data\src\commonMain\sqldelight\com\unbound\data\db`  -> should end with `ALL CHECKS PASSED`.
4. Open `ANTIGRAVITY_PROMPT_1_DATA_MODULE.md`, paste everything below the line into Antigravity. Wait for its report.
5. Then do the same with `ANTIGRAVITY_PROMPT_2_RUST_METADATA.md` (it needs Prompt 1 merged first).
6. Bring me both reports. I review them in one pass, then we start the design phase (references) or Phase 3 - your call.

Read `PHASE2_ARCHITECTURE.md` first if you want the reasoning (it explains why the database is SQLite and not DuckDB - that is the one decision where I deviated from your blueprint, and you can veto it).
