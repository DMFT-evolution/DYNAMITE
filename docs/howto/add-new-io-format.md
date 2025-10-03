# How-to: Add a New IO Format

1) Extend `src/io/` with a writer/reader for the new format.
2) Integrate with existing async export flow (respect `--async-export`).
3) Add CLI flag(s) to select the format if needed.
4) Update `usage.md` and `reading-outputs.md`.
5) Provide a small example file and reader snippet for users.
