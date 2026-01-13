# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/file.svg" alt="IO icon"/> How-to: Add a New IO Format

DMFE currently supports:

- **HDF5** (`data.h5`) when HDF5 is available (compile-time or runtime-optional)
- **Binary fallback** (`data.bin`) always available
- **Text summaries** for quick plotting

If you want to add a new output format (e.g. NPZ, Zarr, CSV bundles), it’s best to follow the existing “modular writers + driver” split.

## 1) Identify whether it’s a replacement or an additional export

- **Replacement**: only one “main state” file is written (like HDF5 vs binary today).
- **Additional export**: keep writing `data.h5`/`data.bin` for resume, but also write an *extra* file for convenience.

For most formats (NPZ/CSV), *additional export* is safer because it doesn’t affect resume compatibility.

## 2) Add the writer module

Create a focused pair under `src/io/` + `include/io/`.

Useful reference modules:

- `src/io/io_save_driver.cpp` — orchestrates the save and async behavior.
- `src/io/io_save_binary.cpp` — baseline “always works” format.
- `src/io/io_save_hdf5.cpp` — HDF5 writer (runtime-optional where enabled).
- `src/io/io_save_params.cpp` — `params.txt` provenance.

Prefer taking a `SimulationDataSnapshot` as input (so writers are decoupled from the live simulation object).

## 3) Hook into the save driver

Integrate into the existing async flow:

1. Keep respecting `--async-export`.
2. Ensure failures fall back cleanly (either to an existing format or to “skip extra export”).
3. Update progress reporting only if it’s user-visible; otherwise keep it silent.

## 4) If you need a new CLI switch

- Add a flag in the CLI parser (see the config/CLI source listed in `Overview.md`).
- Store it in `SimulationConfig` and include it in `params.txt`.

## 5) Document and provide a tiny reader example

1. Update `docs/usage.md` (Outputs section).
2. Update `docs/tutorials/reading-outputs.md` with:
	- how to detect the format
	- minimal Python snippet (or point to a script in `scripts/`)

## Compatibility note

If the new format replaces the resume file, you must update both save and load paths (`src/io/io_input.cpp`) and consider version-compat behavior.
