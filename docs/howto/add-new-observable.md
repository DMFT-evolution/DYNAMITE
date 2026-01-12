# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/function.svg" alt="Observable icon"/> How-to: Add a New Observable

This guide is for developers who want to add a new derived quantity to the run outputs (HDF5/binary + optional text summaries).

## Pick the “kind” of observable

Most additions fall into one of these lanes:

- **Saved state** (must be available for resume): add it to the serialized snapshot/state.
- **Derived diagnostic** (not needed for resume): compute it at save time and write it to text and/or HDF5.
- **Per-step telemetry**: log it in debug mode (`-D true`) or append it to `step_metrics.txt`.

If you’re unsure: start as a *derived diagnostic*; promote it to saved state only if you need exact restart reproducibility.

## Where to add data

### 1) Simulation state / snapshot (resume-safe)

Files to inspect first:

- `include/simulation/simulation_data.hpp` — core in-memory state.
- `src/io/io_snapshot_host.cpp` — builds a `SimulationDataSnapshot` for the writers.
- `include/io/io_utils.hpp` — save/load function declarations.
- `src/io/io_save_binary.cpp` and `src/io/io_input.cpp` — binary save/load.
- `src/io/io_save_hdf5.cpp` — HDF5 save (compile-time or runtime-optional).

Steps:

1) Add a new field to the snapshot/state struct(s) (prefer plain `std::vector<double>` / scalars).
2) Populate it in `createDataSnapshot()`.
3) Serialize it:
	- binary: extend the writer and reader in lockstep.
	- HDF5: add a dataset with the same shape as the corresponding binary payload.
4) Update any versioning/compat checks if the on-disk format changes.

### 2) Derived diagnostics (no format changes)

If the observable can be computed from existing fields at save time, avoid changing file formats:

1) Compute it in the save path (e.g. alongside other summaries in `src/io/`).
2) Write it to a new text file (e.g. `my_observable.txt`) and/or an HDF5 dataset.
3) Mention it in `docs/usage.md` and `docs/tutorials/reading-outputs.md`.

This path is usually the lowest friction.

## Where to compute it

Common places:

- **During integration:** `src/EOMs/time_steps.cpp` and `src/EOMs/runge_kutta.cpp` (if it must be accumulated as the solver steps).
- **At save time:** the snapshot + I/O modules under `src/io/` (good for derived diagnostics).

If it requires GPU state, prefer computing from the snapshot (host) to keep the I/O code simple; the CUDA path already copies the required vectors when saving.

## Add a text output (recommended)

Even if you also add HDF5, a small text summary is convenient for quick plotting.

- Put it next to other summaries in `src/io/`.
- Keep format simple: whitespace-separated columns with a short header line.

## Docs + sanity checks

1) Update `docs/usage.md` (Outputs section) to list the new file/dataset.
2) Add one quick check to `docs/tutorials/reading-outputs.md` (e.g. expected sign/limits).
3) Run a tiny simulation and verify:
	- output file is created
	- values are finite
	- (if resume-safe) restart reproduces the observable
