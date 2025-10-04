# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/function.svg" alt="Observable icon"/> How-to: Add a New Observable

1) Define data structure in `include/simulation/simulation_data.hpp` (or suitable module).
2) Compute it in the time step (SERK2/RK) or post-step hook.
3) Export: extend IO (`src/io/io_output.*`) to write it to HDF5/bin and text.
4) Document: update `docs/usage.md` and list in outputs.
5) Test: add a short run and check values/logs.
