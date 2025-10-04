# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/architecture.svg" alt="Architecture icon"/> Architecture

The code mirrors physics operations:

- EOMs (`include/EOMs/`): RK54 (adaptive) with auto-switch to SSPRK104; optional SERK2 trials post-sparsification.
- Interpolation (`include/interpolation/`): sparse 2D interpolation kernels and index maps.
- Convolution (`include/convolution/`): memory-kernel evaluations.
- Sparsification (`include/sparsify/`): maintains sublinear memory/compute with error control.
- Simulation (`include/simulation/`): orchestration, checkpoints, and async I/O.
- Core/Math/IO/Version: utilities, numeric primitives, I/O (HDF5 runtime), version policy.

Accuracy knobs:

- Grid length L (512/1024/2048): convergence in observables vs. runtime.
- Integrator tolerance `-e`, min step `-d`: trade accuracy vs. cost.
- Sparsification mode: aggressive vs. conservative.

GPU/CPU paths are interchangeable; ensure convergence checks are done on your platform of choice.
