# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/gpu.svg" alt="GPU icon"/> Tutorial: Building for a cluster

On a cluster the program is often compiled on a login node and run on compute nodes with different CPUs or GPUs. Two settings matter: the host code must not be optimized for the CPU of the login node, and the GPU code must be compiled for the GPUs of the compute nodes.

## Build

```bash
cmake -S . -B build -DDMFE_PORTABLE_BUILD=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j "$(getconf _NPROCESSORS_ONLN)"
```

`DMFE_PORTABLE_BUILD=ON` switches off `-march=native` and uses portable link settings. `CMAKE_CUDA_ARCHITECTURES` lists the compute capabilities of the GPUs you will use, for example `80` for A100 or `90` for H100; several values can be given separated by semicolons. On nodes without GPUs, configure with `-DDMFE_WITH_CUDA=OFF`.

The CUDA runtime is linked statically by default, so the compute nodes only need the NVIDIA driver, not the CUDA toolkit.

## Run

A batch script for one GPU might look like

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd /path/to/DYNAMITE              # Grid_data/ is read from the working directory
./RG-Evo -p 3 -q 4 -l 0.5 -T inf -G 0 -L 512 -e 1e-10 -w 1 \
         -t 1e6 -m 1e8 -D false -o /scratch/$USER/Results/
```

- `-D false` avoids per-step output and the extra synchronization of debug mode.
- `-w 1` fixes the number of sparsification sweeps, so that the result does not depend on how much GPU memory other jobs use.
- Choose `-o` on a file system with enough space: checkpoints of long runs reach several GB.
- If the job ends because of the time limit, submitting the same command again continues the run from the last checkpoint (see [Continuing a run](../usage.md#continuing-a-run)). Checkpoints are written every `--sparsify-interval` steps; a smaller value loses less work when a job is killed.

For CPU nodes, set the number of threads to the number of cores allocated to the job, e.g. `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
