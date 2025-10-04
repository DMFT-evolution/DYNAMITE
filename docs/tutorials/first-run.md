# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/file.svg" alt="Tutorial icon"/> Tutorial: First Run

Goal: build and run DMFE in Release on L=512 grid and inspect outputs.

1) Build:
```bash
./build.sh
```
2) Run a short simulation:
```bash
./RG-Evo -m 20000 -L 512 -l 0.5 -D false
```

3) Inspect outputs:

- Explore the output directory printed at start
- Open `params.txt` for parameters and environment
- If HDF5 was available, inspect `data.h5` (e.g., with h5ls/h5dump)

## Troubleshooting

- Missing `Grid_data/512`: pick another `-L` with existing data (512/1024/2048)
- GPU errors: retry with `--gpu false`
