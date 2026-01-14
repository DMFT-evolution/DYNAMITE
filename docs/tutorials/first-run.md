# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/file.svg" alt="Tutorial icon"/> Tutorial: First Run

**Audience:** new users who want to build and run DYNAMITE once.

Goal: build and run DYNAMITE in Release on L=512 grid and inspect outputs.

1. Build:
```bash
./build.sh
```
2. Run a short simulation:
```bash
./RG-Evo -m 20000 -L 512 -l 0.5 -D false
```

3. Inspect outputs:

- Explore the output directory printed at start
- Open `params.txt` for parameters and environment
- If HDF5 was available, inspect `data.h5` (e.g., with h5ls/h5dump)

## Troubleshooting

- Missing `Grid_data/<L>`: generate it with the grid subcommand. Example for L=512:
	See:
	- [Tutorial → Generate grids](generate-grids.md) (quick start)
	- [How-to → Generate new grids](../howto/generate-grids.md) (full flag reference + validation)
- GPU errors: retry with `--gpu false`

## See also

- [Tutorial → Generate grids](generate-grids.md)
- [Tutorial → CPU-only run](cpu-only.md)
- [Tutorial → Reading outputs](reading-outputs.md)
