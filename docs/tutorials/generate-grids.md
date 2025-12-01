# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Tutorial: Generate Grids

Goal: create a ready-to-use interpolation grid folder `Grid_data/<L>/` (paper defaults) for a first simulation.

## Quick command

```bash
./RG-Evo grid --len 512 --Tmax 100000 --dir 512 \
  --interp-method poly --interp-order 9 --spline-order 5
```

Outputs (`Grid_data/512/`):

- `theta.dat`, `phi1.dat`, `phi2.dat` (grids)
- `int.dat` (quadrature weights)
- `posA*`, `posB*` (stencil positions)
- `inds*`, `weights*` (interpolation metadata)
- `grid_params.txt` (provenance & CLI)

Point runs at the folder via `-L 512` (length) and ensure the directory exists.

## Optional tweaks

- Larger grid: use `--len 1024` (or `2048`).
- Rational interpolation: `--interp-method rational --fh-stencil 15` for extra stability.
- Index remapping: `--alpha 0.2 --delta 0.25` (recorded; default 0 keeps paper grid).
- Validate without writing: append `--validate` to compare with existing files.

## Troubleshooting

- Missing folder: rerun the quick command above.
- Mismatch after code updates: regenerate with `--validate` then rebuild.
- Disk space: remove unused large folders (keep only the L values you need).

## See also

- How-to → Generate new grids (full flag reference & theory)
- Concepts → Interpolation grids (mathematical background)
- Tutorial → First Run (integrating the grid in a simulation)

