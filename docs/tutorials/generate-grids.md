# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Tutorial: Generate Grids

**Audience:** new users who want a working grid folder for their first run.

Goal: create a ready-to-use interpolation grid folder `Grid_data/<L>/` (paper defaults) for a first simulation.

## Quick command

```bash
./RG-Evo grid --len 512 --Tmax 100000 --dir 512 \
  --interp-method index-poly --interp-order 9 --spline-order 5
```

Outputs (`Grid_data/512/`):

- `theta.dat`, `phi1.dat`, `phi2.dat` (grids)
- `int.dat` (quadrature weights)
- `posA*`, `posB*` (stencil positions)
- `inds*`, `weights*` (interpolation metadata)
- `grid_params.txt` (provenance & CLI)

Point runs at the folder via `-L 512` (length) and ensure the directory exists.

## Common variations

- Larger grid: `--len 1024` (or `2048`).
- Reproducibility check: append `--validate` to compare with the existing folder.
- If you hit numerical corner cases on very irregular nodes: try `--interp-method rational --fh-stencil 15`.

## Troubleshooting

- Missing folder: rerun the quick command above.
- Mismatch after code updates: run with `--validate` to confirm, then regenerate.
- Disk space: remove unused `Grid_data/<L>/` folders (keep only the L values you use).

## See also

- [How-to → Generate new grids](../howto/generate-grids.md) (full flag reference, file formats, validation)
- [Concepts → Interpolation grids](../concepts/interpolation-grids.md) (mathematical background and paper equations)
- [Tutorial → First Run](first-run.md) (integrating the grid in a simulation)

