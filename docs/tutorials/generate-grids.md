# Tutorial: Generate grids quickly

This short tutorial shows how to create a ready-to-use grid folder under `Grid_data/<L>/` for a first run.

## Quick command

```bash
./RG-Evo grid --len 512 --Tmax 100000 --dir 512 \
  --spline-order 5 --interp-method poly --interp-order 9
```

This produces:
- `theta.dat`, `phi1.dat`, `phi2.dat` (grids)
- `int.dat` (B-spline quadrature, degree 5 by default)
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat`
- `inds*.dat` and `weights*.dat` (interpolation metadata)

## Tips
- Use `--len 1024` or `2048` for larger grids.
- `--interp-method rational --interp-order 9 --fh-stencil 14` can improve stability on irregular nodes.
- Use `--validate` to compare newly computed grids with an existing folder without writing files.

For more detail, see How-to → Generate new grids.
