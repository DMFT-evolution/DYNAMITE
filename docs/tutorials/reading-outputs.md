# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/file.svg" alt="File icon"/> Tutorial: Reading Outputs

DMFE writes either HDF5 (`data.h5`) or binary (`data.bin`) plus text summaries.

- HDF5 datasets: `QKv`, `QRv`, `dQKv`, `dQRv`, `t1grid`, `rvec`, `drvec`
- Attributes: parameters, time, iteration, version info

Inspect with common tools:

```bash
h5ls -r data.h5
h5dump -n data.h5
```

## Binary fallback

- See `include/io/io_utils.hpp` for formats; use provided helper readers (planned) or convert with a small script.
