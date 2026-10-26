# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Development icon"/> Development

Contributions are welcome. For larger changes, please open an issue on [GitHub](https://github.com/DMFT-evolution/DYNAMITE/issues) first, so that we can discuss them. Pull requests should be focused on one change, state how it was tested, and update the documentation where the behaviour visible to users changes.

The layout of the source is described in [Code structure and file formats](../concepts/architecture.md); changing the model is discussed in [Adapting the code to another model](../concepts/other-models.md).

## Conventions

- C++17; CUDA code only in `.cu` files, guarded by `DMFE_WITH_CUDA` where it is referenced from host code.
- A function with a CPU and a GPU implementation keeps the same interface in both, and both are changed together.
- Every new command-line option is listed in the help text (`src/core/config.cpp`), written to `params.txt` (`src/io/io_save_params.cpp`), and documented in [Usage](../usage.md). If it changes the results, it is also compared when a run is continued (`checkParametersMatch` in `src/io/io_input.cpp`).
- Changes to the numerics are checked against the regression tests below, and on both backends.

## Regression tests

`tests/run_regression.py` runs short simulations and compares them with exact results:

| Test | Check |
|---|---|
| T1 | pure $p=2$ model, quench from $T_0=\infty$ to $T=0$, up to $t=1$: $C(t,0)$ and $E(t)$ against the exact solution (tolerance $10^{-6}$), and $E(t)$ above $E_{\rm th}$ |
| T3 | the same model with sparsification switched off and with two sweeps per pass: the errors with respect to the exact solution must agree to $10^{-7}$ |
| T5 | mixed 3+4 model started in equilibrium at $T_0=T=2$, up to $t=1$: the dynamics is stationary, so $E(t)=-1/T$ and $\mu(t)=T+f'(1)/T$ must hold at all times (tolerance $10^{-7}$). This checks the forcing by the initial state, the thermal noise and a covariance with two orders, which T1 does not. |
| T4 | CPU against GPU (only with `--gpu`; skipped if no GPU is available) |

```bash
pip install mpmath
python3 tests/run_regression.py --exe ./RG-Evo            # CPU tests
python3 tests/run_regression.py --exe ./RG-Evo --gpu      # including the CPU/GPU comparison
```

The exact solution is in `tests/exact_p2.py`. The tests run on the CPU and take several minutes. The exit status is 0 if all tests pass. The CI workflow `.github/workflows/build.yml` builds the CPU-only version and runs T1, T3 and T5 on every push and pull request (about four minutes).

## Documentation

The documentation is written in Markdown under `docs/` and built with MkDocs:

```bash
pip install mkdocs mkdocs-material mkdocs-exclude mkdocs-asciinema-player
mkdocs serve        # preview at http://127.0.0.1:8000
mkdocs build        # static site in site/
```

`docs/gen_api.sh` generates an API reference from the Doxygen comments (requires Doxygen and doxybook2, and `DMFE_WITH_API=1`); it is not part of the published site.

## Releases

1. Update `CHANGELOG.md`.
2. Check that the CPU and the CUDA build compile, the regression tests pass and `mkdocs build` succeeds.
3. Tag the release as `vX.Y.Z`. Runs can only be continued across versions with the same `X.Y` (see [Version compatibility](../concepts/version-compatibility.md)), so increase `Y` when the content of the checkpoint or the meaning of a parameter changes.
4. Update the version and date in `CITATION.cff`.
