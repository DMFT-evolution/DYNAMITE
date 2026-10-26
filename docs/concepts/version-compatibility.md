# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/tag.svg" alt="Version tag icon"/> Version compatibility

A run is continued from its checkpoint only if the program version that wrote it is compatible with the current one (see [Continuing a run](../usage.md#continuing-a-run)). The version is taken from the `code_version` entry of the run's `params.txt` and compared with that of the running program (`./RG-Evo -v`):

| Case | Result |
|---|---|
| same version | continued |
| same first two version numbers (e.g. `1.4.0` and `1.4.2`) | continued; a warning is printed if the Git commits differ |
| no version recorded | continued with a warning |
| different first two version numbers | not continued and a new run directory is created; continued with a warning if `-I true` is given |

`./RG-Evo -c path/to/params.txt` reports the result of this comparison without starting a run.

If you continue a run with `-I true`, or with a different commit of the same version, note the commit in your records and check that the observables continue smoothly across the restart.

The version, the Git commit and branch, and whether the source had uncommitted changes are recorded when the program is built (`cmake/dmfe_version.cmake`). The version is the output of `git describe --tags --always --dirty`. A build from a source tree that is not a Git checkout, e.g. a release archive, reports the version set by the CMake option `DMFE_FALLBACK_VERSION` and the commit `unknown`.

The comparison is implemented in `src/version/version_compat.cpp`.
