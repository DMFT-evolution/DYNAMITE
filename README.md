# DMFE

CUDA / C++ project.

## Build

From the project root (this directory):

```bash
./build.sh
```

Or manually:

```bash
cmake -S . -B build
cmake --build build -j $(nproc)
```

If you previously ran cmake from a parent directory accidentally (e.g. from $HOME), remove the stray build directory you created and re-run from here.

## Run

After build the executable is placed in the project root:

```bash
./StreamCluster
```

## Options

HDF5 detection is optional. Install dev packages (e.g. `libhdf5-dev`) before configuring if needed.

Set a specific build type:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## Clean build

```bash
./build.sh build --clean
```
