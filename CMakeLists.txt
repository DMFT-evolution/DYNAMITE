cmake_minimum_required(VERSION 3.24)

# [CPC-REV] Must be set before any add_library()/add_executable() so that the
# generated compile_commands.json covers the whole program and not only the
# targets declared after it (referee 2, point 5: tooling/reproducibility).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional CUDA support with tri-state behavior (AUTO/ON/OFF)
option(DMFE_WITH_CUDA "Enable CUDA support" ON)

# Optional tail-fit debug output
option(DMFE_TAIL_FIT_DEBUG "Enable tail-fit debug output" OFF)

# Prefer clang++-14 as CUDA host compiler only if explicitly requested
option(DMFE_PREFER_CLANG_HOST "Prefer clang++ as CUDA host compiler for nvcc" OFF)

# Detect CUDA if AUTO or ON
if(DMFE_WITH_CUDA)
    # Check if CUDA compiler is available before enabling the language
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
    # nvcc 11.x + GCC >= 12 requires this flag even during compiler-id checks
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler" CACHE STRING "" FORCE)
        message(STATUS "CUDA compiler found: enabling CUDA support")
        # Prefer clang++-14 as nvcc host if available (reduces GCC12 interactions)
        if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
            find_program(CLANGPP_14 NAMES clang++-14)
            if(CLANGPP_14)
                set(CMAKE_CUDA_HOST_COMPILER ${CLANGPP_14})
                message(STATUS "Using ${CLANGPP_14} as CUDA host compiler")
            endif()
        endif()
        project(DYNAMITE LANGUAGES C CXX CUDA)
        set(DMFE_WITH_CUDA ON)
        # Now find the CUDA toolkit after CUDA language is enabled
        find_package(CUDAToolkit REQUIRED)
    else()
        message(WARNING "CUDA compiler not found; building CPU-only version")
        set(DMFE_WITH_CUDA OFF)
        project(DYNAMITE LANGUAGES C CXX)
    endif()
else()
    message(STATUS "CUDA support disabled; building CPU-only version")
    project(DYNAMITE LANGUAGES C CXX)
endif()

# [CPC-REV] macOS/Apple-Clang portability (referee 2, point 5).
# Apple Clang ships without libomp, so CMake's FindOpenMP fails unless it is told
# where a Homebrew/MacPorts libomp lives. Probe the usual prefixes and set
# OpenMP_ROOT before find_package(); FindOpenMP then discovers "-Xclang -fopenmp"
# together with the matching include/lib directories automatically.
if(APPLE AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
    set(_dmfe_omp_hints /opt/homebrew/opt/libomp /usr/local/opt/libomp /opt/local)
    find_program(DMFE_BREW_EXECUTABLE brew)
    if(DMFE_BREW_EXECUTABLE)
        execute_process(COMMAND "${DMFE_BREW_EXECUTABLE}" --prefix libomp
                        OUTPUT_VARIABLE _dmfe_brew_libomp
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        if(_dmfe_brew_libomp)
            list(APPEND _dmfe_omp_hints "${_dmfe_brew_libomp}")
        endif()
    endif()
    foreach(_dmfe_hint IN LISTS _dmfe_omp_hints)
        if(EXISTS "${_dmfe_hint}/include/omp.h")
            set(OpenMP_ROOT "${_dmfe_hint}" CACHE PATH "libomp installation prefix")
            message(STATUS "OpenMP: using libomp from ${OpenMP_ROOT}")
            break()
        endif()
    endforeach()
endif()

# [CPC-REV] Not REQUIRED any more: a missing OpenMP must degrade to a serial
# build, not abort the configuration. All omp.h includes are now _OPENMP-guarded.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING
        "OpenMP not found; building single-threaded. "
        "On macOS install it with 'brew install libomp' (and, if needed, re-run "
        "cmake with -DOpenMP_ROOT=$(brew --prefix libomp)).")
endif()

option(USE_HDF5 "Enable HDF5 support (compile-time link)" OFF)
# Optional Boost.Multiprecision (headers-only). If not found, we fall back to long double.
include(CheckIncludeFileCXX)
check_include_file_cxx("boost/multiprecision/cpp_dec_float.hpp" DMFE_HAVE_BOOST_MP)
if(DMFE_HAVE_BOOST_MP)
    add_compile_definitions(DMFE_HAVE_BOOST_MP=1)
else()
    add_compile_definitions(DMFE_HAVE_BOOST_MP=0)
endif()
option(USE_HDF5_RUNTIME "Enable runtime-optional HDF5 (dlopen)" ON)
option(DMFE_NATIVE "Enable -march=native for host compilation" ON)
option(DMFE_STATIC_OPENMP "Link OpenMP statically for portability across systems" OFF)

# Portable/cluster build helper: disable -march=native and prefer shared cudart
option(DMFE_PORTABLE_BUILD "Build portable binary for heterogeneous cluster nodes" OFF)
if(DMFE_PORTABLE_BUILD)
    set(DMFE_NATIVE OFF CACHE BOOL "" FORCE)
    set(DMFE_STATIC_OPENMP ON CACHE BOOL "" FORCE)
    if(DMFE_WITH_CUDA)
        set(DMFE_STATIC_CUDART OFF CACHE BOOL "" FORCE)
    endif()
endif()

if(DMFE_WITH_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
    # Consider setting this explicitly for the cluster nodes, e.g. -DCMAKE_CUDA_ARCHITECTURES=80
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# Debug helper: enable device debug, no fast-math, and Thrust debug checks
option(DMFE_DEBUG "Enable CUDA device debug and Thrust debug checks" OFF)

set(CMAKE_CXX_STANDARD 17)
if(DMFE_WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer static CUDA runtime to allow running on systems without CUDA (only when CUDA is enabled)
if(DMFE_WITH_CUDA)
    option(DMFE_STATIC_CUDART "Link CUDA runtime statically to run on systems without CUDA installed" ON)
    option(DMFE_BUILD_SHARED_VARIANT "Also build a shared-runtime variant to avoid nvlink host-lib warnings" ON)
else()
    set(DMFE_STATIC_CUDART OFF)
    set(DMFE_BUILD_SHARED_VARIANT OFF)
endif()

if(DMFE_PORTABLE_BUILD)
    set(DMFE_STATIC_CUDART OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Consider setting this explicitly for the cluster nodes, e.g. -DCMAKE_CUDA_ARCHITECTURES=80
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Base flags
set(common_host_flags -O3)
if(DMFE_WITH_CUDA)
    set(common_cuda_flags --extended-lambda --use_fast_math -allow-unsupported-compiler)
    # Workaround: nvcc 11.x + GCC >= 12 trips over libstdc++ locale headers when __STRICT_ANSI__ is set.
    # Undefine it for CUDA compilations when using CUDA 11.* with GCC 12+.
    if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_LESS 12)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
            list(APPEND common_cuda_flags -U__STRICT_ANSI__)
            message(STATUS "Applying GCC 12 + CUDA 11.x workaround for locale headers")
        endif()
    endif()
endif()

# Toolchain-specific host flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(OpenMP_CXX_FOUND)
        # [CPC-REV] OpenMP_CXX_FLAGS may contain several shell words (AppleClang:
        # "-Xclang -fopenmp"). Without separate_arguments() CMake passes the whole
        # string as ONE argv entry and the compiler aborts with
        #   clang++: error: unknown argument: '-Xclang -fopenmp'
        separate_arguments(DMFE_OPENMP_CXX_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
        list(APPEND common_host_flags ${DMFE_OPENMP_CXX_FLAGS})
        # Add OpenMP flags to CUDA compilation for host code in .cu files
        if(DMFE_WITH_CUDA)
            string(REPLACE ";" "," DMFE_OPENMP_CUDA_FLAGS "${DMFE_OPENMP_CXX_FLAGS}")
            list(APPEND common_cuda_flags -Xcompiler=${DMFE_OPENMP_CUDA_FLAGS})
        endif()
        if(DMFE_STATIC_OPENMP)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                list(APPEND common_host_flags -static-libgomp)
                if(DMFE_WITH_CUDA)
                    list(APPEND common_cuda_flags -Xcompiler=-static-libgomp)
                endif()
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
                # For LLVM libomp, attempt static linking
                list(APPEND common_host_flags -lomp)
                if(DMFE_WITH_CUDA)
                    list(APPEND common_cuda_flags -Xcompiler=-lomp)
                endif()
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
            elseif(APPLE)
                # [CPC-REV] Darwin provides neither crt0.o nor a static libSystem,
                # so '-static' cannot link. Link libomp statically instead when the
                # archive is available; otherwise fall back to the shared library.
                if(OpenMP_ROOT AND EXISTS "${OpenMP_ROOT}/lib/libomp.a")
                    message(STATUS "Static OpenMP on macOS: ${OpenMP_ROOT}/lib/libomp.a")
                    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_ROOT}/lib/libomp.a")
                else()
                    message(WARNING "DMFE_STATIC_OPENMP requested but libomp.a not found; linking libomp dynamically.")
                endif()
            endif()
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|PGI")
    list(APPEND common_host_flags -fast)
elseif(MSVC)
    list(APPEND common_host_flags /O2)
endif()

# [CPC-REV] -march=native only when allowed AND actually accepted by the compiler.
# Some Apple/ARM toolchains reject -march=native and only understand -mcpu=native.
if(DMFE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" DMFE_HAS_MARCH_NATIVE)
    if(DMFE_HAS_MARCH_NATIVE)
        list(APPEND common_host_flags -march=native)
    else()
        check_cxx_compiler_flag("-mcpu=native" DMFE_HAS_MCPU_NATIVE)
        if(DMFE_HAS_MCPU_NATIVE)
            message(STATUS "-march=native not supported; using -mcpu=native")
            list(APPEND common_host_flags -mcpu=native)
        else()
            message(STATUS "Neither -march=native nor -mcpu=native supported; skipping")
        endif()
    endif()
endif()

# Debug mode adjusts flags
if(DMFE_DEBUG)
    if(DMFE_WITH_CUDA)
        # Disable fast-math for determinism and better diagnostics
        list(REMOVE_ITEM common_cuda_flags --use_fast_math)
        # Device debug, no device optimizations, line info
        list(APPEND common_cuda_flags -G -lineinfo -Xptxas=-O0)
    endif()
    # Host debug
    set(common_host_flags -O0 -g)
endif()

# [CPC-REV][PERF] Stamp the CUDA toolkit version at configure time.  VersionInfo
# used to determine it by running `nvcc --version` through popen() at *runtime*,
# which (a) reported "No CUDA" in params.txt on any compute node without nvcc in
# PATH, even for a CUDA build, and (b) forked a shell on every single run.
if(DMFE_WITH_CUDA AND CMAKE_CUDA_COMPILER_VERSION)
    add_compile_definitions(DMFE_CUDA_TOOLKIT_VERSION="${CMAKE_CUDA_COMPILER_VERSION}")
endif()

if(DMFE_WITH_CUDA)
    message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA runtime linkage: ${DMFE_STATIC_CUDART}? (ON=static, OFF=shared)")
    message(STATUS "CUDA flags: ${common_cuda_flags}")
endif()
message(STATUS "Host compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Host flags: ${common_host_flags}")

# Define source lists
set(GPU_SOURCES "")
set(CPU_SOURCES "")

if(DMFE_WITH_CUDA)
    # CUDA sources (.cu files)
    set(GPU_SOURCES
        src/core/device_utils.cu
        src/core/stream_pool.cu
        src/core/device_constants.cu
        src/core/compute_utils.cu
        src/core/device_factory.cu
        src/core/backend_dispatch.cu
    src/io/io_output.cu
    src/core/gpu_memory_utils.cu
        src/interpolation/index_mat.cu
        src/interpolation/index_vec.cu
        src/interpolation/interpolation_core.cu
        src/convolution/convolution.cu
        src/math/math_ops.cu
        src/math/math_sigma.cu
        src/EOMs/time_steps.cu
        src/EOMs/runge_kutta.cu
        src/search/search_utils.cu
        src/sparsify/sparsify_utils.cu
    src/simulation/tail_fit.cu
    )
endif()

# CPU-only sources (.cpp files) - always included
set(CPU_SOURCES
    src/sparsify/sparsify_utils.cpp   # [CPC-REV] single host sparsification sweep
    src/core/config.cpp
    src/core/initialization.cpp
    src/core/program_start_time.cpp
    src/math/math_sigma.cpp
    src/math/lambert_w.cpp
    src/search/search_utils.cpp
    src/core/host_utils.cpp
    src/core/device_utils.cpp
    # [CPC-REV] backend_dispatch.cpp defines update(), energy(),
    # initializeGPUBackend() and synchronizeCompressedData() -- the same four
    # symbols as src/core/backend_dispatch.cu, and without any preprocessor
    # guard.  In a CUDA build both objects therefore ended up in the same static
    # library and which definition got linked was left to the linker (an ODR
    # violation).  If the CPU one had won, initializeGPUBackend() would have
    # reported "program was built without CUDA support" on a machine with a
    # perfectly good GPU.  It is now compiled only in CPU-only builds.
    src/convolution/convolution.cpp
    src/interpolation/index_vec.cpp
    src/interpolation/index_mat.cpp
    src/interpolation/interpolation_core.cpp
    src/io/io_input.cpp
    src/io/io_save_params.cpp
    src/io/io_save_binary.cpp
    src/io/io_save_hdf5.cpp
    src/io/io_save_compressed.cpp
    src/io/io_save_driver.cpp
    src/io/io_save_history.cpp   # [CPC-REV] history files, all builds
    src/io/io_snapshot_host.cpp
    src/io/io_utils.cpp
    src/math/math_ops.cpp
    src/simulation/simulation_control.cpp
    src/simulation/simulation_runner.cpp
    src/simulation/tail_fit.cpp
    src/ui/simulation_text_ui.cpp
    src/version/version_info.cpp
    src/version/version_compat.cpp
    src/EOMs/time_steps.cpp
    src/EOMs/runge_kutta.cpp
    src/grid/theta_grid.cpp
    src/grid/grid_io.cpp
    src/grid/phi_grid.cpp
    src/grid/grid_cli.cpp
    src/grid/pos_grid.cpp

    src/grid/integration.cpp
    src/grid/interpolation_weights.cpp
)

# Build the high-precision theta grid generator with strict FP (no fast-math, no contraction)
# Use per-source COMPILE_OPTIONS so these appear at the end of the compile line.
# Guard Clang-specific flags with a compiler check.
set_property(SOURCE src/grid/theta_grid.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:Clang>:-fno-fast-math>"
    "$<$<CXX_COMPILER_ID:Clang>:-ffp-contract=off>"
)

# Re-apply fast-math to all other CPU sources explicitly so we don't pass both
# -ffast-math and -fno-fast-math to the same translation unit.
set(FASTMATH_CPU_SOURCES ${CPU_SOURCES})
list(REMOVE_ITEM FASTMATH_CPU_SOURCES src/grid/theta_grid.cpp)
set_source_files_properties(${FASTMATH_CPU_SOURCES} PROPERTIES COMPILE_OPTIONS
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-ffast-math>"
)

# [CPC-REV] src/sparsify/sparsify_utils.cpp used to be compiled ONLY in CPU-only
# builds, while a second, non-equivalent copy of sparsifyNscale() lived in
# sparsify_utils.cu and was used by CUDA builds.  There is now a single host
# implementation, compiled in both configurations.

# [CPC-REV] CPU-only backend dispatch (see the note in CPU_SOURCES above).
if(NOT DMFE_WITH_CUDA)
    list(APPEND CPU_SOURCES src/core/backend_dispatch.cpp)
endif()

# Be explicit about source file languages to avoid CMake misclassifying some .cpp
# files as CUDA when CUDA is enabled (which can cause nvcc to compile libstdc++ headers).
if(DMFE_WITH_CUDA)
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()
set_source_files_properties(${CPU_SOURCES} PROPERTIES LANGUAGE CXX)

# Some C++ sources use Thrust device vectors and must be compiled with nvcc when CUDA is enabled.
if(DMFE_WITH_CUDA)
    set_source_files_properties(
        src/simulation/simulation_control.cpp
        src/simulation/simulation_runner.cpp
        # These C++ sources reference thrust::device_vector APIs (e.g., energyGPU)
        # and must be compiled with the CUDA toolchain to ensure ABI/Name mangling
        # matches the definitions in .cu files when using NVHPC toolchains.
        src/io/io_save_params.cpp
        src/io/io_output.cpp
        # [CPC-REV] the single host sparsification sweep now lives here; its header
        # pulls <cuda_runtime.h> when CUDA is enabled, so compile it with the CUDA
        # toolchain in that configuration (it contains no device code).
        src/sparsify/sparsify_utils.cpp
        PROPERTIES LANGUAGE CUDA)
    # Explicitly force the UI implementation as C++ (not CUDA) even when CUDA is enabled
    set_source_files_properties(src/ui/simulation_text_ui.cpp PROPERTIES LANGUAGE CXX)
endif()

add_library(DFME-core ${GPU_SOURCES} ${CPU_SOURCES})

target_include_directories(DFME-core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# [CPC-REV] Version and Git state are stamped at build time (they used to be read
# by running git when the program starts, i.e. from whatever directory it was
# started in).  DMFE_FALLBACK_VERSION is used for builds outside a Git checkout.
set(DMFE_HOME_OUTPUT_DIR "" CACHE STRING
    "Output root used when the executable lies inside the home directory, e.g. /nobackups/{USER}/Results/ ({USER} is replaced by the user name). Empty: Results/ below the working directory.")
if(DMFE_HOME_OUTPUT_DIR)
    target_compile_definitions(DFME-core PUBLIC DMFE_HOME_OUTPUT_DIR="${DMFE_HOME_OUTPUT_DIR}")
endif()
set(DMFE_FALLBACK_VERSION "1.4.0.0" CACHE STRING "Version reported when building outside a Git checkout")
set(DMFE_VERSION_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/dmfe_version_stamp.hpp")
add_custom_target(dmfe_version_stamp ALL
    COMMAND ${CMAKE_COMMAND}
            -DSRC=${CMAKE_CURRENT_SOURCE_DIR}
            -DOUT=${DMFE_VERSION_HEADER}
            -DFALLBACK=${DMFE_FALLBACK_VERSION}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dmfe_version.cmake
    BYPRODUCTS ${DMFE_VERSION_HEADER}
    COMMENT "Stamping version information"
    VERBATIM)
add_dependencies(DFME-core dmfe_version_stamp)
target_include_directories(DFME-core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

if(DMFE_WITH_CUDA)
    target_include_directories(DFME-core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(DFME-core PUBLIC DMFE_WITH_CUDA=1)
else()
    target_compile_definitions(DFME-core PUBLIC DMFE_WITH_CUDA=0)
endif()

if(DMFE_TAIL_FIT_DEBUG)
    target_compile_definitions(DFME-core PUBLIC DMFE_TAIL_FIT_DEBUG=1)
else()
    target_compile_definitions(DFME-core PUBLIC DMFE_TAIL_FIT_DEBUG=0)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(DFME-core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_HDF5)
    find_package(HDF5 COMPONENTS C CXX)
    if(HDF5_FOUND)
        target_compile_definitions(DFME-core PUBLIC USE_HDF5)
        target_link_libraries(DFME-core PRIVATE $<HOST_LINK:HDF5::HDF5>)
    else()
        message(WARNING "HDF5 not found; building without HDF5 support")
    endif()
endif()

if(USE_HDF5_RUNTIME)
    target_sources(DFME-core PRIVATE src/io/h5_runtime.cpp)
    target_compile_definitions(DFME-core PUBLIC H5_RUNTIME_OPTIONAL=1)
endif()

# CPU source files are already in CPU_SOURCES list above

add_executable(RG-Evo main.cpp)

target_link_libraries(RG-Evo PRIVATE DFME-core)
target_link_options(RG-Evo PRIVATE $<HOST_LINK:-ldl>)

# Ensure device linking stage unifies __constant__ definitions across TUs (CUDA only)
if(DMFE_WITH_CUDA)
    set_target_properties(DFME-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(RG-Evo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

set_target_properties(DFME-core PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RG-Evo PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Set per-target CUDA runtime linkage (CUDA only)
if(DMFE_WITH_CUDA AND DMFE_STATIC_CUDART)
    set_target_properties(DFME-core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    set_target_properties(RG-Evo PROPERTIES CUDA_RUNTIME_LIBRARY Static)
elseif(DMFE_WITH_CUDA)
    set_target_properties(DFME-core PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    set_target_properties(RG-Evo PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

# Optional shared-runtime variant (CUDA only)
if(DMFE_WITH_CUDA AND DMFE_BUILD_SHARED_VARIANT)
    add_executable(RG-Evo-shared main.cpp)
    target_link_libraries(RG-Evo-shared PRIVATE DFME-core)
    target_link_options(RG-Evo-shared PRIVATE $<HOST_LINK:-ldl>)
    set_target_properties(RG-Evo-shared PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    target_compile_options(RG-Evo-shared PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${common_cuda_flags}>
        $<$<COMPILE_LANGUAGE:CXX>:${common_host_flags}>
    )
    if(DMFE_DEBUG)
        target_compile_definitions(RG-Evo-shared PUBLIC THRUST_DEBUG=1)
    endif()
endif()

foreach(tgt IN ITEMS DFME-core RG-Evo)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${common_host_flags}>
    )
    if(DMFE_WITH_CUDA)
        target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${common_cuda_flags}>
        )
    endif()
    if(DMFE_DEBUG)
        target_compile_definitions(${tgt} PUBLIC THRUST_DEBUG=1)
    endif()
endforeach()

# [CPC-REV] moved to the top of the file (before project()); kept here as a no-op
# so that an out-of-date build directory still gets the variable set.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Simple test for barycentric interpolation weights (standalone, no DFME-core linkage)
add_executable(test_interp_weights
    src/grid/test_interpolation_weights.cpp
    src/grid/theta_grid.cpp
    src/grid/interpolation_weights.cpp
    src/math/lambert_w.cpp
)
set_target_properties(test_interp_weights PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    LINKER_LANGUAGE CXX
)
target_include_directories(test_interp_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)