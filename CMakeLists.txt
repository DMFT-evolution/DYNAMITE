cmake_minimum_required(VERSION 3.24)

# Prefer clang++-14 as CUDA host compiler only if explicitly requested
option(DMFE_PREFER_CLANG_HOST "Prefer clang++ as CUDA host compiler for nvcc" OFF)
if(DMFE_PREFER_CLANG_HOST AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(CLANGPP_14 NAMES clang++-14)
    if(CLANGPP_14)
        # Must be set before enabling CUDA language
        set(CMAKE_CUDA_HOST_COMPILER ${CLANGPP_14})
        message(STATUS "Using ${CLANGPP_14} as CUDA host compiler")
    endif()
endif()

project(DMFE LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)

option(USE_HDF5 "Enable HDF5 support (compile-time link)" OFF)
option(USE_HDF5_RUNTIME "Enable runtime-optional HDF5 (dlopen)" ON)
option(DMFE_NATIVE "Enable -march=native for host compilation" ON)
option(DMFE_STATIC_OPENMP "Link OpenMP statically for portability across systems" OFF)

# Portable/cluster build helper: disable -march=native and prefer shared cudart
option(DMFE_PORTABLE_BUILD "Build portable binary for heterogeneous cluster nodes" OFF)
if(DMFE_PORTABLE_BUILD)
    set(DMFE_NATIVE OFF CACHE BOOL "" FORCE)
    set(DMFE_STATIC_OPENMP ON CACHE BOOL "" FORCE)
endif()

# Debug helper: enable device debug, no fast-math, and Thrust debug checks
option(DMFE_DEBUG "Enable CUDA device debug and Thrust debug checks" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer static CUDA runtime to allow running on systems without CUDA
option(DMFE_STATIC_CUDART "Link CUDA runtime statically to run on systems without CUDA installed" ON)
option(DMFE_BUILD_SHARED_VARIANT "Also build a shared-runtime variant to avoid nvlink host-lib warnings" ON)

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
set(common_cuda_flags --extended-lambda --use_fast_math -allow-unsupported-compiler)
set(common_host_flags -O3)

# Toolchain-specific host flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND common_host_flags -ffast-math)
    if(OpenMP_CXX_FOUND)
        list(APPEND common_host_flags ${OpenMP_CXX_FLAGS})
        # Add OpenMP flags to CUDA compilation for host code in .cu files
        list(APPEND common_cuda_flags -Xcompiler=${OpenMP_CXX_FLAGS})
        if(DMFE_STATIC_OPENMP)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                list(APPEND common_host_flags -static-libgomp)
                list(APPEND common_cuda_flags -Xcompiler=-static-libgomp)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # For LLVM libomp, attempt static linking
                list(APPEND common_host_flags -lomp)
                list(APPEND common_cuda_flags -Xcompiler=-lomp)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
            endif()
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|PGI")
    list(APPEND common_host_flags -fast)
elseif(MSVC)
    list(APPEND common_host_flags /O2)
endif()

# -march=native only when allowed
if(DMFE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND common_host_flags -march=native)
endif()

# Debug mode adjusts flags
if(DMFE_DEBUG)
    # Disable fast-math for determinism and better diagnostics
    list(REMOVE_ITEM common_cuda_flags --use_fast_math)
    # Device debug, no device optimizations, line info
    list(APPEND common_cuda_flags -G -lineinfo -Xptxas=-O0)
    # Host debug
    set(common_host_flags -O0 -g)
endif()

message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA runtime linkage: ${DMFE_STATIC_CUDART}? (ON=static, OFF=shared)")
message(STATUS "Host compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Host flags: ${common_host_flags}")
message(STATUS "CUDA flags: ${common_cuda_flags}")

add_library(DFME-core
    src/core/device_utils.cu
    src/core/stream_pool.cu
    src/core/device_constants.cu
    src/core/host_utils.cu
    src/core/compute_utils.cu
    src/io/io_utils.cu
    src/io/io_output.cu
    src/io/io_input.cu
    src/core/gpu_memory_utils.cu
    src/core/config.cu
    src/core/initialization.cu
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
    src/simulation/simulation_control.cu
    src/simulation/simulation_runner.cu
    src/version/version_info.cu
    src/version/version_compat.cu
)

target_include_directories(DFME-core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
)

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

# Add OpenMP-enabled CPU source files
target_sources(DFME-core PRIVATE src/math/math_sigma.cpp)
target_sources(DFME-core PRIVATE src/search/search_utils.cpp)
target_sources(DFME-core PRIVATE src/core/host_utils.cpp)
target_sources(DFME-core PRIVATE src/convolution/convolution.cpp)
target_sources(DFME-core PRIVATE src/interpolation/index_vec.cpp)
target_sources(DFME-core PRIVATE src/interpolation/index_mat.cpp)

add_executable(RG-Evo main.cu)
target_link_libraries(RG-Evo PRIVATE DFME-core)
target_link_options(RG-Evo PRIVATE $<HOST_LINK:-ldl>)

# Ensure device linking stage unifies __constant__ definitions across TUs
set_target_properties(DFME-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RG-Evo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(DFME-core PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RG-Evo PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Set per-target CUDA runtime linkage
if(DMFE_STATIC_CUDART)
    set_target_properties(DFME-core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    set_target_properties(RG-Evo PROPERTIES CUDA_RUNTIME_LIBRARY Static)
else()
    set_target_properties(DFME-core PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    set_target_properties(RG-Evo PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

# Optional shared-runtime variant
if(DMFE_BUILD_SHARED_VARIANT)
    add_executable(RG-Evo-shared main.cu)
    target_link_libraries(RG-Evo-shared PRIVATE DFME-core)
    target_link_options(RG-Evo-shared PRIVATE $<HOST_LINK:-ldl>)
    set_target_properties(RG-Evo-shared PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    foreach(tgt IN ITEMS RG-Evo-shared)
        target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${common_cuda_flags}>
            $<$<COMPILE_LANGUAGE:CXX>:${common_host_flags}>
        )
        if(DMFE_DEBUG)
            target_compile_definitions(${tgt} PUBLIC THRUST_DEBUG=1)
        endif()
    endforeach()
endif()

foreach(tgt IN ITEMS DFME-core RG-Evo)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${common_cuda_flags}>
        $<$<COMPILE_LANGUAGE:CXX>:${common_host_flags}>
    )
    if(DMFE_DEBUG)
        target_compile_definitions(${tgt} PUBLIC THRUST_DEBUG=1)
    endif()
endforeach()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)