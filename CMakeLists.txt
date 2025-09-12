cmake_minimum_required(VERSION 3.24)

# Prefer clang++-14 as CUDA host compiler if available (nvcc error suggested using it)
find_program(CLANGPP_14 NAMES clang++-14)
if(CLANGPP_14)
	# Must be set before enabling CUDA language
	set(CMAKE_CUDA_HOST_COMPILER ${CLANGPP_14})
	message(STATUS "Using ${CLANGPP_14} as CUDA host compiler")
endif()

project(DMFE LANGUAGES C CXX CUDA)

option(USE_HDF5 "Enable HDF5 support (compile-time link)" OFF)
option(USE_HDF5_RUNTIME "Enable runtime-optional HDF5 (dlopen)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Avoid nvlink warnings caused by host-only system libs being injected via
# static CUDA runtime. Shared runtime keeps those on the host link only.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if(NOT CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

add_library(DFME-core
	src/core/device_utils.cu
	src/core/stream_pool.cu
	src/core/device_constants.cu
	src/core/host_utils.cu
	src/core/compute_utils.cu
	src/core/io_utils.cu
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
)

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

add_executable(RG-Evo main.cu)
target_link_libraries(RG-Evo PRIVATE DFME-core)
# No need to link HDF5 on RG-Evo; runtime loader handles it when present.
# Ensure libdl is available for the loader on host link only
target_link_options(RG-Evo PRIVATE $<HOST_LINK:-ldl>)

# Ensure device linking stage unifies __constant__ definitions across TUs
set_target_properties(DFME-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RG-Evo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(DFME-core PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RG-Evo PROPERTIES
	LINKER_LANGUAGE CXX
	# Put the executable in the current project folder (independent of folder name)
	RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

set(common_cuda_flags --extended-lambda --use_fast_math -allow-unsupported-compiler)
set(common_host_flags -O3 -ffast-math)

foreach(tgt IN ITEMS DFME-core RG-Evo)
	target_compile_options(${tgt} PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:${common_cuda_flags}>
		$<$<COMPILE_LANGUAGE:CXX>:${common_host_flags}>
	)
endforeach()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
