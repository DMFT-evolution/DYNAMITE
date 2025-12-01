#ifndef DMFE_CONFIG_BUILD_HPP
#define DMFE_CONFIG_BUILD_HPP

// This header exposes compile-time build configuration macros.
// DMFE_WITH_CUDA is defined by CMake to indicate whether CUDA support is compiled in.

// Normalize DMFE_WITH_CUDA to a numeric 0/1 so it is safe in #if expressions across compilers (incl. nvcc).
// CMake might define it as ON/OFF/TRUE/FALSE/YES/NO or numeric 0/1. We map the common tokens to numbers
// using token pasting without defining global ON/OFF macros (we also clean up helpers afterwards).
#ifndef DMFE_WITH_CUDA
    #define DMFE_WITH_CUDA 0
#else
    // Helper token-paste mapping: DMFE_PP_VALUE_<TOKEN> -> 0/1
    #define DMFE_PP_VALUE_ON    1
    #define DMFE_PP_VALUE_YES   1
    #define DMFE_PP_VALUE_TRUE  1
    #define DMFE_PP_VALUE_OFF   0
    #define DMFE_PP_VALUE_NO    0
    #define DMFE_PP_VALUE_FALSE 0
    #define DMFE_PP_CAT_(a,b) a##b
    #define DMFE_PP_CAT(a,b) DMFE_PP_CAT_(a,b)
    #define DMFE_PP_VALUE(token) DMFE_PP_CAT(DMFE_PP_VALUE_, token)

    // Evaluate to 1 if numeric 1 or token mapped to 1, else 0.
    #if (DMFE_WITH_CUDA + 0) == 1 || (DMFE_PP_VALUE(DMFE_WITH_CUDA) == 1)
        #undef DMFE_WITH_CUDA
        #define DMFE_WITH_CUDA 1
    #else
        // If explicitly 0 or maps to 0, normalize to 0; otherwise default to 0
        #undef DMFE_WITH_CUDA
        #define DMFE_WITH_CUDA 0
    #endif

    // Cleanup helper macros to avoid leaking into translation units
    #undef DMFE_PP_VALUE_ON
    #undef DMFE_PP_VALUE_YES
    #undef DMFE_PP_VALUE_TRUE
    #undef DMFE_PP_VALUE_OFF
    #undef DMFE_PP_VALUE_NO
    #undef DMFE_PP_VALUE_FALSE
    #undef DMFE_PP_CAT_
    #undef DMFE_PP_CAT
    #undef DMFE_PP_VALUE
#endif

// Normalize DMFE_TAIL_FIT_DEBUG similarly
#ifndef DMFE_TAIL_FIT_DEBUG
    #define DMFE_TAIL_FIT_DEBUG 0
#else
    #if (DMFE_TAIL_FIT_DEBUG + 0) == 1
        #undef DMFE_TAIL_FIT_DEBUG
        #define DMFE_TAIL_FIT_DEBUG 1
    #else
        #undef DMFE_TAIL_FIT_DEBUG
        #define DMFE_TAIL_FIT_DEBUG 0
    #endif
#endif

// CUDA qualifier macros: when CUDA is enabled, use CUDA keywords; otherwise define as empty.
#if DMFE_WITH_CUDA
    #define DMFE_DEVICE __device__
    #define DMFE_HOST __host__
    #define DMFE_GLOBAL __global__
    #define DMFE_HOST_DEVICE __host__ __device__
#else
    #define DMFE_DEVICE
    #define DMFE_HOST
    #define DMFE_GLOBAL
    #define DMFE_HOST_DEVICE
#endif

#endif // DMFE_CONFIG_BUILD_HPP
