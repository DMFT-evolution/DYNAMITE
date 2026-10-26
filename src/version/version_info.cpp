#include "version/version_info.hpp"
#include <sstream>
#include "dmfe_version_stamp.hpp"   // generated at build time by cmake/dmfe_version.cmake

VersionInfo::VersionInfo() {
    // [CPC-REV] Git state of the source tree at build time (see cmake/dmfe_version.cmake).
    // It used to be obtained by running git when the program started, which reported the
    // state of the current working directory instead of the build.
    git_hash = DMFE_GIT_HASH;
    git_branch = DMFE_GIT_BRANCH;
    git_tag = DMFE_GIT_TAG;
    git_dirty = (DMFE_GIT_DIRTY != 0);
    build_date = __DATE__;
    build_time = __TIME__;
#ifdef __GNUC__
    std::ostringstream gcc_version;
    gcc_version << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    compiler_version = gcc_version.str();
#elif defined(__clang__)
    std::ostringstream clang_version;
    clang_version << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    compiler_version = clang_version.str();
#else
    compiler_version = "Unknown compiler";
#endif
    // [CPC-REV][PERF] The CUDA toolkit version is stamped in at configure time.
    // It used to be discovered by running `nvcc --version` through popen() on every
    // run, which reported "No CUDA" in params.txt whenever the compute node had no
    // nvcc in PATH -- i.e. for most cluster jobs, including CUDA builds -- and cost
    // a shell fork per run.  DMFE_CUDA_TOOLKIT_VERSION is defined by CMake exactly
    // when the build really does contain CUDA code.
#ifdef DMFE_CUDA_TOOLKIT_VERSION
    {
        std::string v = DMFE_CUDA_TOOLKIT_VERSION;
        // Keep the historical "CUDA <major>.<minor>" shape.
        const size_t first = v.find('.');
        if (first != std::string::npos) {
            const size_t second = v.find('.', first + 1);
            if (second != std::string::npos) v = v.substr(0, second);
        }
        cuda_version = "CUDA " + v;
    }
#else
    cuda_version = "No CUDA";
#endif
    code_version = DMFE_CODE_VERSION;
}

std::string VersionInfo::toString() const {
    std::ostringstream oss;
    oss << "Version: " << code_version;
    if (git_dirty) oss << " (modified)";
    oss << "\nGit Hash: " << git_hash;
    oss << "\nGit Branch: " << git_branch;
    if (git_tag != "unknown") {
        oss << "\nGit Tag: " << git_tag;
    }
    oss << "\nBuild Date: " << build_date << " " << build_time;
    oss << "\nCompiler: " << compiler_version;
    oss << "\nCUDA: " << cuda_version;
    return oss.str();
}

VersionInfo g_version_info; // global instance
