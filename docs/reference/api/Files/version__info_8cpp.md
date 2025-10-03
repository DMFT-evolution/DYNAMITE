---
title: src/version/version_info.cpp

---

# src/version/version_info.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| std::string | **[executeCommand](#function-executecommand)**(const std::string & command) |

## Attributes

|                | Name           |
| -------------- | -------------- |
| VersionInfo | **[g_version_info](#variable-g-version-info)**  |


## Functions Documentation

### function executeCommand

```cpp
static std::string executeCommand(
    const std::string & command
)
```



## Attributes Documentation

### variable g_version_info

```cpp
VersionInfo g_version_info;
```



## Source code

```cpp
#include "version/version_info.hpp"
#include <array>
#include <memory>
#include <cstdio>
#include <sstream>

static std::string executeCommand(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        return "unknown";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

VersionInfo::VersionInfo() {
    git_hash = executeCommand("git rev-parse HEAD 2>/dev/null");
    if (git_hash == "unknown") {
        git_hash = executeCommand("git rev-parse --short HEAD 2>/dev/null");
    }
    git_branch = executeCommand("git rev-parse --abbrev-ref HEAD 2>/dev/null");
    git_tag = executeCommand("git describe --tags --exact-match 2>/dev/null");
    std::string git_status = executeCommand("git diff-index --quiet HEAD -- 2>/dev/null; echo $?");
    git_dirty = (git_status != "0");
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
    // Runtime check for CUDA version (replaces compile-time __CUDACC__ check)
    std::string nvcc_output = executeCommand("nvcc --version 2>/dev/null");
    if (nvcc_output.find("release") != std::string::npos) {
        // Extract version, e.g., "release 12.1" -> "CUDA 12.1"
        size_t pos = nvcc_output.find("release ");
        if (pos != std::string::npos) {
            std::string version = nvcc_output.substr(pos + 8, 4);  // Assumes format like "12.1"
            cuda_version = "CUDA " + version;
        } else {
            cuda_version = "CUDA (version unknown)";
        }
    } else {
        cuda_version = "No CUDA";
    }
    code_version = executeCommand("git describe --tags --always --dirty 2>/dev/null");
    if (code_version == "unknown") {
        code_version = "v1.0.0-dev";
    }
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
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
