---
title: include/version/version_info.hpp

---

# include/version/version_info.hpp



## Classes

|                | Name           |
| -------------- | -------------- |
| struct | **[VersionInfo](Classes/structVersionInfo.md)**  |
| struct | **[VersionAnalysis](Classes/structVersionAnalysis.md)**  |

## Types

|                | Name           |
| -------------- | -------------- |
| enum class| **[VersionCompatibility](#enum-versioncompatibility)** { IDENTICAL, COMPATIBLE, WARNING, INCOMPATIBLE} |

## Attributes

|                | Name           |
| -------------- | -------------- |
| VersionInfo | **[g_version_info](#variable-g-version-info)**  |

## Types Documentation

### enum VersionCompatibility

| Enumerator | Value | Description |
| ---------- | ----- | ----------- |
| IDENTICAL | |   |
| COMPATIBLE | |   |
| WARNING | |   |
| INCOMPATIBLE | |   |






## Attributes Documentation

### variable g_version_info

```cpp
VersionInfo g_version_info;
```



## Source code

```cpp
#pragma once
#include <string>
#include <vector>

struct VersionInfo {
    std::string git_hash;
    std::string git_branch;
    std::string git_tag;
    bool git_dirty;
    std::string build_date;
    std::string build_time;
    std::string compiler_version;
    std::string cuda_version;
    std::string code_version;
    VersionInfo();
    std::string toString() const;
};

enum class VersionCompatibility {
    IDENTICAL,
    COMPATIBLE,
    WARNING,
    INCOMPATIBLE
};

struct VersionAnalysis {
    VersionCompatibility level;
    std::string file_version;
    std::string current_version;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

extern VersionInfo g_version_info;
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
