---
title: include/version/version_compat.hpp

---

# include/version/version_compat.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| VersionAnalysis | **[analyzeVersionCompatibility](#function-analyzeversioncompatibility)**(const std::string & paramFilename) |
| bool | **[checkVersionCompatibilityInteractive](#function-checkversioncompatibilityinteractive)**(const std::string & paramFilename) |
| VersionAnalysis | **[checkVersionCompatibility](#function-checkversioncompatibility)**(const std::string & paramFilename) |
| bool | **[checkVersionCompatibilityBasic](#function-checkversioncompatibilitybasic)**(const std::string & paramFilename) |


## Functions Documentation

### function analyzeVersionCompatibility

```cpp
VersionAnalysis analyzeVersionCompatibility(
    const std::string & paramFilename
)
```


### function checkVersionCompatibilityInteractive

```cpp
bool checkVersionCompatibilityInteractive(
    const std::string & paramFilename
)
```


### function checkVersionCompatibility

```cpp
VersionAnalysis checkVersionCompatibility(
    const std::string & paramFilename
)
```


### function checkVersionCompatibilityBasic

```cpp
bool checkVersionCompatibilityBasic(
    const std::string & paramFilename
)
```




## Source code

```cpp
#pragma once
#include "version/version_info.hpp"
#include <string>

VersionAnalysis analyzeVersionCompatibility(const std::string& paramFilename);
bool checkVersionCompatibilityInteractive(const std::string& paramFilename);
VersionAnalysis checkVersionCompatibility(const std::string& paramFilename);
bool checkVersionCompatibilityBasic(const std::string& paramFilename);
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
