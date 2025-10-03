---
title: include/core/host_utils.hpp

---

# include/core/host_utils.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[Product](#function-product)**(const std::vector< double > & vec1, const std::vector< double > & vec2, std::vector< double > & result) |
| void | **[scaleVec](#function-scalevec)**(const std::vector< double > & vec1, double real, std::vector< double > & result) |
| void | **[printVectorDifference](#function-printvectordifference)**(const std::vector< double > & a, const std::vector< double > & b) |


## Functions Documentation

### function Product

```cpp
void Product(
    const std::vector< double > & vec1,
    const std::vector< double > & vec2,
    std::vector< double > & result
)
```


### function scaleVec

```cpp
void scaleVec(
    const std::vector< double > & vec1,
    double real,
    std::vector< double > & result
)
```


### function printVectorDifference

```cpp
void printVectorDifference(
    const std::vector< double > & a,
    const std::vector< double > & b
)
```




## Source code

```cpp
// host_utils.hpp - Small host-side vector helpers used across the project
#pragma once

#include <vector>
// Thrust and CUDA types are not required for these host-only declarations

// Component-wise product: result[i] = vec1[i] * vec2[i]
void Product(const std::vector<double>& vec1, const std::vector<double>& vec2, std::vector<double>& result);
// Scale: result[i] = vec1[i] * real
void scaleVec(const std::vector<double>& vec1, double real, std::vector<double>& result);
// Print per-index absolute differences and their total
void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b);
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
