---
title: src/math/math_sigma.cpp

---

# src/math/math_sigma.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[SigmaR](#function-sigmar)**(const std::vector< double > & qk, const std::vector< double > & qr, std::vector< double > & result) |
| std::vector< double > | **[SigmaK10](#function-sigmak10)**(const std::vector< double > & qk) |
| std::vector< double > | **[SigmaR10](#function-sigmar10)**(const std::vector< double > & qk, const std::vector< double > & qr) |
| std::vector< double > | **[SigmaK01](#function-sigmak01)**(const std::vector< double > & qk) |
| std::vector< double > | **[SigmaR01](#function-sigmar01)**(const std::vector< double > & qk, const std::vector< double > & qr) |
| void | **[SigmaK](#function-sigmak)**(const std::vector< double > & qk, std::vector< double > & result) |


## Functions Documentation

### function SigmaR

```cpp
void SigmaR(
    const std::vector< double > & qk,
    const std::vector< double > & qr,
    std::vector< double > & result
)
```


### function SigmaK10

```cpp
std::vector< double > SigmaK10(
    const std::vector< double > & qk
)
```


### function SigmaR10

```cpp
std::vector< double > SigmaR10(
    const std::vector< double > & qk,
    const std::vector< double > & qr
)
```


### function SigmaK01

```cpp
std::vector< double > SigmaK01(
    const std::vector< double > & qk
)
```


### function SigmaR01

```cpp
std::vector< double > SigmaR01(
    const std::vector< double > & qk,
    const std::vector< double > & qr
)
```


### function SigmaK

```cpp
void SigmaK(
    const std::vector< double > & qk,
    std::vector< double > & result
)
```




## Source code

```cpp
#include "math/math_sigma.hpp"
#include "math/math_ops.hpp"
#include <vector>
#include <omp.h>

// CPU versions with OpenMP parallelization
void SigmaR(const std::vector<double>& qk, const std::vector<double>& qr, std::vector<double>& result)
{
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]) * qr[i];
    }
}

std::vector<double> SigmaK10(const std::vector<double>& qk)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return std::move(result);
}

std::vector<double> SigmaR10(const std::vector<double>& qk, const std::vector<double>& qr)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDDflambda(qk[i]) * qr[i];
    }
    return std::move(result);
}

std::vector<double> SigmaK01(const std::vector<double>& qk)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = 0.0;
    }
    return std::move(result);
}

std::vector<double> SigmaR01(const std::vector<double>& qk, const std::vector<double>& qr)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return std::move(result);
}

// CPU version of SigmaK with OpenMP parallelization
void SigmaK(const std::vector<double>& qk, std::vector<double>& result)
{
    #pragma omp parallel for if(qk.size() > 1000)
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
