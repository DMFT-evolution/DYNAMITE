---
title: src/convolution/convolution.cpp

---

# src/convolution/convolution.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[ConvA](#function-conva)**(const std::vector< double > & f, const std::vector< double > & g, const double t) |
| std::vector< double > | **[ConvR](#function-convr)**(const std::vector< double > & f, const std::vector< double > & g, const double t) |


## Functions Documentation

### function ConvA

```cpp
std::vector< double > ConvA(
    const std::vector< double > & f,
    const std::vector< double > & g,
    const double t
)
```


### function ConvR

```cpp
std::vector< double > ConvR(
    const std::vector< double > & f,
    const std::vector< double > & g,
    const double t
)
```




## Source code

```cpp
#include "convolution/convolution.hpp"
#include "core/globals.hpp"
#include <omp.h>

std::vector<double> ConvA(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->h_integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(depth, 0.0);
    if (depth == 1)
    {
        double temp = 0.0;
        #pragma omp parallel for reduction(+:temp)
        for (size_t j = 0; j < length; j++)
        {
            temp += t * sim->h_integ[j] * f[j] * g[j];
        }
        out[0] = temp;
    }
    else
    {
        #pragma omp parallel for
        for (size_t j = 0; j < depth; j++)
        {
            for (size_t i = 0; i < length; i++)
            {
                out[j] += sim->h_integ[i] * f[j * length + i] * g[j * length + i];
            }
            out[j] *= t * sim->h_theta[j];
        }
    }
    return out;
}

std::vector<double> ConvR(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->h_integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(length, 0.0);
    #pragma omp parallel for
    for (size_t j = 0; j < length; j++)
    {
        for (size_t i = 0; i < depth; i++)
        {
            out[j] += sim->h_integ[i] * f[j * length + i] * g[j * length + i];
        }
        out[j] *= t * (1 - sim->h_theta[j]);
    }
    return out;
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:51 +0200
