---
title: src/interpolation/index_mat.cpp

---

# src/interpolation/index_mat.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[indexMatAll](#function-indexmatall)**(const vector< double > & posx, const vector< size_t > & indsy, const vector< double > & weightsy, const vector< double > & dtratio, vector< double > & qK_result, vector< double > & qR_result) |

## Attributes

|                | Name           |
| -------------- | -------------- |
| SimulationConfig | **[config](#variable-config)**  |
| SimulationData * | **[sim](#variable-sim)**  |


## Functions Documentation

### function indexMatAll

```cpp
void indexMatAll(
    const vector< double > & posx,
    const vector< size_t > & indsy,
    const vector< double > & weightsy,
    const vector< double > & dtratio,
    vector< double > & qK_result,
    vector< double > & qR_result
)
```



## Attributes Documentation

### variable config

```cpp
SimulationConfig config;
```


### variable sim

```cpp
SimulationData * sim;
```



## Source code

```cpp
#include "interpolation/index_mat.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include <algorithm>
#include <numeric>
#include <omp.h>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

// CPU version of indexMatAll
void indexMatAll(const vector<double>& posx, const vector<size_t>& indsy,
    const vector<double>& weightsy, const vector<double>& dtratio,
    vector<double>& qK_result, vector<double>& qR_result)
{
    size_t prod = indsy.size();
    size_t dims2 = weightsy.size();
    size_t depth = dims2 / prod;
    size_t t1len = dtratio.size();

    double inx, inx2, inx3;
    size_t inds, indsx;

    #pragma omp parallel for private(inx, inx2, inx3, indsx, inds) schedule(static)
    for (size_t j = 0; j < prod; j++)
    {
        indsx = max(min((size_t)posx[j], (size_t)(posx[prod - 1] - 0.5)), (size_t)1);
        inx = posx[j] - indsx;
        inx2 = inx * inx;
        inx3 = inx2 * inx;
        inds = (indsx - 1) * config.len + indsy[j];

        auto weights_start = weightsy.begin() + depth * j;
        if (indsx < t1len - 1)
        {
            qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + config.len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + config.len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + 2 * config.len + inds, 0.0) / dtratio[indsx + 1];
            qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + 2 * config.len + inds, 0.0) / dtratio[indsx + 1];
        }
        else
        {
            qK_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + config.len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + config.len + inds, 0.0);
            qR_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0);
        }
    }
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:51 +0200
