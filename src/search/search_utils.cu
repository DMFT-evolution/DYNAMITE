#include "search/search_utils.hpp"
#include "core/config.hpp"
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

// External declaration for global config variable
extern SimulationConfig config;

thrust::device_vector<double> bsearchPosSortedGPU_slow(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    // Output vector
    thrust::device_vector<double> result(elem_size);

    // Step 1: Perform thrust::lower_bound to get insertion indices
    thrust::device_vector<size_t> indices(elem_size);

    thrust::lower_bound(
        thrust::device,
        list.begin(), list.end(),      // search in list
        elem.begin(), elem.end(),      // search values
        indices.begin()                // output positions
    );

    // Step 2: Interpolate between list[m-1] and list[m]
    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(elem_size),
        result.begin(),
        [list_ptr = thrust::raw_pointer_cast(list.data()),
         elem_ptr = thrust::raw_pointer_cast(elem.data()),
         indices_ptr = thrust::raw_pointer_cast(indices.data()),
         list_size] __device__ (size_t i) -> double
        {
            size_t m = indices_ptr[i];
            double El = elem_ptr[i];

            // Handle edge cases
            if (m == 0)
                return 1.0; // El < list[0]
            if (m == list_size)
                return (double)m; // El >= list.back()

            double Lm_1 = list_ptr[m - 1];
            double Lm = list_ptr[m];
            double denom = Lm - Lm_1;

            // If no difference, return m
            if (denom == 0.0)
                return (double)m + 1;

            // Linear interpolation
            return (double)m + (El - Lm_1) / denom;
        }
    );

    return move(result); 
}

__global__ __launch_bounds__(64, 1) void bsearch_interp_kernel(
    const double* __restrict__ list,
    const double* __restrict__ elem,
    double* __restrict__ result,
    size_t list_size,
    size_t elem_size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elem_size) return;

    double El = elem[i]* list[list_size - 1]; // Scale element by the last element of the list

    // Binary search (manual lower_bound)
    size_t left = 0, right = list_size;
    while (left < right) {
        size_t mid = (left + right) / 2;
        if (list[mid] < El)
            left = mid + 1;
        else
            right = mid;
    }

    size_t m = left;

    // Handle edge cases
    if (m == 0) {
        result[i] = 1.0;
        return;
    }
    if (m == list_size) {
        result[i] = (double)m;
        return;
    }

    double Lm_1 = list[m - 1];
    double Lm = list[m];
    double denom = Lm - Lm_1;

    result[i] = (denom == 0.0)
        ? (double)m + 1
        : (double)m + (El - Lm_1) / denom;
}

thrust::device_vector<double> bsearchPosSortedGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    cudaStream_t stream)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    thrust::device_vector<double> result(elem_size);

    int threads = 64;
    int blocks = (elem_size + threads - 1) / threads;

    bsearch_interp_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(list.data()),
        thrust::raw_pointer_cast(elem.data()),
        thrust::raw_pointer_cast(result.data()),
        list_size,
        elem_size
    );

    // cudaDeviceSynchronize(); //removed // Optional, remove if used asynchronously
    return move(result);
}

void bsearchPosSortedGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    thrust::device_vector<double>& result,
    cudaStream_t stream)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    int threads = 64;
    int blocks = (elem_size + threads - 1) / threads;

    bsearch_interp_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(list.data()),
        thrust::raw_pointer_cast(elem.data()),
        thrust::raw_pointer_cast(result.data()),
        list_size,
        elem_size
    );

    // cudaDeviceSynchronize(); //removed // Optional, remove if used asynchronously
}

thrust::device_vector<double> isearchPosSortedInitGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    const thrust::device_vector<double>& inits)
{
    size_t N = elem.size();
    assert(inits.size() == N);
    size_t length = list.size();
    double last = list.back();

    thrust::device_vector<double> result(N);

    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(N),
        result.begin(),
        [list_ptr = thrust::raw_pointer_cast(list.data()),
         elem_ptr = thrust::raw_pointer_cast(elem.data()),
         init_ptr = thrust::raw_pointer_cast(inits.data()),
         length,last] __device__ (size_t i) -> double 
        {
            double El = last * elem_ptr[i];

            size_t n1 = length;
            size_t n0 = static_cast<size_t>(floor(init_ptr[i]));
            size_t m  = min(n0 + 1, length);
            bool even = true;

            double Lm = list_ptr[m - 1];

            if (Lm > El)
                n1 = max(m - 2, size_t(1));

            double l0, l1;
            while (n0 < n1 &&
                   (l0 = list_ptr[n0 - 1]) <= El &&
                   El <= (l1 = list_ptr[n1 - 1]))
            {
                even = !even;
                if (even) {
                    double frac = (El - l0) / (l1 - l0);
                    m = n0 + static_cast<size_t>(round(frac * (n1 - n0)));
                } else {
                    m = (n0 + n1) >> 1;
                }
                Lm = list_ptr[m - 1];
                if (Lm == El) {
                    n0 = m;
                    n1 = m - 1;
                } else if (Lm < El) {
                    n0 = m + 1;
                } else {
                    n1 = m - 1;
                }
            }

            double out;
            if (Lm <= El) {
                if (m == length) {
                    out = static_cast<double>(m);
                } else {
                    out = m + (El - Lm) / (list_ptr[m] - Lm);
                }
            } else {
                if (m > 1) {
                    out = m - (El - Lm) / (list_ptr[m - 2] - Lm);
                } else {
                    out = static_cast<double>(m);
                }
            }

            return move(out);
        }
    );

    return move(result);
}
