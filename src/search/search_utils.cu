#include "search/search_utils.cuh"
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

dmfe::device_vector<double> bsearchPosSortedGPU_slow(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    // Output vector
    dmfe::device_vector<double> result(elem_size);

    // Step 1: Perform thrust::lower_bound to get insertion indices
    dmfe::device_vector<size_t> indices(elem_size);

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

// ============================================================================
// [PERF] Two searches, one launch.
//
// interpolateGPU locates both theta (L elements) and phi2 (L*L elements) in the
// same t1grid, once per Runge-Kutta stage.  The theta search is only L/64 = 8
// blocks, so it is almost entirely launch latency.  The two are merged into one
// launch over the concatenated index space; entry A occupies [0, nA) and entry B
// [nA, nA+nB).
//
// BIT-EXACTNESS: the body below is bsearch_interp_kernel character for character,
// with the same `list` and the same `list_size`; only which (elem, result) pair a
// thread works on changes.  Each output element is produced by the same sequence
// of comparisons and the same final expression.
// ============================================================================
__global__ __launch_bounds__(64, 1) void bsearch_interp_pair_kernel(
    const double* __restrict__ list,
    const double* __restrict__ elemA,
    double* __restrict__ resultA,
    size_t nA,
    const double* __restrict__ elemB,
    double* __restrict__ resultB,
    size_t nB,
    size_t list_size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA + nB) return;

    const double* __restrict__ elem;
    double* __restrict__ result;
    if (i < nA) { elem = elemA; result = resultA; }
    else        { elem = elemB; result = resultB; i -= nA; }

    double El = elem[i] * list[list_size - 1]; // Scale element by the last element of the list

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

void bsearchPosSortedPairGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elemA,
    dmfe::device_vector<double>& resultA,
    const dmfe::device_vector<double>& elemB,
    dmfe::device_vector<double>& resultB,
    cudaStream_t stream)
{
    const size_t nA = elemA.size();
    const size_t nB = elemB.size();
    const int threads = 64;                        // must match bsearch_interp_kernel
    const int blocks = static_cast<int>((nA + nB + threads - 1) / threads);

    bsearch_interp_pair_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(list.data()),
        thrust::raw_pointer_cast(elemA.data()),
        thrust::raw_pointer_cast(resultA.data()), nA,
        thrust::raw_pointer_cast(elemB.data()),
        thrust::raw_pointer_cast(resultB.data()), nB,
        list.size());
}

dmfe::device_vector<double> bsearchPosSortedGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    cudaStream_t stream)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    dmfe::device_vector<double> result(elem_size, 0.0, dmfe::cuda_async_allocator<double>(stream));

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
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    dmfe::device_vector<double>& result,
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

dmfe::device_vector<double> isearchPosSortedInitGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    const dmfe::device_vector<double>& inits)
{
    size_t N = elem.size();
    assert(inits.size() == N);
    size_t length = list.size();
    double last = list.back();

    dmfe::device_vector<double> result(N);

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
