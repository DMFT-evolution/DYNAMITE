//Compile with nvcc -ccbin clang++ --extended-lambda --use_fast_math -Xcompiler "-O3 -march=native -ffast-math" -o minimalStream minimalStream.cu

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <numeric>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

using namespace std;

template <int P>
constexpr double pow_const(double q) {
    return q * pow_const<P - 1>(q);
}

template <>
constexpr double pow_const<0>(double) {
    return 1.0;
}

template <>
constexpr double pow_const<-1>(double) {
    return 0.0;
}

// Device-compatible version of pow_const
template <int P>
__device__ __forceinline__ double pow_const_device(double q) {
    return q * pow_const_device<P - 1>(q);
}

template <>
__device__ __forceinline__ double pow_const_device<0>(double) {
    return 1.0;
}

template <>
__device__ __forceinline__ double pow_const_device<-1>(double) {
    return 0.0;
}

template <typename T>
thrust::device_vector<T> get_slice(const thrust::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    size_t end = start + len;

    if (end > vec.size()) {
        throw std::out_of_range("get_slice: slice out of range");
    }

    return thrust::device_vector<T>(vec.begin() + start, vec.begin() + end);
}

template <typename T>
thrust::device_ptr<T> get_slice_ptr(const thrust::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    if (start + len > vec.size()) {
        throw std::out_of_range("get_slice_ptr: slice out of range");
    }

    // Cast away const to match thrust::device_ptr<T> constructor
    return thrust::device_ptr<T>(
        const_cast<T*>(thrust::raw_pointer_cast(vec.data()) + start)
    );
}
template <typename T>
void set_slice(thrust::device_vector<T>& vec, size_t i, const thrust::device_vector<T>& slice) {
    size_t start = i * slice.size() ;
    size_t end = start + slice.size() ;

    if (end > vec.size()) {
        throw std::out_of_range("set_slice: slice write out of range");
    }

    thrust::copy(slice.begin(), slice.end(), vec.begin() + start);
}

template <typename T>
void set_slice_ptr(thrust::device_vector<T>& vec, size_t i, const thrust::device_ptr<T>& slice, size_t slice_len) {
    size_t start = i * slice_len;
    size_t end = start + slice_len;

    if (end > vec.size()) {
        throw std::out_of_range("set_slice_ptr: slice write out of range");
    }

    thrust::copy(slice, slice + slice_len, vec.begin() + start);
}

__global__ __launch_bounds__(64, 1) void indexMatAllKernel_dynamic(const double* __restrict__ posx,
                                          const size_t* __restrict__ indsy,
                                          const double* __restrict__ weightsy,
                                          const double* __restrict__ dtratio,
                                          double* __restrict__ qK_result,
                                          double* __restrict__ qR_result,
                                          const double* __restrict__ QKv,
                                          const double* __restrict__ QRv,
                                          const double* __restrict__ dQKv,
                                          const double* __restrict__ dQRv,
                                          size_t len,
                                          size_t depth,
                                          size_t t1len,
                                          size_t prod) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    // Clamp index to avoid overflow
    size_t max_indx = (size_t)(posx[prod - 1] - 0.5);
    size_t indsx = max(min((size_t)(posx[j]), max_indx), (size_t)1);

    double inx = posx[j] - indsx;
    double inx2 = inx * inx;
    double inx3 = inx2 * inx;
    size_t inds = (indsx - 1) * len + indsy[j];

    // Load weights into registers
    const double* weights = weightsy + j * depth;

    // Prepare accumulators
    double qK0 = 0.0, qK1 = 0.0, dqK1 = 0.0, dqK2 = 0.0;
    double qR0 = 0.0, qR1 = 0.0, dqR1 = 0.0, dqR2 = 0.0;

    for (size_t d = 0; d < depth; ++d) {
        size_t offset = inds + d;
        size_t offset1 = len + offset;
        size_t offset2 = 2 * len + offset;

        double w = weights[d];

        qK0  += w * QKv[offset];
        qK1  += w * QKv[offset1];
        dqK1 += w * dQKv[offset1];

        qR0  += w * QRv[offset];
        qR1  += w * QRv[offset1];
        dqR1 += w * dQRv[offset1];

        if (indsx >= t1len - 1) continue; // Skip if indsx is out of bounds
        dqK2 += w * dQKv[offset2];
        dqR2 += w * dQRv[offset2];
    }

    if (indsx < t1len - 1) {
        double denom = dtratio[indsx + 1];

        qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * qK0 +
                       (inx - 2 * inx2 + inx3) * dqK1 +
                       (3 * inx2 - 2 * inx3) * qK1 +
                       (-inx2 + inx3) * dqK2 / denom;

        qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * qR0 +
                       (inx - 2 * inx2 + inx3) * dqR1 +
                       (3 * inx2 - 2 * inx3) * qR1 +
                       (-inx2 + inx3) * dqR2 / denom;
    } else {
        qK_result[j] = (1 - inx2) * qK0 +
                       inx2 * qK1 +
                       (inx - inx2) * dqK1;

        qR_result[j] = (1 - inx2) * qR0 +
                       inx2 * qR1 +
                       (inx - inx2) * dqR1;
    }
}

template<int DEPTH>
__global__ __launch_bounds__(64, 1) void indexMatAllKernel_specialized(const double* __restrict__ posx,
                                          const size_t* __restrict__ indsy,
                                          const double* __restrict__ weightsy,
                                          const double* __restrict__ dtratio,
                                          double* __restrict__ qK_result,
                                          double* __restrict__ qR_result,
                                          const double* __restrict__ QKv,
                                          const double* __restrict__ QRv,
                                          const double* __restrict__ dQKv,
                                          const double* __restrict__ dQRv,
                                          size_t len,
                                          size_t t1len,
                                          size_t prod) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    // Clamp index to avoid overflow
    size_t max_indx = (size_t)(posx[prod - 1] - 0.5);
    size_t indsx = max(min((size_t)(posx[j]), max_indx), (size_t)1);

    double inx = posx[j] - indsx;
    double inx2 = inx * inx;
    double inx3 = inx2 * inx;
    size_t inds = (indsx - 1) * len + indsy[j];

    // Load weights into registers
    const double* weights = weightsy + j * DEPTH;

    // Prepare accumulators
    double qK0 = 0.0, qK1 = 0.0, dqK1 = 0.0, dqK2 = 0.0;
    double qR0 = 0.0, qR1 = 0.0, dqR1 = 0.0, dqR2 = 0.0;

    #pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        size_t offset = inds + d;
        size_t offset1 = len + offset;
        size_t offset2 = 2 * len + offset;

        double w = weights[d];

        qK0  += w * QKv[offset];
        qK1  += w * QKv[offset1];
        dqK1 += w * dQKv[offset1];

        qR0  += w * QRv[offset];
        qR1  += w * QRv[offset1];
        dqR1 += w * dQRv[offset1];

        if (indsx >= t1len - 1) continue; // Skip if indsx is out of bounds
        dqK2 += w * dQKv[offset2];
        dqR2 += w * dQRv[offset2];
    }

    if (indsx < t1len - 1) {
        double denom = dtratio[indsx + 1];

        qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * qK0 +
                       (inx - 2 * inx2 + inx3) * dqK1 +
                       (3 * inx2 - 2 * inx3) * qK1 +
                       (-inx2 + inx3) * dqK2 / denom;

        qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * qR0 +
                       (inx - 2 * inx2 + inx3) * dqR1 +
                       (3 * inx2 - 2 * inx3) * qR1 +
                       (-inx2 + inx3) * dqR2 / denom;
    } else {
        qK_result[j] = (1 - inx2) * qK0 +
                       inx2 * qK1 +
                       (inx - inx2) * dqK1;

        qR_result[j] = (1 - inx2) * qR0 +
                       inx2 * qR1 +
                       (inx - inx2) * dqR1;
    }
}

class StreamPool {
    std::vector<cudaStream_t> streams;
public:
    StreamPool(size_t N) {
        streams.resize(N);
        for (auto& s : streams) cudaStreamCreate(&s);
    }
    ~StreamPool() {
        for (auto& s : streams) cudaStreamDestroy(s);
    }
    cudaStream_t operator[](size_t i) const { return streams[i]; }
};

// Eventually it might be better to use a custom vector class that uses thrust::device_vector internally and allows to overload math operators.
class DeviceArray {
public:
    thrust::device_vector<double> data;


    DeviceArray() = default;          // <-- default constructor
    DeviceArray(size_t n) : data(n) {}
    DeviceArray(const thrust::device_vector<double>& v) : data(v) {}
    DeviceArray(const std::vector<double>& host_vec) : data(host_vec) {}

    DeviceArray& operator=(const std::vector<double>& host_vec) {
        data = host_vec;  // thrust::device_vector supports assignment from host vector
        return *this;
    }

    size_t size() const { return data.size(); }

    void resize(size_t n) { data.resize(n); }

    void push_back(const double& value) {
        data.push_back(value);
    }

    void reserve(size_t new_cap) {
        data.reserve(new_cap);
    }

    // Insert at position (pos is iterator on DeviceArray)
    template <typename Iterator>
    void insert(typename thrust::device_vector<double>::iterator pos,
                Iterator first, Iterator last) {
        data.insert(pos, first, last);
    }

    // Iterators
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    double back() const { return data.back(); }

    double operator[](size_t i) const {
        return data[i];  // returns by value, okay
    }

    // Elementwise multiplication
    friend DeviceArray operator*(const DeviceArray& a, const DeviceArray& b) {
        assert(a.size() == b.size());
        DeviceArray result(a.size());
        thrust::transform(
            a.data.begin(), a.data.end(),
            b.data.begin(),
            result.data.begin(),
            thrust::multiplies<double>()
        );
        return move(result);
    }

    // Elementwise addition
    friend DeviceArray operator+(const DeviceArray& a, const DeviceArray& b) {
        assert(a.size() == b.size());
        DeviceArray result(a.size());
        thrust::transform(
            a.data.begin(), a.data.end(),
            b.data.begin(),
            result.data.begin(),
            thrust::plus<double>()
        );
        return move(result);
    }

    // Convert back to raw thrust vector
    const thrust::device_vector<double>& vec() const { return data; }
    thrust::device_vector<double>& vec() { return data; }
};

// Optimized dispatcher class
class IndexMatAllOptimizer {
private:
    static size_t cached_depth;
    static std::function<void(const thrust::device_vector<double>&,
                             const thrust::device_vector<size_t>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             thrust::device_vector<double>&,
                             thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             size_t,
                             cudaStream_t)> cached_kernel;

    template<int DEPTH>
    static void call_specialized(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream)
    {
        size_t prod = indsy.size();
        size_t t1len = dtratio.size();

        int threads = 64;
        int blocks = (prod + threads - 1) / threads;

        indexMatAllKernel_specialized<DEPTH><<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(posx.data()),
            thrust::raw_pointer_cast(indsy.data()),
            thrust::raw_pointer_cast(weightsy.data()),
            thrust::raw_pointer_cast(dtratio.data()),
            thrust::raw_pointer_cast(qK_result.data()),
            thrust::raw_pointer_cast(qR_result.data()),
            thrust::raw_pointer_cast(QKv.data()),
            thrust::raw_pointer_cast(QRv.data()),
            thrust::raw_pointer_cast(dQKv.data()),
            thrust::raw_pointer_cast(dQRv.data()),
            len, t1len, prod
        );
    }

    static void call_dynamic(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream)
    {
        size_t prod = indsy.size();
        size_t depth = weightsy.size() / prod;
        size_t t1len = dtratio.size();

        int threads = 64;
        int blocks = (prod + threads - 1) / threads;

        indexMatAllKernel_dynamic<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(posx.data()),
            thrust::raw_pointer_cast(indsy.data()),
            thrust::raw_pointer_cast(weightsy.data()),
            thrust::raw_pointer_cast(dtratio.data()),
            thrust::raw_pointer_cast(qK_result.data()),
            thrust::raw_pointer_cast(qR_result.data()),
            thrust::raw_pointer_cast(QKv.data()),
            thrust::raw_pointer_cast(QRv.data()),
            thrust::raw_pointer_cast(dQKv.data()),
            thrust::raw_pointer_cast(dQRv.data()),
            len, depth, t1len, prod
        );
    }

public:
    static void setupKernel(size_t depth) {
        if (cached_depth != depth) {
            cached_depth = depth;
            
            switch(depth) {
                case 9:  cached_kernel = call_specialized<9>; break;
                case 11: cached_kernel = call_specialized<11>; break;
                case 13: cached_kernel = call_specialized<13>; break;
                case 15: cached_kernel = call_specialized<15>; break;
                default: cached_kernel = call_dynamic; break;
            }
        }
    }
    
    static void execute(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream = 0)
    {
        cached_kernel(posx, indsy, weightsy, dtratio, qK_result, qR_result, 
                     QKv, QRv, dQKv, dQRv, len, stream);
    }
};

// Static member definitions
size_t IndexMatAllOptimizer::cached_depth = 0;
std::function<void(const thrust::device_vector<double>&,
                  const thrust::device_vector<size_t>&,
                  const thrust::device_vector<double>&,
                  const thrust::device_vector<double>&,
                  thrust::device_vector<double>&,
                  thrust::device_vector<double>&,
                  const thrust::device_vector<double>&,
                  const thrust::device_vector<double>&,
                  const thrust::device_vector<double>&,
                  const thrust::device_vector<double>&,
                  size_t,
                  cudaStream_t)> IndexMatAllOptimizer::cached_kernel;

// Modified indexMatAllGPU function
void indexMatAllGPU(const thrust::device_vector<double>& posx,
                    const thrust::device_vector<size_t>& indsy,
                    const thrust::device_vector<double>& weightsy,
                    const thrust::device_vector<double>& dtratio,
                    thrust::device_vector<double>& qK_result,
                    thrust::device_vector<double>& qR_result,
                    const thrust::device_vector<double>& QKv,
                    const thrust::device_vector<double>& QRv,
                    const thrust::device_vector<double>& dQKv,
                    const thrust::device_vector<double>& dQRv,
                    size_t len, 
                    cudaStream_t stream = 0) 
{
    const size_t depth = weightsy.size() / indsy.size();
    
    // Setup the optimal kernel based on depth
    IndexMatAllOptimizer::setupKernel(depth);
    
    // Execute the cached kernel
    IndexMatAllOptimizer::execute(posx, indsy, weightsy, dtratio, qK_result, qR_result,
                                 QKv, QRv, dQKv, dQRv, len, stream);
}

constexpr int p = 3;
constexpr int p2 = 12;
constexpr double lambda = 0.3;
constexpr double TMCT = 0.805166;
constexpr double T0 = 1e50;
// double T0=1e50;
constexpr double Gamma = 0.0;
constexpr int maxLoop = 1e8;

constexpr double tmax = 1e7; //time to evolve to
constexpr double delta_t_min = 1e-5; //initial and minimal time step
constexpr double delta_max = 1e-10; //maximal error per step
constexpr double rmax[] = {3,13}; // stability range of SSPRK(10,4)

double delta;
double delta_old;
int loop;
double specRad;
double delta_t;
size_t len = 512;
int ord;
bool gpu = false; // Use GPU if true, CPU if false

vector<double> theta, phi1, phi2, posA1y, posA2y, posB2y, weightsA1y, weightsA2y, weightsB2y, posB1xOld, posB2xOld, integ;
vector<size_t> indsA1y, indsA2y, indsB2y;

vector<double> t1grid, delta_t_ratio;

vector<double> QKv, QRv, dQKv, dQRv, rInt, drInt, rvec, drvec;

vector<double> SigmaKA1int, SigmaRA1int, SigmaKB1int, SigmaRB1int, SigmaKA2int, SigmaRA2int, SigmaKB2int, SigmaRB2int;
vector<double> QKA1int, QRA1int, QKB1int, QRB1int, QKA2int, QRA2int, QKB2int, QRB2int;

__constant__ int d_p;
__constant__ int d_p2;
__constant__ double d_lambda;

// Device pointers for GPU memory
// double *d_theta, *d_phi1, *d_phi2, *d_posA1y, *d_posA2y, *d_posB2y, *d_weightsA1y, *d_weightsA2y, *d_weightsB2y, *d_posB1xOld, *d_posB2xOld, *d_integ;
// size_t *d_indsA1y, *d_indsA2y, *d_indsB2y;

// double *d_t1grid, *d_delta_t_ratio;

// double *d_QKv, *d_QRv, *d_dQKv, *d_dQRv, *d_rInt, *d_drInt, *d_rvec, *d_drvec;

// double *d_SigmaKA1int, *d_SigmaRA1int, *d_SigmaKB1int, *d_SigmaRB1int, *d_SigmaKA2int, *d_SigmaRA2int, *d_SigmaKB2int, *d_SigmaRB2int;
// double *d_QKA1int, *d_QRA1int, *d_QKB1int, *d_QRB1int, *d_QKA2int, *d_QRA2int, *d_QKB2int, *d_QRB2int;

struct SimulationData {
    thrust::device_vector<double> d_theta, d_phi1, d_phi2, d_posA1y, d_posA2y, d_posB2y, d_weightsA1y, d_weightsA2y, d_weightsB2y, d_posB1xOld, d_posB2xOld, d_integ;
    thrust::device_vector<size_t> d_indsA1y, d_indsA2y, d_indsB2y;

    thrust::device_vector<double> d_t1grid, d_delta_t_ratio;

    thrust::device_vector<double> d_QKv, d_QRv, d_dQKv, d_dQRv, d_rInt, d_drInt, d_rvec, d_drvec;

    thrust::device_vector<double> d_SigmaKA1int, d_SigmaRA1int, d_SigmaKB1int, d_SigmaRB1int, d_SigmaKA2int, d_SigmaRA2int, d_SigmaKB2int, d_SigmaRB2int;
    thrust::device_vector<double> d_QKA1int, d_QRA1int, d_QKB1int, d_QRB1int, d_QKA2int, d_QRA2int, d_QKB2int, d_QRB2int;
    thrust::device_vector<double> convA1_1, convA2_1, convA1_2, convA2_2, convR_1, convR_2, convR_3, convR_4;

    thrust::device_vector<double> temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

    thrust::device_vector<size_t> Stemp0, Stemp1;

    thrust::device_vector<double> error_result;
};

struct RKData {
    size_t stages = 10;
    size_t posCount = 3;
    int init = 0; // Flag to indicate which RK data is initialized
    double *avec, *bvec, *b2vec, *cvec;
    double gt, gtfinal, gte, ht, gt0;
    thrust::device_vector<double> gK, gR, gRfinal, gKfinal, gKe, gRe, gK0, gR0;
    thrust::device_vector<double> posB1xvec, posB2xvec;
    thrust::device_vector<double> hK,hR, hK0, hR0, d_avec;
};

SimulationData* sim = nullptr;
RKData* rk = nullptr;

inline std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vectors must be the same size for += operation.");
    }

    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<double>());
    return lhs;
}

// Overload the + operator for std::vector<double>
inline vector<double> operator+(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for addition.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::plus<>());

    return move(result);
}

inline vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for addition.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::minus<>());

    return move(result);
}

// Overload the * operator for the element-wise product of two vectors
inline vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for element-wise multiplication.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise multiplication
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::multiplies<>());

    return move(result);
}

// Overload the * operator for the product of a vector and a scalar
inline vector<double> operator*(const vector<double>& vec, double scalar) {
    vector<double> result(vec.size());

    // Use std::transform to multiply each element by the scalar
    std::transform(vec.begin(), vec.end(), result.begin(), [scalar](double val) { return val * scalar; });

    return move(result);
}

// Overload the * operator for the product of a scalar and a vector (commutative)
inline vector<double> operator*(double scalar, const vector<double>& vec) {
    return vec * scalar; // Reuse the previous operator
}

__device__ __host__ inline size_t max_device(size_t a, size_t b) {
    return (a > b) ? a : b;
}

__device__ __host__ inline double max_device(double a, double b) {
    return (a > b) ? a : b;
}

__device__ __host__ inline size_t min_device(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__device__ __host__ inline double min_device(double a, double b) {
    return (a < b) ? a : b;
}

void Product(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
}

thrust::device_vector<double> ProductGPU(
    const thrust::device_vector<double>& a,
    const thrust::device_vector<double>& b)
{
    assert(a.size() == b.size());
    thrust::device_vector<double> result(a.size());

    thrust::transform(
        a.begin(), a.end(),
        b.begin(),
        result.begin(),
        thrust::multiplies<double>()
    );
    return move(result);
}

void Sum(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}


// Element-wise addition of two thrust::device_vector<double>
thrust::device_vector<double> SumGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b) {
    assert(a.size() == b.size());
    thrust::device_vector<double> result(a.size());
    thrust::transform(
        a.begin(), a.end(),
        b.begin(),
        result.begin(),
        thrust::plus<double>()
    );
    return move(result);
}

void SumGPU(const thrust::device_vector<double>& a,
            const thrust::device_vector<double>& b,
            thrust::device_vector<double>& out) {
    assert(a.size() == b.size() && out.size() == a.size());
    thrust::transform(
        a.begin(), a.end(),
        b.begin(),
        out.begin(),
        thrust::plus<double>()
    );
}

thrust::device_vector<double> MAGPU(
    thrust::device_ptr<const double> a,
    thrust::device_ptr<const double> b,
    double c,
    size_t len)
{
    thrust::device_vector<double> result(len);

    thrust::transform(
        thrust::device,
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(len),
        result.begin(),
        [a, b, c] __device__ (size_t i) {
            return a[i] + b[i] * c;
        });
    
    return move(result);
}

void MAGPU_ptr(
    thrust::device_ptr<const double> a,
    thrust::device_ptr<const double> b,
    double c,
    thrust::device_vector<double>& result,
    size_t len)
{
    assert(result.size() >= len);

    thrust::transform(
        thrust::device,
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(len),
        result.begin(),
        [a, b, c] __device__ (size_t i) {
            return a[i] + b[i] * c;
        });
}

void Subtract(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }


    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
}

// Element-wise subtraction of two thrust::device_vector<double>
thrust::device_vector<double> SubtractGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b) {
    assert(a.size() == b.size());
    thrust::device_vector<double> result(a.size());
    thrust::transform(
        a.begin(), a.end(),
        b.begin(),
        result.begin(),
        thrust::minus<double>()
    );
    return move(result);
}

void scaleVec(const vector<double>& vec1, double real, vector<double>& result) {
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * real;
    }
}

// Example: Multiply a thrust::device_vector by a scalar
thrust::device_vector<double> scalarMultiply(const thrust::device_vector<double>& vec, double scalar) {
    thrust::device_vector<double> result(vec.size());
    thrust::transform(
        vec.begin(), vec.end(),
        result.begin(),
        [scalar] __device__(double x) { return x * scalar; }
    );
    return move(result);
}

// Example: Multiply a thrust::device_vector by a scalar
thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& vec, double scalar, size_t len) {
    thrust::device_vector<double> result(len);
    thrust::transform(
        vec, vec + len,
        result.begin(),
        [scalar] __device__(double x) { return x * scalar; }
    );
    return move(result);
}

__global__ void update_gK_gR(
    double* gK, const double* gKfinal, const double* gK0,
    double* gR, const double* gRfinal, const double* gR0,
    int N
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        gK[i] += gK0[i] - gKfinal[i];
        gR[i] += gR0[i] - gRfinal[i];
    }
}

void AddSubtractGPU(thrust::device_vector<double>& gK,
                  const thrust::device_vector<double>& gKfinal,
                  const thrust::device_vector<double>& gK0,
                  thrust::device_vector<double>& gR,
                  const thrust::device_vector<double>& gRfinal,
                  const thrust::device_vector<double>& gR0,
                  cudaStream_t stream = 0)
{
    const int N = gK.size();
    const int blockSize = 128;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    auto d_gK = thrust::raw_pointer_cast(gK.data());
    auto d_gKf = thrust::raw_pointer_cast(gKfinal.data());
    auto d_gK0 = thrust::raw_pointer_cast(gK0.data());
    auto d_gR = thrust::raw_pointer_cast(gR.data());
    auto d_gRf = thrust::raw_pointer_cast(gRfinal.data());
    auto d_gR0 = thrust::raw_pointer_cast(gR0.data());

    update_gK_gR<<<numBlocks, blockSize, 0, stream>>>(d_gK, d_gKf, d_gK0, d_gR, d_gRf, d_gR0, N);
}

StreamPool& getDefaultStreamPool() {
    static StreamPool defaultPool(10);  // Constructed once, reused
    return defaultPool;
}

void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Vectors must be of the same length.\n";
        std::cerr << "Size of vector a: " << a.size() << ", Size of vector b: " << b.size() << "\n";
        return;
    }

    double diff = 0.0;
    std::cout << "Differences between vectors:\n";
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::abs(a[i] - b[i]);
        std::cout << "Index " << i << ": |" << a[i] << " - " << b[i] << "| = " << diff << "\n";
    }

    diff=0.0;
    std::cout << "Total Differences between vectors: ";
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::abs(a[i] - b[i]);
    }
    std::cout << diff << "\n";
}

__global__ void FusedUpdateKernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ extra1,  // can be nullptr
    const double* __restrict__ extra2,  // can be nullptr
    const double* __restrict__ extra3,  // can be nullptr
    const double* __restrict__ delta, // can be nullptr
    const double* __restrict__ subtract, // can be nullptr
    double* __restrict__ out,
    const double* alpha,
    const double* beta,
    size_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double result = alpha[0] * a[i] + beta[0] * b[i];

    if (extra1) result += extra1[i];
    if (extra2) result += extra2[i];
    if (extra3) result += extra3[i];
    if (subtract) result -= delta[i] * subtract[i];

    out[i] = result;
}

void FusedUpdate(
    const thrust::device_ptr<double>& a,
    const thrust::device_ptr<double>& b,
    const thrust::device_vector<double>& out,
    const double* alpha,
    const double* beta,
    const thrust::device_vector<double>* delta = nullptr,
    const thrust::device_vector<double>* extra1 = nullptr,
    const thrust::device_vector<double>* extra2 = nullptr,
    const thrust::device_vector<double>* extra3 = nullptr,
    const thrust::device_ptr<double>& subtract = nullptr,
    cudaStream_t stream = 0) {
    size_t N = out.size();

    int threads = 64;
    int blocks = (N + threads - 1) / threads;

    FusedUpdateKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(a),
        thrust::raw_pointer_cast(b),
        extra1 ? thrust::raw_pointer_cast(extra1->data()) : nullptr,
        extra2 ? thrust::raw_pointer_cast(extra2->data()) : nullptr,
        extra3 ? thrust::raw_pointer_cast(extra3->data()) : nullptr,
        delta ? thrust::raw_pointer_cast(delta->data()) : nullptr,
        subtract ? thrust::raw_pointer_cast(subtract) : nullptr,
        thrust::raw_pointer_cast(const_cast<thrust::device_vector<double>&>(out).data()),
        alpha, beta, N
    );
    // cudaDeviceSynchronize();
}

__global__ void FusedQRKernel(
    const double* __restrict__ qR,
    const double* __restrict__ theta,
    const double* __restrict__ conv1,
    const double* __restrict__ conv2,
    const double* __restrict__ r,
    double* __restrict__ out,
    size_t len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double rEnd = r[len - 1];
    if (i < len) {
        out[i] = conv1[i] - qR[i] * rEnd + theta[i] * (qR[i] * r[i] - conv2[i]);
    }
}

bool isCompatibleGPUInstalled() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return false;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        // Check compute capability
        if (deviceProp.major > 8 || (deviceProp.major == 8 && deviceProp.minor >= 6)) {
            std::cout << "Compatible GPU found: " << deviceProp.name << std::endl;
            return true;
        }
    }

    std::cerr << "No compatible GPU found (compute capability >= 8.6)." << std::endl;
    return false;
}

void QRstepFused(const thrust::device_ptr<double>& qR,
                 const thrust::device_vector<double>& theta,
                 const thrust::device_vector<double>& conv1,
                 const thrust::device_vector<double>& conv2,
                 const thrust::device_vector<double>& r,
                 double* out,
                 cudaStream_t stream = 0) {
    size_t len = theta.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    FusedQRKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(conv1.data()),
        thrust::raw_pointer_cast(conv2.data()),
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(out), len
    );
    // cudaDeviceSynchronize();
}

auto absDiff = [] __host__ __device__ (const thrust::tuple<double, double>& tup) {
    return fabs(thrust::get<0>(tup) - thrust::get<1>(tup));
};

struct AbsDiff {
    __host__ __device__
    double operator()(const thrust::tuple<double, double>& tup) const {
        return fabs(thrust::get<0>(tup) - thrust::get<1>(tup));
    }
};

vector<double> getLastLenEntries(const vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw invalid_argument("len is greater than the size of the vector.");
    }
    return vector<double>(vec.end() - len, vec.end());
}

thrust::device_vector<double> getLastLenEntriesGPU(const thrust::device_vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw std::invalid_argument("len is greater than the size of the vector.");
    }
    return thrust::device_vector<double>(vec.end() - len, vec.end());
}

vector<double> importVectorFromFile(const string& filename) {
    vector<double> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    double value;
    while (file >> value) {
        data.push_back(value);
        // Skip tabs or newlines automatically handled by `>>`
    }

    file.close();
    return data;
}

vector<size_t> importIntVectorFromFile(const string& filename) {
    vector<size_t> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    size_t value;
    while (file >> value) {
        data.push_back(value);
        // Skip tabs or newlines automatically handled by `>>`
    }

    file.close();
    return data;
}

void import()
{

    std::ostringstream basePath;
    basePath << "Grid_data/" << len << "/";
    std::string prefix = basePath.str();
    theta       = importVectorFromFile(prefix + "theta.dat");
    phi1        = importVectorFromFile(prefix + "phi1.dat");
    phi2        = importVectorFromFile(prefix + "phi2.dat");
    posA1y      = importVectorFromFile(prefix + "posA1y.dat");
    posA2y      = importVectorFromFile(prefix + "posA2y.dat");
    posB2y      = importVectorFromFile(prefix + "posB2y.dat");
    indsA1y     = importIntVectorFromFile(prefix + "indsA1y.dat");
    indsA2y     = importIntVectorFromFile(prefix + "indsA2y.dat");
    indsB2y     = importIntVectorFromFile(prefix + "indsB2y.dat");
    weightsA1y  = importVectorFromFile(prefix + "weightsA1y.dat");
    weightsA2y  = importVectorFromFile(prefix + "weightsA2y.dat");
    weightsB2y  = importVectorFromFile(prefix + "weightsB2y.dat");
    integ       = importVectorFromFile(prefix + "int.dat");
    
    ord = weightsB2y.size() / (len * len) - 2;
}

// Function to copy vectors to the GPU
void copyVectorsToGPU() {

    sim->d_theta = theta; // Copying to thrust::device_vector
    sim->d_phi1 = phi1;   
    sim->d_phi2 = phi2;   
    sim->d_posA1y = posA1y; 
    sim->d_posA2y = posA2y; 
    sim->d_posB2y = posB2y; 
    sim->d_indsA1y = indsA1y; 
    sim->d_indsA2y = indsA2y; 
    sim->d_indsB2y = indsB2y; 
    sim->d_weightsA1y = weightsA1y; 
    sim->d_weightsA2y = weightsA2y; 
    sim->d_weightsB2y = weightsB2y; 
    sim->d_integ = integ; 

    sim->d_posB1xOld = posB1xOld; 
    sim->d_posB2xOld = posB2xOld; 

    sim->d_SigmaKA1int = SigmaKA1int; 
    sim->d_SigmaRA1int = SigmaRA1int; 
    sim->d_SigmaKB1int = SigmaKB1int; 
    sim->d_SigmaRB1int = SigmaRB1int; 
    sim->d_SigmaKA2int = SigmaKA2int; 
    sim->d_SigmaRA2int = SigmaRA2int; 
    sim->d_SigmaKB2int = SigmaKB2int; 
    sim->d_SigmaRB2int = SigmaRB2int; 

    sim->d_QKA1int = QKA1int; 
    sim->d_QRA1int = QRA1int; 
    sim->d_QKB1int = QKB1int; 
    sim->d_QRB1int = QRB1int; 
    sim->d_QKA2int = QKA2int; 
    sim->d_QRA2int = QRA2int; 
    sim->d_QKB2int = QKB2int; 
    sim->d_QRB2int = QRB2int; 

    sim->d_QKv = QKv; 
    sim->d_QRv = QRv; 
    sim->d_dQKv = dQKv; 
    sim->d_dQRv = dQRv; 
    sim->d_rvec = rvec; 
    sim->d_drvec = drvec; 

    sim->d_rInt = rInt; 
    sim->d_drInt = drInt; 
    sim->d_t1grid = t1grid; 
    sim->d_delta_t_ratio = delta_t_ratio; 

    sim->convA1_1.resize(len);
    sim->convA2_1.resize(len);
    sim->convA1_2.resize(len);
    sim->convA2_2.resize(len);
    sim->convR_1.resize(len);
    sim->convR_2.resize(len);
    sim->convR_3.resize(len);
    sim->convR_4.resize(len);

    sim->temp0.resize(len);
    sim->temp1.resize(len);
    sim->temp2.resize(len);
    sim->temp3.resize(len);
    sim->temp4.resize(len);
    sim->temp5.resize(len);
    sim->temp6.resize(len);
    sim->temp7.resize(len);
    sim->temp8.resize(len);
    sim->temp9.resize(len);

    sim->Stemp0.resize(len);
    sim->Stemp1.resize(len);

    sim->error_result.resize(1, 0.0);

    std::cout << "All vectors copied to GPU memory." << std::endl;
}

void copyParametersToDevice(int p_host, int p2_host, double lambda_host) {
    cudaMemcpyToSymbol(d_p, &p_host, sizeof(int));
    cudaMemcpyToSymbol(d_p2, &p2_host, sizeof(int));
    cudaMemcpyToSymbol(d_lambda, &lambda_host, sizeof(double));
}

void copyVectorsToCPU() {
    QKv.resize(sim->d_QKv.size());
    thrust::copy(sim->d_QKv.begin(), sim->d_QKv.end(), QKv.begin());
    QRv.resize(sim->d_QRv.size());
    thrust::copy(sim->d_QRv.begin(), sim->d_QRv.end(), QRv.begin());
    dQKv.resize(sim->d_dQKv.size());
    thrust::copy(sim->d_dQKv.begin(), sim->d_dQKv.end(), dQKv.begin());
    dQRv.resize(sim->d_dQRv.size());
    thrust::copy(sim->d_dQRv.begin(), sim->d_dQRv.end(), dQRv.begin());
    rvec.resize(sim->d_rvec.size());
    thrust::copy(sim->d_rvec.begin(), sim->d_rvec.end(), rvec.begin());
    drvec.resize(sim->d_drvec.size());
    thrust::copy(sim->d_drvec.begin(), sim->d_drvec.end(), drvec.begin());
    rInt.resize(sim->d_rInt.size());
    thrust::copy(sim->d_rInt.begin(), sim->d_rInt.end(), rInt.begin());
    drInt.resize(sim->d_drInt.size());
    thrust::copy(sim->d_drInt.begin(), sim->d_drInt.end(), drInt.begin());
    t1grid.resize(sim->d_t1grid.size());
    thrust::copy(sim->d_t1grid.begin(), sim->d_t1grid.end(), t1grid.begin());
    delta_t_ratio.resize(sim->d_delta_t_ratio.size());
    thrust::copy(sim->d_delta_t_ratio.begin(), sim->d_delta_t_ratio.end(), delta_t_ratio.begin());

    // Stuff below this line is not necessary and can be remove after testing.
    SigmaKA1int.resize(sim->d_SigmaKA1int.size());
    thrust::copy(sim->d_SigmaKA1int.begin(), sim->d_SigmaKA1int.end(), SigmaKA1int.begin());
    SigmaRA1int.resize(sim->d_SigmaRA1int.size());
    thrust::copy(sim->d_SigmaRA1int.begin(), sim->d_SigmaRA1int.end(), SigmaRA1int.begin());
    SigmaKB1int.resize(sim->d_SigmaKB1int.size());
    thrust::copy(sim->d_SigmaKB1int.begin(), sim->d_SigmaKB1int.end(), SigmaKB1int.begin());
    SigmaRB1int.resize(sim->d_SigmaRB1int.size());
    thrust::copy(sim->d_SigmaRB1int.begin(), sim->d_SigmaRB1int.end(), SigmaRB1int.begin());
    SigmaKA2int.resize(sim->d_SigmaKA2int.size());
    thrust::copy(sim->d_SigmaKA2int.begin(), sim->d_SigmaKA2int.end(), SigmaKA2int.begin());
    SigmaRA2int.resize(sim->d_SigmaRA2int.size());
    thrust::copy(sim->d_SigmaRA2int.begin(), sim->d_SigmaRA2int.end(), SigmaRA2int.begin());
    SigmaKB2int.resize(sim->d_SigmaKB2int.size());
    thrust::copy(sim->d_SigmaKB2int.begin(), sim->d_SigmaKB2int.end(), SigmaKB2int.begin());
    SigmaRB2int.resize(sim->d_SigmaRB2int.size());
    thrust::copy(sim->d_SigmaRB2int.begin(), sim->d_SigmaRB2int.end(), SigmaRB2int.begin());

    QKA1int.resize(sim->d_QKA1int.size());
    thrust::copy(sim->d_QKA1int.begin(), sim->d_QKA1int.end(), QKA1int.begin());
    QRA1int.resize(sim->d_QRA1int.size());
    thrust::copy(sim->d_QRA1int.begin(), sim->d_QRA1int.end(), QRA1int.begin());
    QKB1int.resize(sim->d_QKB1int.size());
    thrust::copy(sim->d_QKB1int.begin(), sim->d_QKB1int.end(), QKB1int.begin());
    QRB1int.resize(sim->d_QRB1int.size());
    thrust::copy(sim->d_QRB1int.begin(), sim->d_QRB1int.end(), QRB1int.begin());
    QKA2int.resize(sim->d_QKA2int.size());
    thrust::copy(sim->d_QKA2int.begin(), sim->d_QKA2int.end(), QKA2int.begin());
    QRA2int.resize(sim->d_QRA2int.size());
    thrust::copy(sim->d_QRA2int.begin(), sim->d_QRA2int.end(), QRA2int.begin());
    QKB2int.resize(sim->d_QKB2int.size());
    thrust::copy(sim->d_QKB2int.begin(), sim->d_QKB2int.end(), QKB2int.begin());
    QRB2int.resize(sim->d_QRB2int.size());
    thrust::copy(sim->d_QRB2int.begin(), sim->d_QRB2int.end(), QRB2int.begin());

    posA1y.resize(sim->d_posA1y.size());
    thrust::copy(sim->d_posA1y.begin(), sim->d_posA1y.end(), posA1y.begin());
    posA2y.resize(sim->d_posA2y.size());
    thrust::copy(sim->d_posA2y.begin(), sim->d_posA2y.end(), posA2y.begin());
    posB2y.resize(sim->d_posB2y.size());
    thrust::copy(sim->d_posB2y.begin(), sim->d_posB2y.end(), posB2y.begin());
    posB1xOld.resize(sim->d_posB1xOld.size());
    thrust::copy(sim->d_posB1xOld.begin(), sim->d_posB1xOld.end(), posB1xOld.begin());
    posB2xOld.resize(sim->d_posB2xOld.size());
    thrust::copy(sim->d_posB2xOld.begin(), sim->d_posB2xOld.end(), posB2xOld.begin());
}

void clearAllVectors() {
    sim->d_QKv.clear();
    sim->d_QKv.shrink_to_fit();
    sim->d_QRv.clear();
    sim->d_QRv.shrink_to_fit();
    sim->d_dQKv.clear();
    sim->d_dQKv.shrink_to_fit();
    sim->d_dQRv.clear();
    sim->d_dQRv.shrink_to_fit();

    sim->d_rInt.clear();
    sim->d_rInt.shrink_to_fit();
    sim->d_drInt.clear();
    sim->d_drInt.shrink_to_fit();
    sim->d_rvec.clear();
    sim->d_rvec.shrink_to_fit();
    sim->d_drvec.clear();
    sim->d_drvec.shrink_to_fit();

    sim->d_SigmaKA1int.clear();
    sim->d_SigmaKA1int.shrink_to_fit();
    sim->d_SigmaRA1int.clear();
    sim->d_SigmaRA1int.shrink_to_fit();
    sim->d_SigmaKB1int.clear();
    sim->d_SigmaKB1int.shrink_to_fit();
    sim->d_SigmaRB1int.clear();
    sim->d_SigmaRB1int.shrink_to_fit();
    sim->d_SigmaKA2int.clear();
    sim->d_SigmaKA2int.shrink_to_fit();
    sim->d_SigmaRA2int.clear();
    sim->d_SigmaRA2int.shrink_to_fit();
    sim->d_SigmaKB2int.clear();
    sim->d_SigmaKB2int.shrink_to_fit();
    sim->d_SigmaRB2int.clear();
    sim->d_SigmaRB2int.shrink_to_fit();

    sim->d_QKA1int.clear();
    sim->d_QKA1int.shrink_to_fit();
    sim->d_QRA1int.clear();
    sim->d_QRA1int.shrink_to_fit();
    sim->d_QKB1int.clear();
    sim->d_QKB1int.shrink_to_fit();
    sim->d_QRB1int.clear();
    sim->d_QRB1int.shrink_to_fit();
    sim->d_QKA2int.clear();
    sim->d_QKA2int.shrink_to_fit();
    sim->d_QRA2int.clear();
    sim->d_QRA2int.shrink_to_fit();
    sim->d_QKB2int.clear();
    sim->d_QKB2int.shrink_to_fit();
    sim->d_QRB2int.clear();
    sim->d_QRB2int.shrink_to_fit();

    sim->d_theta.clear();
    sim->d_theta.shrink_to_fit();
    sim->d_phi1.clear();
    sim->d_phi1.shrink_to_fit();
    sim->d_phi2.clear();
    sim->d_phi2.shrink_to_fit();

    sim->d_posA1y.clear();
    sim->d_posA1y.shrink_to_fit();
    sim->d_posA2y.clear();
    sim->d_posA2y.shrink_to_fit();
    sim->d_posB2y.clear();
    sim->d_posB2y.shrink_to_fit();

    sim->d_weightsA1y.clear();
    sim->d_weightsA1y.shrink_to_fit();
    sim->d_weightsA2y.clear();
    sim->d_weightsA2y.shrink_to_fit();
    sim->d_weightsB2y.clear();
    sim->d_weightsB2y.shrink_to_fit();

    sim->d_posB1xOld.clear();
    sim->d_posB1xOld.shrink_to_fit();
    sim->d_posB2xOld.clear();
    sim->d_posB2xOld.shrink_to_fit();

    sim->d_indsA1y.clear();
    sim->d_indsA1y.shrink_to_fit();
    sim->d_indsA2y.clear();
    sim->d_indsA2y.shrink_to_fit();
    sim->d_indsB2y.clear();
    sim->d_indsB2y.shrink_to_fit();

    sim->d_integ.clear();
    sim->d_integ.shrink_to_fit();
    sim->d_t1grid.clear();
    sim->d_t1grid.shrink_to_fit();
    sim->d_delta_t_ratio.clear();
    sim->d_delta_t_ratio.shrink_to_fit();
}

// Example: Copy a std::vector<double> to device memory
double* copyVectorToDevice(const std::vector<double>& host_vec) {
    double* device_ptr = nullptr;
    size_t bytes = host_vec.size() * sizeof(double);
    cudaMalloc(&device_ptr, bytes);
    cudaMemcpy(device_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice);
    return device_ptr;
}

// Example: Copy a std::vector<size_t> to device memory
size_t* copyVectorToDevice(const std::vector<size_t>& host_vec) {
    size_t* device_ptr = nullptr;
    size_t bytes = host_vec.size() * sizeof(size_t);
    cudaMalloc(&device_ptr, bytes);
    cudaMemcpy(device_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice);
    return device_ptr;
}

__device__ __forceinline__ double fast_pow_int(double base, int exp) {
    if (exp == 0) return 1.0;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;
    if (exp == 4) { double sq = base * base; return sq * sq; }
    if (exp == 5) { double sq = base * base; return sq * sq * base; }
    if (exp == 6) { double sq = base * base; return sq * sq * sq; }
    if (exp == 7) { double sq = base * base; double cu = sq * base; return cu * cu * base; }
    if (exp == 8) { double sq = base * base; double qu = sq * sq; return qu * qu; }
    if (exp == 9) { double cu = base * base * base; return cu * cu * cu; }
    if (exp == 10) { double sq = base * base; double qu = sq * sq; return qu * qu * sq; }
    if (exp == 11) { double sq = base * base; double qu = sq * sq; return qu * qu * sq * base; }
    if (exp == 12) { double cu = base * base * base; double si = cu * cu; return si * si; }
    if (exp == 13) { double cu = base * base * base; double si = cu * cu; return si * si * base; }
    if (exp == 14) { double sq = base * base; double qu = sq * sq; double oc = qu * qu; return oc * oc * sq; }
    
    // For larger exponents, use iterative approach
    double result = 1.0;
    double current = base;
    while (exp > 0) {
        if (exp & 1) result *= current;
        current *= current;
        exp >>= 1;
    }
    return result;
}

inline double flambda(const double q)
{
    return lambda * pow_const<p>(q) + (1 - lambda) * pow_const<p2>(q);
}

inline double Dflambda(const double q)
{
    return lambda * p * pow_const<p - 1>(q) + (1 - lambda) * p2 * pow_const<p2 - 1>(q);
}

inline double DDflambda(const double q)
{
    return lambda * p * (p - 1) * pow_const<p - 2>(q) + (1 - lambda) * p2 * (p2 - 1) * pow_const<p2 - 2>(q);
}

inline double DDDflambda(const double q)
{
    return lambda * p * (p - 1) * (p - 2) * pow_const<p - 3>(q) + (1 - lambda) * p2 * (p2 - 1) * (p2 - 2) * pow_const<p2 - 3>(q);
}

__device__ double flambdaGPU(const double q)
{
    return lambda * pow_const_device<p>(q) + (1 - lambda) * pow_const_device<p2>(q);
}

__device__ double DflambdaGPU(const double q) 
{
    return lambda * p * pow_const_device<p - 1>(q) + (1 - lambda) * p2 * pow_const_device<p2 - 1>(q);
}

__device__ double DflambdaGPU2(const double q) 
{
    double term1 = 0.0, term2 = 0.0;
    
    if (d_p > 0) {
        term1 = d_lambda * d_p * fast_pow_int(q, d_p - 1);
    }
    
    if (d_p2 > 0) {
        term2 = (1 - d_lambda) * d_p2 * fast_pow_int(q, d_p2 - 1);
    }
    
    return term1 + term2;
}

__device__ double DDflambdaGPU(const double q) 
{
    return lambda * p * (p - 1) * pow_const_device<p - 2>(q) + (1 - lambda) * p2 * (p2 - 1) * pow_const_device<p2 - 2>(q);
}

__device__ double DDDflambdaGPU(const double q) 
{
    return lambda * p * (p - 1) * (p - 2) * pow_const_device<p - 3>(q) + (1 - lambda) * p2 * (p2 - 1) * (p2 - 2) * pow_const_device<p2 - 3>(q);
}

__global__ void computeScale(double* temp2, double T0) {
    temp2[0] = temp2[0] / T0;
}

__global__ void computeProduct(double* a, const double* b, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] *= b[idx];
    }
}

__global__ void computeSum(double* a, const double* b, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] += b[idx];
    }
}

__global__ void computeSum(const double* a, const double* b, double* c, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void computeMA(double* a, const double* b, double c, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] += b[idx] * c;
    }
}

__global__ void computeWeightedSum(
    const double* __restrict__ gK0, 
    const double* __restrict__ hK,  // Flattened array hK_{i,j}
    const double* __restrict__ a,  // Coefficients a_i
    double* __restrict__ result,  // Output array for the sum over j
    double dt,  // Time step
    int n,  // Number of rows
    size_t len                     // Number of columns (j)
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    if (j >= len) return;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {  // Iterate over rows (i runs from 0 to 5)
        sum += dt * a[i] * hK[i * len + j];
    }
    result[j] = gK0[j] + sum;  // Store the result for column j
}

__global__ void computeSigmaKandRKernel(
    const double* __restrict__ qK,
    const double* __restrict__ qR,
    double* __restrict__ sigmaK,
    double* __restrict__ sigmaR,
    size_t len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        double qk = qK[i];
        double qr = qR[i];

        sigmaK[i] = DflambdaGPU(qk);
        sigmaR[i] = DDflambdaGPU(qk) * qr;
    }
}

__global__ void computeCopy(
    const double* __restrict__ src,
    double* __restrict__ dest,
    size_t offset,
    size_t len,
    double factor = 1.0)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        dest[offset + i] = factor * src[i];
    }
}

__global__ void computeError(
    const double* __restrict__ gKfinal,
    const double* __restrict__ gKe,
    const double* __restrict__ gRfinal,
    const double* __restrict__ gRe,
    double* __restrict__ result,
    size_t len)
{
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_sum = 0.0;
    
    // Process both K and R errors in one pass
    if (i < len) {
        local_sum += fabs(gKfinal[i] - gKe[i]);
        local_sum += fabs(gRfinal[i] - gRe[i]);
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// thrust::device_vector<double> ProductGPU(
//     const thrust::device_vector<double>& a,
//     const thrust::device_vector<double>& b)
// {
//     assert(a.size() == b.size());
//     thrust::device_vector<double> result(a.size());

//     thrust::transform(
//         a.begin(), a.end(),
//         b.begin(),
//         result.begin(),
//         thrust::multiplies<double>()
//     );
//     return move(result);
// }

void indexVecLN3(const vector<double>& __restrict weights, const vector<size_t>& __restrict inds,
                 vector<double>& __restrict qk_result, vector<double>& __restrict qr_result) {
    size_t prod = inds.size();
    size_t length = QKv.size() - len;
    size_t depth = weights.size() / prod;
    const double* QK_start = &QKv[length];
    const double* QR_start = &QRv[length];


    for (size_t j = 0; j < prod; j++) {
        const double* weights_start = &weights[depth * j];
        qk_result[j] = std::inner_product(weights_start, weights_start + depth, QK_start + inds[j], 0.0);
        qr_result[j] = std::inner_product(weights_start, weights_start + depth, QR_start + inds[j], 0.0);
    }
}

template <int DEPTH>
__global__ __launch_bounds__(64, 1)  // 256 threads/block, allow ~2 blocks/SM
void indexVecLN3Kernel_specialized(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t depth,
    size_t offset,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t index = inds[j];
    const double* w = weights + j * depth;

    double qk = 0.0, qr = 0.0;
    for (int d = 0; d < depth; ++d) {  // use fixed loop for unrolling
        double wj = w[d];
        qk += wj * QKv[offset + index + d];
        qr += wj * QRv[offset + index + d];
    }

    qk_out[j] = qk;
    qr_out[j] = qr;
}

// Estimate: 15 depth × 2 loads + 2 accum + 2 stores = moderate register usage
// __launch_bounds__ ensures we get good block scheduling
__global__ __launch_bounds__(64, 1)  // 256 threads/block, allow ~2 blocks/SM
void indexVecLN3Kernel(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t depth,
    size_t offset,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t index = inds[j];
    const double* w = weights + j * depth;

    double qk = 0.0, qr = 0.0;
    for (int d = 0; d < depth; ++d) {  // use fixed loop for unrolling
        double wj = w[d];
        qk += wj * QKv[offset + index + d];
        qr += wj * QRv[offset + index + d];
    }

    qk_out[j] = qk;
    qr_out[j] = qr;
}

// Optimized dispatcher class for indexVecLN3GPU
class IndexVecLN3Optimizer {
private:
    static size_t cached_depth;
    static std::function<void(const thrust::device_vector<double>&,
                             const thrust::device_vector<size_t>&,
                             thrust::device_vector<double>&,
                             thrust::device_vector<double>&,
                             cudaStream_t)> cached_kernel;

    template<int DEPTH>
    static void call_specialized(
        const thrust::device_vector<double>& weights,
        const thrust::device_vector<size_t>& inds,
        thrust::device_vector<double>& qk_result,
        thrust::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
        const size_t prod = inds.size();
        const size_t depth = weights.size() / prod;
        const size_t offset = sim->d_QKv.size() - len;

        const double* QKv = thrust::raw_pointer_cast(sim->d_QKv.data());
        const double* QRv = thrust::raw_pointer_cast(sim->d_QRv.data());
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());

        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;

        indexVecLN3Kernel_specialized<DEPTH><<<blocks, threads, 0, stream>>>(
            QKv, QRv, W, I, QK_out, QR_out, depth, offset, prod
        );
    }

    static void call_dynamic(
        const thrust::device_vector<double>& weights,
        const thrust::device_vector<size_t>& inds,
        thrust::device_vector<double>& qk_result,
        thrust::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
        const size_t prod = inds.size();
        const size_t depth = weights.size() / prod;
        const size_t offset = sim->d_QKv.size() - len;

        const double* QKv = thrust::raw_pointer_cast(sim->d_QKv.data());
        const double* QRv = thrust::raw_pointer_cast(sim->d_QRv.data());
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());

        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;

        indexVecLN3Kernel<<<blocks, threads, 0, stream>>>(
            QKv, QRv, W, I, QK_out, QR_out, depth, offset, prod
        );
    }

public:
    static void setupKernel(size_t depth) {
        if (cached_depth != depth) {
            cached_depth = depth;
            
            switch(depth) {
                case 9:  cached_kernel = call_specialized<9>; break;
                case 11: cached_kernel = call_specialized<11>; break;
                case 13: cached_kernel = call_specialized<13>; break;
                case 15: cached_kernel = call_specialized<15>; break;
                default: cached_kernel = call_dynamic; break;
            }
        }
    }
    
    static void execute(
        const thrust::device_vector<double>& weights,
        const thrust::device_vector<size_t>& inds,
        thrust::device_vector<double>& qk_result,
        thrust::device_vector<double>& qr_result,
        cudaStream_t stream = 0)
    {
        cached_kernel(weights, inds, qk_result, qr_result, stream);
    }
};

// Static member definitions
size_t IndexVecLN3Optimizer::cached_depth = 0;
std::function<void(const thrust::device_vector<double>&,
                  const thrust::device_vector<size_t>&,
                  thrust::device_vector<double>&,
                  thrust::device_vector<double>&,
                  cudaStream_t)> IndexVecLN3Optimizer::cached_kernel;

void indexVecLN3GPU(
    const thrust::device_vector<double>& weights,
    const thrust::device_vector<size_t>& inds,
    thrust::device_vector<double>& qk_result,
    thrust::device_vector<double>& qr_result,
    cudaStream_t stream = 0)
{
    const size_t depth = weights.size() / inds.size();
    
    // Setup the optimal kernel based on depth
    IndexVecLN3Optimizer::setupKernel(depth);
    
    // Execute the cached kernel
    IndexVecLN3Optimizer::execute(weights, inds, qk_result, qr_result, stream);
}

void indexVecN(const size_t length, const vector<double>& __restrict weights, const vector<size_t>& __restrict inds, const vector<double>& __restrict dtratio, vector<double>& __restrict qK_result, vector<double>& __restrict qR_result)
{
    size_t dims[] = { len,len };
    size_t t1len = dtratio.size();


    for (size_t i = 0; i < dims[0]; i++)
    {
        double in3 = weights[i] * weights[i];
        double in4 = in3 * weights[i];
        if (inds[i] < t1len - 1)
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
                qK_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * QKv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * QKv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * dQKv[inds[i] * dims[1] + j] - (in3 + in4) * dQKv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
                qR_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * QRv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * QRv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * dQRv[inds[i] * dims[1] + j] - (in3 + in4) * dQRv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
            }
        }
        else
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
                qK_result[j + dims[1] * i] = (1 - in3) * QKv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * dQKv[inds[i] * dims[1] + j] + in3 * QKv[inds[i] * dims[1] + j];
                qR_result[j + dims[1] * i] = (1 - in3) * QRv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * dQRv[inds[i] * dims[1] + j] + in3 * QRv[inds[i] * dims[1] + j];
            }
        }
    }
}

void indexVecNGPU(const thrust::device_vector<double>& weights, 
                  const thrust::device_vector<size_t>& inds, 
                  const thrust::device_vector<double>& dtratio, 
                  thrust::device_vector<double>& qK_result, 
                  thrust::device_vector<double>& qR_result,
                  const thrust::device_vector<double>& QKv, 
                  const thrust::device_vector<double>& QRv, 
                  const thrust::device_vector<double>& dQKv, 
                  const thrust::device_vector<double>& dQRv, 
                  size_t len,
                  cudaStream_t stream = 0) {
    size_t prod = len * len;
    size_t t1len = dtratio.size();

    const double* QKv_ptr = thrust::raw_pointer_cast(QKv.data());
    const double* QRv_ptr = thrust::raw_pointer_cast(QRv.data());
    const double* dQKv_ptr = thrust::raw_pointer_cast(dQKv.data());
    const double* dQRv_ptr = thrust::raw_pointer_cast(dQRv.data());
    const double* weights_ptr = thrust::raw_pointer_cast(weights.data());
    const size_t* inds_ptr = thrust::raw_pointer_cast(inds.data());
    const double* dtratio_ptr = thrust::raw_pointer_cast(dtratio.data());
    double* qK_result_ptr = thrust::raw_pointer_cast(qK_result.data());
    double* qR_result_ptr = thrust::raw_pointer_cast(qR_result.data());

    thrust::for_each_n(
        thrust::cuda::par.on(stream),
        thrust::make_counting_iterator<size_t>(0),
        prod,
        [=] __device__(size_t tid) {
            size_t i = tid / len;
            size_t j = tid % len;
            double in1 = weights_ptr[i];
            double in2 = in1 * in1;
            double in3 = in2 * in1;

            size_t ind = inds_ptr[i];
            size_t curr_idx = ind * len + j;
            size_t base_idx = curr_idx - len;
            size_t next_idx = curr_idx + len;

            if (ind < t1len - 1) {

                qK_result_ptr[tid] = (1 - 3 * in2 - 2 * in3) * QKv_ptr[base_idx] +
                                     (3 * in2 + 2 * in3) * QKv_ptr[curr_idx] -
                                     (in1 + 2 * in2 + in3) * dQKv_ptr[curr_idx] -
                                     (in2 + in3) * dQKv_ptr[next_idx] / dtratio_ptr[inds_ptr[i] + 1];

                qR_result_ptr[tid] = (1 - 3 * in2 - 2 * in3) * QRv_ptr[base_idx] +
                                     (3 * in2 + 2 * in3) * QRv_ptr[curr_idx] -
                                     (in1 + 2 * in2 + in3) * dQRv_ptr[curr_idx] -
                                     (in2 + in3) * dQRv_ptr[next_idx] / dtratio_ptr[inds_ptr[i] + 1];
            } else {
                qK_result_ptr[tid] = (1 - in2) * QKv_ptr[base_idx] -
                                     (in1 + in2) * dQKv_ptr[curr_idx] +
                                     in2 * QKv_ptr[curr_idx];

                qR_result_ptr[tid] = (1 - in2) * QRv_ptr[base_idx] -
                                     (in1 + in2) * dQRv_ptr[curr_idx] +
                                     in2 * QRv_ptr[curr_idx];
            }
        });
}

void indexVecR2(const vector<double>& __restrict in1, const vector<double>& __restrict in2, const vector<double>& __restrict in3, const vector<size_t>& __restrict inds, const vector<double>& __restrict dtratio, vector<double>& __restrict result)
{
    size_t dims = inds.size();
    size_t t1len = dtratio.size();


    for (size_t i = 0; i < dims; i++)
    {
        if (inds[i] < t1len - 1)
        {
            result[i] = (1 - 3 * pow_const<2>(in3[i]) - 2 * pow_const<3>(in3[i])) * in1[inds[i] - 1] + (3 * pow_const<2>(in3[i]) + 2 * pow_const<3>(in3[i])) * in1[inds[i]] - (in3[i] + 2 * pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i]] - (pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i] + 1] / dtratio[inds[i] + 1];
        }
        else
        {
            result[i] = (1 - pow_const<2>(in3[i])) * in1[inds[i] - 1] + pow_const<2>(in3[i]) * in1[inds[i]] - (in3[i] + pow_const<2>(in3[i])) * in2[inds[i]];
        }
    }
}

__global__ __launch_bounds__(64, 1)
void indexVecR2Kernel(
    const double* __restrict__ in1,
    const double* __restrict__ in2,
    const double* __restrict__ in3,
    const size_t* __restrict__ inds,
    const double* __restrict__ dtratio,
    double* __restrict__ result,
    size_t dims,
    size_t t1len)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dims) return;

    double u = in3[i];
    double u2 = u * u;
    double u3 = u2 * u;

    size_t ind = inds[i];
    double val;

    if (ind < t1len - 1) {
        double in1m1 = in1[ind - 1];
        double in10  = in1[ind];
        double in20  = in2[ind];
        double in21  = in2[ind + 1];
        double dtr   = dtratio[ind + 1];

        val = (1.0 - 3.0 * u2 - 2.0 * u3) * in1m1
            + (3.0 * u2 + 2.0 * u3) * in10
            - (u + 2.0 * u2 + u3) * in20
            - (u2 + u3) * in21 / dtr;
    } else {
        double in1m1 = in1[ind - 1];
        double in10  = in1[ind];
        double in20  = in2[ind];

        val = (1.0 - u2) * in1m1
            + u2 * in10
            - (u + u2) * in20;
    }

    result[i] = val;
}

void indexVecR2GPU(
    const thrust::device_vector<double>& in1,
    const thrust::device_vector<double>& in2,
    const thrust::device_vector<double>& in3,
    const thrust::device_vector<size_t>& inds,
    const thrust::device_vector<double>& dtratio,
    thrust::device_vector<double>& result,
    cudaStream_t stream = 0)
{
    size_t dims = inds.size();
    size_t t1len = dtratio.size();

    int threads = 64;
    int blocks = (dims + threads - 1) / threads;

    indexVecR2Kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(in1.data()),
        thrust::raw_pointer_cast(in2.data()),
        thrust::raw_pointer_cast(in3.data()),
        thrust::raw_pointer_cast(inds.data()),
        thrust::raw_pointer_cast(dtratio.data()),
        thrust::raw_pointer_cast(result.data()),
        dims,
        t1len
    );
}

void indexMatAll(const vector<double>& __restrict posx, const vector<size_t>& __restrict indsy,
    const vector<double>& __restrict weightsy, const vector<double>& __restrict dtratio,
    vector<double>& __restrict qK_result, vector<double>& __restrict qR_result)
{
    size_t prod = indsy.size();
    size_t dims2 = weightsy.size();
    size_t depth = dims2 / prod;
    size_t t1len = dtratio.size();

    double inx, inx2, inx3;
    size_t inds, indsx;

    for (size_t j = 0; j < prod; j++)
    {
        indsx = max(min((size_t)posx[j], (size_t)(posx[prod - 1] - 0.5)), (size_t)1);
        inx = posx[j] - indsx;
        inx2 = inx * inx;
        inx3 = inx2 * inx;
        inds = (indsx - 1) * len + indsy[j];

        auto weights_start = weightsy.begin() + depth * j;
        if (indsx < t1len - 1)
        {
            qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, QKv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQKv.begin() + len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, QKv.begin() + len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQKv.begin() + 2 * len + inds, 0.0) / dtratio[indsx + 1];
            qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, QRv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQRv.begin() + len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, QRv.begin() + len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQRv.begin() + 2 * len + inds, 0.0) / dtratio[indsx + 1];
        }
        else
        {
            qK_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, QKv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, QKv.begin() + len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, dQKv.begin() + len + inds, 0.0);
            qR_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, QRv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, QRv.begin() + len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, dQRv.begin() + len + inds, 0.0);
        }
    }
}

void indexMatAllGPU_slow(const thrust::device_vector<double>& posx, 
                    const thrust::device_vector<size_t>& indsy, 
                    const thrust::device_vector<double>& weightsy, 
                    const thrust::device_vector<double>& dtratio, 
                    thrust::device_vector<double>& qK_result, 
                    thrust::device_vector<double>& qR_result,
                    const thrust::device_vector<double>& QKv, 
                    const thrust::device_vector<double>& QRv, 
                    const thrust::device_vector<double>& dQKv, 
                    const thrust::device_vector<double>& dQRv, 
                    size_t len) {
    size_t prod = indsy.size();
    size_t dims2 = weightsy.size();
    size_t depth = dims2 / prod;
    size_t t1len = dtratio.size();

    const double* posx_ptr = thrust::raw_pointer_cast(posx.data());
    const size_t* indsy_ptr = thrust::raw_pointer_cast(indsy.data());
    const double* weightsy_ptr = thrust::raw_pointer_cast(weightsy.data());
    const double* dtratio_ptr = thrust::raw_pointer_cast(dtratio.data());
    const double* QKv_ptr = thrust::raw_pointer_cast(QKv.data());
    const double* QRv_ptr = thrust::raw_pointer_cast(QRv.data());
    const double* dQKv_ptr = thrust::raw_pointer_cast(dQKv.data());
    const double* dQRv_ptr = thrust::raw_pointer_cast(dQRv.data());
    double* qK_result_ptr = thrust::raw_pointer_cast(qK_result.data());
    double* qR_result_ptr = thrust::raw_pointer_cast(qR_result.data());

    thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        prod,
        [=] __device__(size_t j) {
            size_t indsx = max(min((size_t)posx_ptr[j], (size_t)(posx_ptr[prod - 1] - 0.5)), (size_t)1);
            double inx = posx_ptr[j] - indsx;
            double inx2 = inx * inx;
            double inx3 = inx2 * inx;
            size_t inds = (indsx - 1) * len + indsy_ptr[j];

            const double* weights_start = weightsy_ptr + depth * j;


            auto compute_inner_product = [=] __device__(const double* vec_start) -> double {
                double sum = 0.0;
                for (size_t d = 0; d < depth; ++d) {
                    sum += weights_start[d] * vec_start[d];
                }
                return sum;
            };

            if (indsx < t1len - 1) {
                qK_result_ptr[j] = (1 - 3 * inx2 + 2 * inx3) * compute_inner_product(QKv_ptr + inds) +
                                   (inx - 2 * inx2 + inx3) * compute_inner_product(dQKv_ptr + len + inds) +
                                   (3 * inx2 - 2 * inx3) * compute_inner_product(QKv_ptr + len + inds) +
                                   (-inx2 + inx3) * compute_inner_product(dQKv_ptr + 2 * len + inds) / dtratio_ptr[indsx + 1];

                qR_result_ptr[j] = (1 - 3 * inx2 + 2 * inx3) * compute_inner_product(QRv_ptr + inds) +
                                   (inx - 2 * inx2 + inx3) * compute_inner_product(dQRv_ptr + len + inds) +
                                   (3 * inx2 - 2 * inx3) * compute_inner_product(QRv_ptr + len + inds) +
                                   (-inx2 + inx3) * compute_inner_product(dQRv_ptr + 2 * len + inds) / dtratio_ptr[indsx + 1];
            } else {
                qK_result_ptr[j] = (1 - inx2) * compute_inner_product(QKv_ptr + inds) +
                                   inx2 * compute_inner_product(QKv_ptr + len + inds) +
                                   (inx - inx2) * compute_inner_product(dQKv_ptr + len + inds);

                qR_result_ptr[j] = (1 - inx2) * compute_inner_product(QRv_ptr + inds) +
                                   inx2 * compute_inner_product(QRv_ptr + len + inds) +
                                   (inx - inx2) * compute_inner_product(dQRv_ptr + len + inds);
            }
        });
}

void SigmaK(const vector<double>& qk, vector<double>& result)
{
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}

void SigmaKGPU(const thrust::device_vector<double>& qk, thrust::device_vector<double>& result, cudaStream_t stream = 0) {
    assert(qk.size() == result.size());

    thrust::transform(
        thrust::cuda::par.on(stream),
        qk.begin(), qk.end(),
        result.begin(),
        [] __device__(double qk_val) {
            return DflambdaGPU(qk_val);
        }
    );
}

void SigmaR(const vector<double>& qk, const vector<double>& qr, vector<double>& result)
{
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]) * qr[i];
    }
}

void SigmaRGPU(const thrust::device_vector<double>& qk, 
               const thrust::device_vector<double>& qr, 
               thrust::device_vector<double>& result,
               cudaStream_t stream = 0) {
    assert(qk.size() == qr.size() && qk.size() == result.size());

    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(thrust::make_tuple(qk.begin(), qr.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(qk.end(), qr.end())),
        result.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );
}

vector<double> SigmaK10(const vector<double>& qk)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return move(result);
}

vector<double> SigmaR10(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDDflambda(qk[i]) * qr[i];
    }
    return move(result);
}

vector<double> SigmaK01(const vector<double>& qk)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = 0.0;
    }
    return move(result);
}

vector<double> SigmaR01(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return move(result);
}

vector<double> ConvA(const vector<double>& f, const vector<double>& g, const double t)
{
    size_t length = integ.size();
    size_t depth = f.size() / length;
    vector<double> out(depth, 0.0);
    if (depth == 1)
    {
        double temp = 0.0;
        for (size_t j = 0; j < length; j++)
        {
            temp += t * integ[j] * f[j] * g[j];
        }
        out[0] = temp;
    }
    else
    {
        for (size_t j = 0; j < depth; j++)
        {
            for (size_t i = 0; i < length; i++)
            {
                out[j] += integ[i] * f[j * length + i] * g[j * length + i];
            }
            out[j] *= t * theta[j];
        }
    }
    return move(out);
}

thrust::device_vector<double> ConvAGPU_slow(const thrust::device_vector<double>& f, 
                                       const thrust::device_vector<double>& g, 
                                       const double t, 
                                       const thrust::device_vector<double>& integ, 
                                       const thrust::device_vector<double>& theta) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        depth,
        [=] __device__ (size_t j) {
            double sum = 0.0;
            for (size_t i = 0; i < length; ++i) {
                sum += f_ptr[j * length + i] * g_ptr[j * length + i] * integ_ptr[i];
            }
            double scale = (depth == 1) ? 1.0 : theta_ptr[j];
            out_ptr[j] = t * sum * scale;
        }
    );

    return move(out);
}

__global__ void ConvAGPUKernel(const double* __restrict__ f,
                               const double* __restrict__ g,
                               const double* __restrict__ integ,
                               const double* __restrict__ theta,
                               double* __restrict__ out,
                               const double* __restrict__ t,
                               size_t length,
                               size_t depth) {
    extern __shared__ double sdata[];  // Shared memory for block-level reduction

    const int j = blockIdx.x;          // Each block computes out[j]
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    double sum = 0.0;

    // Partial summation over i
    for (int i = tid; i < length; i += nthreads) {
        sum += f[j * length + i] * g[j * length + i] * integ[i];
    }

    // Store to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        double scale = t[0] * ((depth == 1) ? 1.0 : theta[j]);
        out[j] = scale * sdata[0];
    }
}

__global__ void ConvAGPUKernel(const double* __restrict__ f,
                               const double* __restrict__ g,
                               const double* __restrict__ integ,
                               const double* __restrict__ theta,
                               double* __restrict__ out,
                               double t,
                               size_t length,
                               size_t depth) {
    extern __shared__ double sdata[];  // Shared memory for block-level reduction

    const int j = blockIdx.x;          // Each block computes out[j]
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    double sum = 0.0;

    // Partial summation over i
    for (int i = tid; i < length; i += nthreads) {
        sum += f[j * length + i] * g[j * length + i] * integ[i];
    }

    // Store to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        double scale = t * ((depth == 1) ? 1.0 : theta[j]);
        out[j] = scale * sdata[0];
    }
}

thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = (theta.size() == depth) ? thrust::raw_pointer_cast(theta.data()) : nullptr;
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = threads * sizeof(double);

    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    // cudaDeviceSynchronize();

    return move(out);
}

__forceinline__ void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       thrust::device_vector<double>& out,
                                       thrust::device_vector<double>& t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    const double* t_ptr = thrust::raw_pointer_cast(t.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = threads * sizeof(double);

    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t_ptr, length, depth
    );
    // cudaDeviceSynchronize();
}

thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g);
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = (theta.size() == depth) ? thrust::raw_pointer_cast(theta.data()) : nullptr;
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = threads * sizeof(double);

    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    // cudaDeviceSynchronize();

    return move(out);
}

__forceinline__ void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       thrust::device_vector<double>& out,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g);
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = (theta.size() == depth) ? thrust::raw_pointer_cast(theta.data()) : nullptr;
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = threads * sizeof(double);

    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    // cudaDeviceSynchronize();
}

vector<double> ConvR(const vector<double>& f, const vector<double>& g, const double t)
{
    size_t length = integ.size();
    size_t depth = f.size() / length;
    vector<double> out(length, 0.0);
    for (size_t j = 0; j < length; j++)
    {
        for (size_t i = 0; i < depth; i++)
        {
            out[j] += integ[i] * f[j * length + i] * g[j * length + i];
        }
        out[j] *= t * (1 - theta[j]);
    }
    return move(out);
}

thrust::device_vector<double> ConvRGPU_slow(const thrust::device_vector<double>& f, 
                                       const thrust::device_vector<double>& g, 
                                       const double t, 
                                       const thrust::device_vector<double>& integ, 
                                       const thrust::device_vector<double>& theta) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    thrust::device_vector<double> out(length, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        length,
        [=] __device__(size_t j) {
            double sum = 0.0;
            for (size_t i = 0; i < depth; i++) {
                sum += integ_ptr[i] * f_ptr[j * depth + i] * g_ptr[j * depth + i];
            }
            out_ptr[j] = sum * t * (1 - theta_ptr[j]);
        });

    return move(out);
}

__global__ void ConvRKernel(const double* __restrict__ f,
                            const double* __restrict__ g,
                            const double* __restrict__ integ,
                            const double* __restrict__ theta,
                            double* __restrict__ out,
                            double t,
                            size_t length,
                            size_t depth) {
    extern __shared__ double shared[];
    double* integ_shared = shared;
    double* reduction_shared = &shared[length];  // For thread reduction

    int j = blockIdx.x;   // each block handles one output row
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Load integ into shared memory once
    for (int i = tid; i < length; i += nthreads) {
        integ_shared[i] = integ[i];
    }
    __syncthreads();

    // Compute thread-local partial sum
    double sum = 0.0;
    const size_t base = j * length;

    for (size_t i = tid; i < length; i += nthreads) {
        double fval = f[base + i];
        double gval = g[base + i];
        sum += fval * gval * integ_shared[i];
    }

    // Store thread-local result in shared memory
    reduction_shared[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction_shared[tid] += reduction_shared[tid + stride];
        }
        __syncthreads();
    }

    // Final scaling and write
    if (tid == 0) {
        double scale = t * (1.0 - theta[j]);
        out[j] = scale * reduction_shared[0];
    }
}

__global__ void ConvRKernel(const double* __restrict__ f,
                            const double* __restrict__ g,
                            const double* __restrict__ integ,
                            const double* __restrict__ theta,
                            double* __restrict__ out,
                            const double* __restrict__ t,
                            size_t length,
                            size_t depth) {
    extern __shared__ double shared[];
    double* integ_shared = shared;
    double* reduction_shared = &shared[length];  // For thread reduction

    int j = blockIdx.x;   // each block handles one output row
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Load integ into shared memory once
    for (int i = tid; i < length; i += nthreads) {
        integ_shared[i] = integ[i];
    }
    __syncthreads();

    // Compute thread-local partial sum
    double sum = 0.0;
    const size_t base = j * length;

    for (size_t i = tid; i < length; i += nthreads) {
        double fval = f[base + i];
        double gval = g[base + i];
        sum += fval * gval * integ_shared[i];
    }

    // Store thread-local result in shared memory
    reduction_shared[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction_shared[tid] += reduction_shared[tid + stride];
        }
        __syncthreads();
    }

    // Final scaling and write
    if (tid == 0) {
        double scale = t[0] * (1.0 - theta[j]);
        out[j] = scale * reduction_shared[0];
    }
}

thrust::device_vector<double> ConvRGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();              // rows
    size_t depth = f.size() / length;          // output entries

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = length * sizeof(double) + threads * sizeof(double);

    ConvRKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    // cudaDeviceSynchronize();

    return move(out);
}

void ConvRGPU_Stream(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       thrust::device_vector<double>& out,
                                       const thrust::device_vector<double>& t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0) {
    size_t length = integ.size();              // rows
    size_t depth = f.size() / length;          // output entries

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    const double* t_ptr = thrust::raw_pointer_cast(t.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 64;
    size_t shmem = length * sizeof(double) + threads * sizeof(double);

    ConvRKernel<<<depth, threads, shmem, stream>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t_ptr, length, depth
    );
    // cudaDeviceSynchronize();
}

vector<double> QKstep()
{
    vector<double> temp(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
    for (size_t i = 0; i < QKB1int.size(); i += len) {
        temp[i / len] = QKB1int[i];
    }
    vector<double> d1qK = (temp* (Dflambda(QKv[QKv.size() - len]) / T0)) + (qK * (-rInt.back())) + 
    ConvR(SigmaRA2int, QKB2int, t1grid.back()) + ConvA(SigmaRA1int, QKB1int, t1grid.back()) + 
    ConvA(SigmaKA1int, QRB1int, t1grid.back());
    for (size_t i = 0; i < QKB1int.size(); i += len) {
        temp[i / len] = Dflambda(QKB1int[i]);
    }
    vector<double> d2qK = (temp * (QKv[QKv.size() - len] / T0)) + (qR * (2 * Gamma)) + 
    ConvR(QRA2int, SigmaKB2int, t1grid.back()) + ConvA(QRA1int, SigmaKB1int, t1grid.back()) + 
    ConvA(QKA1int, SigmaRB1int, t1grid.back()) - (qK * rInt);
    return d1qK + (d2qK * theta);
}

thrust::device_vector<double> QKstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& QKB1int,
    const thrust::device_vector<double>& QKB2int,
    const thrust::device_vector<double>& QKA1int,
    const thrust::device_vector<double>& QRA1int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& QRB1int,
    const thrust::device_vector<double>& SigmaRA1int,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& SigmaKB1int,
    const thrust::device_vector<double>& SigmaKB2int,
    const thrust::device_vector<double>& SigmaKA1int,
    const thrust::device_vector<double>& SigmaRB1int,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& rInt,
    double T0,
    double Gamma,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    // double t = t1grid.back();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;
    thrust::device_ptr<double> qK = get_slice_ptr(QKv,t1len-1,len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv,t1len-1,len);
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + len;

    thrust::for_each(thrust::cuda::par.on(pool[6]),
        idx_first, idx_last,
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1grid.size() - 1), ptr = QKB1int.data(), qk0 = QKv[(t1len - 1) * len], temp0 = sim->temp0.begin(), temp1 = sim->temp1.begin(), temp4 = sim->temp4.begin(), temp5 = sim->temp5.begin(), temp6 = sim->temp6.begin(), temp7 = sim->temp7.begin(), temp8 = sim->temp8.begin()] __device__ (size_t i) {
            temp0[i] = ptr[i * len];
            temp1[i] = DflambdaGPU(ptr[i * len]);
            if (i == 0) {
                temp4[0] = DflambdaGPU(qk0) / T0;
                temp5[0] = qk0 / T0;  // Use only the first element
                temp6[0] = -(*rInt_back);
                temp7[0] = 2.0 * Gamma;
                temp8[0] = *t1_back;
            }
        }
    );

    // computeScale<<<1,1,0,pool[7]>>>(thrust::raw_pointer_cast(sim->temp4.data()), T0);

    // Step 1: Run reductions
    ConvAGPU_Stream(SigmaRA1int, QKB1int, sim->convA1_1, sim->temp8, integ, theta, pool[0]);
    ConvAGPU_Stream(SigmaKA1int, QRB1int, sim->convA2_1, sim->temp8, integ, theta, pool[1]);
    ConvAGPU_Stream(QRA1int, SigmaKB1int, sim->convA1_2, sim->temp8, integ, theta, pool[2]);
    ConvAGPU_Stream(QKA1int, SigmaRB1int, sim->convA2_2, sim->temp8, integ, theta, pool[3]);
    ConvRGPU_Stream(SigmaRA2int, QKB2int, sim->convR_1, sim->temp8, integ, theta, pool[4]);
    ConvRGPU_Stream(QRA2int, SigmaKB2int, sim->convR_2, sim->temp8, integ, theta, pool[5]);

    // You can optionally sync here or later depending on downstream dependencies
    std::vector<cudaEvent_t> events(7);
    for (int i = 0; i <= 6; ++i) {
        // Create and record an event in each stream
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        cudaEventRecord(events[i], pool[i]);

        // Make streamA wait for each event
        cudaStreamWaitEvent(pool[7], events[i], 0);
        cudaStreamWaitEvent(pool[8], events[i], 0);
    }

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->temp0.data()), qK, sim->temp2, thrust::raw_pointer_cast(sim->temp4.data()), thrust::raw_pointer_cast(sim->temp6.data()), nullptr, &sim->convR_1, &sim->convA1_1, &sim->convA2_1, nullptr, pool[7]);

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(sim->temp1.data()), qR, sim->temp3, thrust::raw_pointer_cast(sim->temp5.data()), thrust::raw_pointer_cast(sim->temp7.data()), &rInt, &sim->convR_2, &sim->convA1_2, &sim->convA2_2, qK, pool[8]);

    // Combine d1qK and d2qK
    computeProduct<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(theta.data()),theta.size());
    computeSum<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->temp2.data()),thrust::raw_pointer_cast(sim->temp3.data()),theta.size());
    // thrust::device_vector<double> result = SumGPU(sim->temp2, ProductGPU(sim->temp3,theta));

    return sim->temp2;
}

vector<double> QRstep()
{
    vector<double> qR = getLastLenEntries(QRv, len);
    vector<double> d1qR = (qR * (-rInt.back())) + ConvR(SigmaRA2int, QRB2int, t1grid.back());
    vector<double> d2qR = (qR * rInt) - ConvR(QRA2int, SigmaRB2int, t1grid.back());
    return d1qR + (d2qR * theta);
}

thrust::device_vector<double> QRstepGPU(
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& rInt,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& QRB2int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& SigmaRB2int,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& theta,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    sim->temp8[0] = t1grid.back();
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len - 1, len);

    ConvRGPU_Stream(SigmaRA2int, QRB2int, sim->convR_1, sim->temp8, integ, theta, pool[0]);
    ConvRGPU_Stream(QRA2int, SigmaRB2int, sim->convR_2, sim->temp8, integ, theta, pool[1]);

    QRstepFused(qR, theta, sim->convR_1, sim->convR_2, rInt, thrust::raw_pointer_cast(sim->temp2.data()));


    return sim->temp2;
}

void QKRstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& QKB1int,
    const thrust::device_vector<double>& QKB2int,
    const thrust::device_vector<double>& QKA1int,
    const thrust::device_vector<double>& QRA1int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& QRB1int,
    const thrust::device_vector<double>& QRB2int,
    const thrust::device_vector<double>& SigmaRA1int,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& SigmaKB1int,
    const thrust::device_vector<double>& SigmaKB2int,
    const thrust::device_vector<double>& SigmaKA1int,
    const thrust::device_vector<double>& SigmaRB1int,
    const thrust::device_vector<double>& SigmaRB2int,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& rInt,
    thrust::device_vector<double>& outK,
    thrust::device_vector<double>& outR,
    double T0,
    double Gamma,
    int n,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    // double t = t1grid.back();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;
    thrust::device_ptr<double> qK = get_slice_ptr(QKv,t1len-1,len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv,t1len-1,len);
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + len;

    thrust::for_each(thrust::cuda::par.on(pool[0]),
        idx_first, idx_last,
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1len - 1), ptr = QKB1int.data(), qk0 = QKv[(t1len - 1) * len], temp0 = sim->temp0.begin(), temp1 = sim->temp1.begin(), temp4 = sim->temp4.begin(), temp5 = sim->temp5.begin(), temp6 = sim->temp6.begin(), temp7 = sim->temp7.begin(), temp8 = sim->temp8.begin()] __device__ (size_t i) {
            temp0[i] = ptr[i * len];
            temp1[i] = DflambdaGPU(ptr[i * len]);
            if (i == 0) {
                temp4[0] = DflambdaGPU(qk0) / T0;
                temp5[0] = qk0 / T0;  // Use only the first element
                temp6[0] = -(*rInt_back);
                temp7[0] = 2.0 * Gamma;
                temp8[0] = *t1_back;
            }
        }
    );

    // std::vector<cudaEvent_t> events(11);
    // // Create and record an event in first stream
    // cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming);
    // cudaEventRecord(events[0], pool[0]);
    // for (int i = 1; i <= 8; ++i) {
    //     // Make pool[i] wait for the event
    //     cudaStreamWaitEvent(pool[i], events[0], 0);
    //     cudaStreamWaitEvent(pool[i], events[0], 0);
    // }

    // computeScale<<<1,1,0,pool[7]>>>(thrust::raw_pointer_cast(sim->temp4.data()), T0);

    // Step 1: Run reductions
    ConvAGPU_Stream(SigmaRA1int, QKB1int, sim->convA1_1, sim->temp8, integ, theta, pool[1]);
    ConvAGPU_Stream(SigmaKA1int, QRB1int, sim->convA2_1, sim->temp8, integ, theta, pool[2]);
    ConvAGPU_Stream(QRA1int, SigmaKB1int, sim->convA1_2, sim->temp8, integ, theta, pool[3]);
    ConvAGPU_Stream(QKA1int, SigmaRB1int, sim->convA2_2, sim->temp8, integ, theta, pool[4]);
    ConvRGPU_Stream(SigmaRA2int, QKB2int, sim->convR_1, sim->temp8, integ, theta, pool[5]);
    ConvRGPU_Stream(QRA2int, SigmaKB2int, sim->convR_2, sim->temp8, integ, theta, pool[6]);
    ConvRGPU_Stream(SigmaRA2int, QRB2int, sim->convR_3, sim->temp8, integ, theta, pool[7]);
    ConvRGPU_Stream(QRA2int, SigmaRB2int, sim->convR_4, sim->temp8, integ, theta, pool[8]);

    // for (int i = 1; i <= 8; ++i) {
    //     // Create and record an event in each stream
    //     cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    //     cudaEventRecord(events[i], pool[i]);

    //     // Make pool[9], pool[10] and pool[11] wait for each event
    //     cudaStreamWaitEvent(pool[9], events[i], 0);
    //     cudaStreamWaitEvent(pool[10], events[i], 0);
    //     cudaStreamWaitEvent(pool[11], events[i], 0);
    // }

    cudaDeviceSynchronize();

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->temp0.data()), qK, sim->temp2, thrust::raw_pointer_cast(sim->temp4.data()), thrust::raw_pointer_cast(sim->temp6.data()), nullptr, &sim->convR_1, &sim->convA1_1, &sim->convA2_1, nullptr, pool[9]);

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(sim->temp1.data()), qR, sim->temp3, thrust::raw_pointer_cast(sim->temp5.data()), thrust::raw_pointer_cast(sim->temp7.data()), &rInt, &sim->convR_2, &sim->convA1_2, &sim->convA2_2, qK, pool[10]);

    // Combine d1qK and d2qK
    computeProduct<<<blocks, threads, 0, pool[10]>>>(thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(theta.data()),len);
    computeSum<<<blocks, threads, 0, pool[10]>>>(thrust::raw_pointer_cast(sim->temp2.data()),thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(outK.data()) + n * len, len);
    // thrust::device_vector<double> result = SumGPU(sim->temp2, ProductGPU(sim->temp3,theta));

    // for (int i = 9; i <= 10; ++i) {
    //     // Create and record an event in each stream
    //     cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    //     cudaEventRecord(events[i], pool[i]);

    //     // Make streamA wait for each event
    //     cudaStreamWaitEvent(pool[11], events[i], 0);
    // }

    QRstepFused(qR, theta, sim->convR_3, sim->convR_4, rInt, thrust::raw_pointer_cast(outR.data()) + n * len, pool[11]);
}

double rstep()
{
    vector<double> sigmaK(len, 0.0), sigmaR(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
    const double t = t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    return Gamma + ConvA(sigmaR, qK, t).front() + ConvA(sigmaK, qR, t).front() + sigmaK.front() * qK.front() / T0;
}

double rstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    const double t = t1grid.back();

    thrust::device_ptr<double> qK = get_slice_ptr(QKv, t1len - 1, len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len- 1, len);

    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    computeSigmaKandRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp1.data()),
        len
    );

    // Compute convolution results
    // sim->temp2 = ConvAGPU(sim->temp1, qK, t, sim->d_integ, sim->d_theta);
    // sim->temp3 = ConvAGPU(sim->temp0, qR, t, sim->d_integ, sim->d_theta);

    ConvAGPU_Stream(sim->temp1, qK, sim->temp2, t, integ, theta, pool[1]);
    ConvAGPU_Stream(sim->temp0, qR, sim->temp3, t, integ, theta, pool[2]);

    // Compute final result
    double result = Gamma +
                    sim->temp2[0] +
                    sim->temp3[0] +
                    sim->temp0[0] * (*qK) / T0;

    
    return result;
}

double drstep()
{
    vector<double> sigmaK(len, 0.0), sigmaR(len, 0.0), dsigmaK(len, 0.0), dsigmaR(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
    vector<double> dqK = getLastLenEntries(dQKv, len);
    vector<double> dqR = getLastLenEntries(dQRv, len);
    const double t = t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR) * dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1).front() + ConvA(sigmaK, qR, 1).front() + ConvA(dsigmaR, qK, t).front() + ConvA(dsigmaK, qR, t).front() + ConvA(sigmaR, dqK, t).front() + ConvA(sigmaK, dqR, t).front() + (dsigmaK.front() * qK.front() + sigmaK.front() * dqK.front()) / T0;
}

double drstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& dQKv,
    const thrust::device_vector<double>& dQRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0) {
    
    size_t len = QKv.size();
    const double t = t1grid.back();
    thrust::device_vector<double> sigmaK(len, 0.0);
    thrust::device_vector<double> sigmaR(len, 0.0);
    thrust::device_vector<double> dsigmaK(len, 0.0);
    thrust::device_vector<double> dsigmaR(len, 0.0);

    // Compute sigmaK
    thrust::transform(
        QKv.begin(), QKv.end(),
        sigmaK.begin(),
        [] __device__(double qk) { return DflambdaGPU(qk); }
    );

    // Compute sigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), QRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), QRv.end())),
        sigmaR.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );

    // Compute dsigmaK
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), dQKv.begin(), dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), dQKv.end(), dQRv.end())),
        dsigmaK.begin(),
        [] __device__(thrust::tuple<double, double, double> qk_dqk_dqr) {
            double qk = thrust::get<0>(qk_dqk_dqr);
            double dqk = thrust::get<1>(qk_dqk_dqr);
            double dqr = thrust::get<2>(qk_dqk_dqr);
            return DDflambdaGPU(qk) * dqk + DflambdaGPU(qk) * dqr;
        }
    );

    // Compute dsigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), QRv.begin(), dQKv.begin(), dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), QRv.end(), dQKv.end(), dQRv.end())),
        dsigmaR.begin(),
        [] __device__(thrust::tuple<double, double, double, double> qk_qr_dqk_dqr) {
            double qk = thrust::get<0>(qk_qr_dqk_dqr);
            double qr = thrust::get<1>(qk_qr_dqk_dqr);
            double dqk = thrust::get<2>(qk_qr_dqk_dqr);
            double dqr = thrust::get<3>(qk_qr_dqk_dqr);
            return DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaR_qK = ConvAGPU(sigmaR, QKv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(sigmaK, QRv, t, integ, theta);
    thrust::device_vector<double> convA_dsigmaR_qK = ConvAGPU(dsigmaR, QKv, t, integ, theta);
    thrust::device_vector<double> convA_dsigmaK_qR = ConvAGPU(dsigmaK, QRv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaR_dqK = ConvAGPU(sigmaR, dQKv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaK_dqR = ConvAGPU(sigmaK, dQRv, t, integ, theta);

    // Compute final result
    double result = convA_sigmaR_qK.front() +
                    convA_sigmaK_qR.front() +
                    convA_dsigmaR_qK.front() +
                    convA_dsigmaK_qR.front() +
                    convA_sigmaR_dqK.front() +
                    convA_sigmaK_dqR.front() +
                    (dsigmaK.front() * QKv.front() + sigmaK.front() * dQKv.front()) / T0;

    return move(result);
}

double drstep2(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double t)
{
    vector<double> sigmaK(qK.size(), 0.0), sigmaR(qK.size(), 0.0), dsigmaK(qK.size(), 0.0), dsigmaR(qK.size(), 0.0);
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR)* dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1).front() + ConvA(sigmaK, qR, 1).front() + ConvA(dsigmaR, qK, t).front() + ConvA(dsigmaK, qR, t).front() + ConvA(sigmaR, dqK, t).front() + ConvA(sigmaK, dqR, t).front() + (dsigmaK.front() * qK.front() + sigmaK.front() * dqK.front()) / T0;
    // return ConvA(sigmaK, dqR, t).front();
}

double drstep2GPU(
    const thrust::device_ptr<double>& QKv,
    const thrust::device_ptr<double>& QRv,
    const thrust::device_ptr<double>& dQKv,
    const thrust::device_ptr<double>& dQRv,
    const double t,
    const double T0,
    StreamPool& pool) {
    
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        QKv, QRv, dQKv, dQRv,
        sim->temp0.begin(),  // replaces sigmaK
        sim->temp1.begin(),
        sim->temp2.begin(),
        sim->temp3.begin()
    ));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        QKv + len, QRv + len, dQKv + len, dQRv + len,
        sim->temp0.begin() + len,  // same here
        sim->temp1.begin() + len,
        sim->temp2.begin() + len,
        sim->temp3.begin() + len
    ));

    thrust::for_each(
        thrust::cuda::par.on(pool[0]),
        begin, end,
        [] __device__ (const thrust::tuple<
            double, double, double, double,
            double&, double&, double&, double&
        >& t) {
            double qk  = thrust::get<0>(t);
            double qr  = thrust::get<1>(t);
            double dqk = thrust::get<2>(t);
            double dqr = thrust::get<3>(t);

            double& sigmaK  = thrust::get<4>(t);  // now refers to sim->temp0
            double& sigmaR  = thrust::get<5>(t);
            double& dsigmaK = thrust::get<6>(t);
            double& dsigmaR = thrust::get<7>(t);

            sigmaK  = DflambdaGPU(qk);
            sigmaR  = DDflambdaGPU(qk) * qr;
            dsigmaK = DDflambdaGPU(qk) * dqk;
            dsigmaR = DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    // sim->temp4 = ConvAGPU(sim->temp1, QKv, 1.0, sim->d_integ, sim->d_theta, pool[1]);
    // sim->temp5 = ConvAGPU(sim->temp0, QRv, 1.0, sim->d_integ, sim->d_theta, pool[2]);
    // sim->temp6= ConvAGPU(sim->temp3, QKv, t, sim->d_integ, sim->d_theta, pool[3]);
    // sim->temp7 = ConvAGPU(sim->temp2, QRv, t, sim->d_integ, sim->d_theta, pool[4]);
    // sim->temp8 = ConvAGPU(sim->temp1, dQKv, t, sim->d_integ, sim->d_theta, pool[5]);
    // sim->temp9 = ConvAGPU(sim->temp0, dQRv, t, sim->d_integ, sim->d_theta, pool[6]);
    ConvAGPU_Stream(sim->temp1, QKv, sim->temp4, 1.0, sim->d_integ, sim->d_theta, pool[1]);
    ConvAGPU_Stream(sim->temp0, QRv, sim->temp5, 1.0, sim->d_integ, sim->d_theta, pool[2]);
    ConvAGPU_Stream(sim->temp3, QKv, sim->temp6, t, sim->d_integ, sim->d_theta, pool[3]);
    ConvAGPU_Stream(sim->temp2, QRv, sim->temp7, t, sim->d_integ, sim->d_theta, pool[4]);
    ConvAGPU_Stream(sim->temp1, dQKv, sim->temp8, t, sim->d_integ, sim->d_theta, pool[5]);
    ConvAGPU_Stream(sim->temp0, dQRv, sim->temp9, t, sim->d_integ, sim->d_theta, pool[6]);

    // You can optionally sync here or later depending on downstream dependencies

    // Compute final result
    double result = sim->temp4[0] + sim->temp5[0] + sim->temp6[0] + sim->temp7[0] + sim->temp8[0] + sim->temp9[0] +
                    (sim->temp2[0] * (*QKv) + sim->temp0[0] * (*dQKv)) / T0;

    return result;
}

// __global__ void fusedDrstepKernel(
//     const double* __restrict__ QKv,
//     const double* __restrict__ QRv,
//     const double* __restrict__ dQKv,
//     const double* __restrict__ dQRv,
//     const double* __restrict__ integ,
//     const double* __restrict__ theta,
//     double t, double T0,
//     size_t len, size_t depth,
//     double* __restrict__ temp0,  // sigmaK
//     double* __restrict__ temp1,  // sigmaR
//     double* __restrict__ temp2,  // dsigmaK
//     double* __restrict__ temp3,  // dsigmaR
//     double* __restrict__ temp4,  // convA_sigmaR_dQKv
//     double* __restrict__ temp5,  // convA_sigmaK_QRv
//     double* __restrict__ temp6,  // convA_dsigmaR_QKv
//     double* __restrict__ temp7,  // convA_dsigmaK_QRv
//     double* __restrict__ temp8,  // convA_sigmaR_dQKv
//     double* __restrict__ temp9   // convA_sigmaK_dQRv
// ) {
//     extern __shared__ double sdata[];  // 6 × threads

//     int j = blockIdx.x;
//     int tid = threadIdx.x;
//     int nthreads = blockDim.x;

//     double sum4 = 0.0, sum5 = 0.0, sum6 = 0.0, sum7 = 0.0, sum8 = 0.0, sum9 = 0.0;

//     for (int i = tid; i < len; i += nthreads) {
//         size_t idx = j * len + i;
//         double qk = QKv[idx];
//         double qr = QRv[idx];
//         double dqk = dQKv[idx];
//         double dqr = dQRv[idx];

//         double sigmaK  = DflambdaGPU(qk);
//         double sigmaR  = DDflambdaGPU(qk) * qr;
//         double dsigmaK = DDflambdaGPU(qk) * dqk;
//         double dsigmaR = DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;

//         double w = integ[i];

//         sum4 += sigmaR  * dqk * w;
//         sum5 += sigmaK  * qr  * w;
//         sum6 += dsigmaR * qk  * w;
//         sum7 += dsigmaK * qr  * w;
//         sum8 += sigmaR  * dqk * w;
//         sum9 += sigmaK  * dqr * w;

//         if (i == 0 && tid == 0) {
//             temp0[j] = sigmaK;
//             temp1[j] = sigmaR;
//             temp2[j] = dsigmaK;
//             temp3[j] = dsigmaR;
//         }
//     }

//     sdata[tid + 0 * nthreads] = sum4;
//     sdata[tid + 1 * nthreads] = sum5;
//     sdata[tid + 2 * nthreads] = sum6;
//     sdata[tid + 3 * nthreads] = sum7;
//     sdata[tid + 4 * nthreads] = sum8;
//     sdata[tid + 5 * nthreads] = sum9;
//     __syncthreads();

//     for (int s = nthreads / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             #pragma unroll
//             for (int k = 0; k < 6; ++k) {
//                 sdata[tid + k * nthreads] += sdata[tid + s + k * nthreads];
//             }
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         double scale = t * ((theta != nullptr) ? theta[j] : 1.0);
//         temp4[j] = scale * sdata[0 * nthreads]; // sigmaR * dQKv
//         temp5[j] = scale * sdata[1 * nthreads]; // sigmaK * QRv
//         temp6[j] = scale * sdata[2 * nthreads]; // dsigmaR * QKv
//         temp7[j] = scale * sdata[3 * nthreads]; // dsigmaK * QRv
//         temp8[j] = scale * sdata[4 * nthreads]; // sigmaR * dQKv (again)
//         temp9[j] = scale * sdata[5 * nthreads]; // sigmaK * dQRv
//     }
// }

// double drstep2GPU(
//     const thrust::device_ptr<double>& QKv,
//     const thrust::device_ptr<double>& QRv,
//     const thrust::device_ptr<double>& dQKv,
//     const thrust::device_ptr<double>& dQRv,
//     double t,
//     double T0,
//     StreamPool& pool
// ) {
//     size_t len = sim->d_integ.size();
//     size_t depth = sim->temp0.size() / len;

//     size_t threads = 64;
//     size_t shmem = threads * 6 * sizeof(double);  // 6 convolution sums

//     fusedDrstepKernel<<<depth, threads, shmem, pool[0]>>>(
//         thrust::raw_pointer_cast(QKv),
//         thrust::raw_pointer_cast(QRv),
//         thrust::raw_pointer_cast(dQKv),
//         thrust::raw_pointer_cast(dQRv),
//         thrust::raw_pointer_cast(sim->d_integ.data()),
//         sim->d_theta.size() == depth ? thrust::raw_pointer_cast(sim->d_theta.data()) : nullptr,
//         t, T0, len, depth,
//         thrust::raw_pointer_cast(sim->temp0.data()),  // sigmaK
//         thrust::raw_pointer_cast(sim->temp1.data()),  // sigmaR
//         thrust::raw_pointer_cast(sim->temp2.data()),  // dsigmaK
//         thrust::raw_pointer_cast(sim->temp3.data()),  // dsigmaR
//         thrust::raw_pointer_cast(sim->temp4.data()),  // convA_sigmaR_dQKv
//         thrust::raw_pointer_cast(sim->temp5.data()),  // convA_sigmaK_QRv
//         thrust::raw_pointer_cast(sim->temp6.data()),  // convA_dsigmaR_QKv
//         thrust::raw_pointer_cast(sim->temp7.data()),  // convA_dsigmaK_QRv
//         thrust::raw_pointer_cast(sim->temp8.data()),  // convA_sigmaR_dQKv (again)
//         thrust::raw_pointer_cast(sim->temp9.data())   // convA_sigmaK_dQRv
//     );

//     double r[7];
//     cudaMemcpyAsync(r + 0, thrust::raw_pointer_cast(sim->temp4.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 1, thrust::raw_pointer_cast(sim->temp5.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 2, thrust::raw_pointer_cast(sim->temp6.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 3, thrust::raw_pointer_cast(sim->temp7.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 4, thrust::raw_pointer_cast(sim->temp8.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 5, thrust::raw_pointer_cast(sim->temp9.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]);
//     cudaMemcpyAsync(r + 6, thrust::raw_pointer_cast(sim->temp2.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]); // dsigmaK
//     double sigmaK;
//     cudaMemcpyAsync(&sigmaK, thrust::raw_pointer_cast(sim->temp0.data()), sizeof(double), cudaMemcpyDeviceToHost, pool[0]); // sigmaK
//     cudaStreamSynchronize(pool[0]);

//     return r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + (r[6] * QKv[0] + sigmaK * dQKv[0]) / T0;
// }

double energyGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    const double t = t1grid.back();

    thrust::device_ptr<double> qK = get_slice_ptr(QKv, t1len - 1, len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len- 1, len);

    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    computeSigmaKandRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp1.data()),
        len
    );

    // Compute convolution results
    ConvAGPU_Stream(sim->temp0, qR, sim->temp2, t, integ, theta);

    // Compute final result
    double result = - (sim->temp2[0] + sim->temp0[0] / T0);

    

    return move(result);
}

void appendAll(const vector<double>& qK,
    const vector<double>& qR,
    const vector<double>& dqK,
    const vector<double>& dqR,
    const double dr,
    const double t)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update t1grid and delta_t_ratio
    t1grid.push_back(t);
    size_t idx = t1grid.size() - 1;
    double tdiff = t1grid[idx] - t1grid[idx - 1];
    if (idx > 1) {
        double prev = t1grid[idx - 1] - t1grid[idx - 2];
        delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        delta_t_ratio.push_back(0.0);
    }

    for (size_t i = 0; i < length; i++)
    {
        QKv.push_back(qK[i]);
        QRv.push_back(qR[i]);
        dQKv.push_back(tdiff * dqK[i]);
        dQRv.push_back(tdiff * dqR[i]);
    }

    // 2) finally update drvec and rvec
    drvec.push_back(tdiff * dr);
    rvec.push_back(rstep());
}

void appendGPU(thrust::device_vector<double>& dest,
                                        const thrust::device_vector<double>& src, double scale = 1.0) {
    size_t required_size = dest.size() + src.size();

    if (dest.capacity() < required_size) {
        // Allocate a new vector with more capacity (e.g., double the required size)
        size_t new_capacity = std::max(required_size, 2 * dest.capacity());
        thrust::device_vector<double> tmp;
        tmp.reserve(new_capacity);            // Preallocate
        tmp.resize(dest.size());              // Set to current size
        thrust::copy(dest.begin(), dest.end(), tmp.begin());
        dest.swap(tmp);                       // Efficient ownership transfer
    }

    size_t insert_pos = dest.size();
    dest.resize(required_size);
    if (scale == 1.){
        thrust::copy(src.begin(), src.end(), dest.begin() + insert_pos);
    }
    else {
        thrust::transform(
        src.begin(), src.end(),
        dest.begin() + insert_pos,
        [scale] __device__ (double val) {
            return val * scale;
        }
    );
    }
}

void appendGPU_ptr(thrust::device_vector<double>& dest,
                                        const thrust::device_ptr<double>& src, double size, double scale = 1.0, cudaStream_t stream = 0) {
    size_t required_size = dest.size() + size;

    if (dest.capacity() < required_size) {
        // Allocate a new vector with more capacity
        dest.reserve(dest.size() + 1000 * size);
    }

    size_t insert_pos = dest.size();
    dest.resize(required_size);

    const int threads = 64;
    const int blocks = (len + threads - 1) / threads;

    computeCopy<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(src),
        thrust::raw_pointer_cast(dest.data()),
        insert_pos,
        size,
        scale
    );
}

void appendAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update sim->d_t1grid and sim->d_delta_t_ratio
    sim->d_t1grid.push_back(t);
    size_t idx = sim->d_t1grid.size() - 1;
    double tdiff = sim->d_t1grid[idx] - sim->d_t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->d_t1grid[idx - 1] - sim->d_t1grid[idx - 2];
        sim->d_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->d_delta_t_ratio.push_back(0.0);
    }

    appendGPU(sim->d_QKv,qK);
    appendGPU(sim->d_QRv,qR);
    appendGPU(sim->d_dQKv, dqK, tdiff);
    appendGPU(sim->d_dQRv, dqR, tdiff);

    // 2) finally update drvec and rvec
    sim->d_drvec.push_back(tdiff * dr);
    sim->d_rvec.push_back(rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, Gamma, T0, pool));
}

void appendAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool)
{
    size_t t1len = sim->d_t1grid.size();
    if (sim->d_t1grid.capacity() < t1len + 1) {
        // Allocate a new vector with more capacity
        sim->d_t1grid.reserve(t1len + 1000);
        sim->d_delta_t_ratio.reserve(t1len + 1000);
    }

    // 1) update sim->d_t1grid and sim->d_delta_t_ratio
    sim->d_t1grid.push_back(t);
    size_t idx = t1len;
    double tdiff = sim->d_t1grid[idx] - sim->d_t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->d_t1grid[idx - 1] - sim->d_t1grid[idx - 2];
        sim->d_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->d_delta_t_ratio.push_back(0.0);
    }

    appendGPU_ptr(sim->d_QKv, qK, len, 1.0, pool[0]);
    appendGPU_ptr(sim->d_QRv, qR, len, 1.0, pool[1]);
    appendGPU_ptr(sim->d_dQKv, dqK, len, tdiff, pool[2]);
    appendGPU_ptr(sim->d_dQRv, dqR, len, tdiff, pool[3]);

    // 2) finally update drvec and rvec
    sim->d_drvec.push_back(tdiff * dr);
    sim->d_rvec.push_back(rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, Gamma, T0, pool));
}

void replaceAll(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double dr, const double t)
{
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        t1grid.back() = t;
        double tdiff = (t1grid[t1grid.size() - 1] - t1grid[t1grid.size() - 2]);

        if (t1grid.size() > 2) {
            delta_t_ratio.back() = tdiff /
                (t1grid[t1grid.size() - 2] - t1grid[t1grid.size() - 3]);
        }
        else {
            delta_t_ratio.back() = 0.0;
        }

        for (size_t i = 0; i < replaceLength; i++)
        {
            QKv[length + i] = qK[i];
            QRv[length + i] = qR[i];
            dQKv[length + i] = tdiff * dqK[i];
            dQRv[length + i] = tdiff * dqR[i];
        }

        drvec.back() = tdiff * dr;
        rvec.back() = rstep();
    }
}

void replaceAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{   
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = sim->d_QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        sim->d_t1grid.back() = t;
        double tdiff = (sim->d_t1grid[sim->d_t1grid.size() - 1] - sim->d_t1grid[sim->d_t1grid.size() - 2]);

        if (sim->d_t1grid.size() > 2) {
            sim->d_delta_t_ratio.back() = tdiff /
                (sim->d_t1grid[sim->d_t1grid.size() - 2] - sim->d_t1grid[sim->d_t1grid.size() - 3]);
        }
        else {
            sim->d_delta_t_ratio.back() = 0.0;
        }

        thrust::copy(qK.begin(), qK.end(), sim->d_QKv.begin() + length);
        thrust::copy(qR.begin(), qR.end(), sim->d_QRv.begin() + length);
        thrust::transform(dqK.begin(), dqK.end(), sim->d_dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
        thrust::transform(dqR.begin(), dqR.end(), sim->d_dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

        sim->d_drvec.back() = tdiff * dr;
        sim->d_rvec.back() = rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, Gamma, T0, pool);
    }
}

void replaceAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool)
{   
    const int threads = 64;
    const int blocks = (len + threads - 1) / threads;

    // Replace the existing values in the vectors with the new values
    const size_t gridSize = sim->d_t1grid.size();
    const double tprev1 = sim->d_t1grid[gridSize - 2];
    const double tdiff = t - tprev1;

    sim->d_t1grid.back() = t;

    if (gridSize > 2) {
        const double tprev2 = sim->d_t1grid[gridSize - 3];
        sim->d_delta_t_ratio.back() = tdiff / (tprev1 - tprev2);
    } else {
        sim->d_delta_t_ratio.back() = 0.0;
    }

    const size_t offset = sim->d_QKv.size() - len;

    // thrust::copy(qK, qK + len, sim->d_QKv.begin() + length);
    // thrust::copy(qR, qR + len, sim->d_QRv.begin() + length);
    // thrust::transform(dqK, dqK + len, sim->d_dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
    // thrust::transform(dqR, dqR + len, sim->d_dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

    computeCopy<<<blocks, threads, 0, pool[0]>>>(thrust::raw_pointer_cast(qK), thrust::raw_pointer_cast(sim->d_QKv.data()), offset, len, 1.0);
    computeCopy<<<blocks, threads, 0, pool[1]>>>(thrust::raw_pointer_cast(qR), thrust::raw_pointer_cast(sim->d_QRv.data()), offset, len, 1.0);
    computeCopy<<<blocks, threads, 0, pool[2]>>>(thrust::raw_pointer_cast(dqK), thrust::raw_pointer_cast(sim->d_dQKv.data()), offset, len, tdiff);
    computeCopy<<<blocks, threads, 0, pool[3]>>>(thrust::raw_pointer_cast(dqR), thrust::raw_pointer_cast(sim->d_dQRv.data()), offset, len, tdiff);

    sim->d_drvec.back() = tdiff * dr;
    sim->d_rvec.back() = rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, Gamma, T0, pool);
}

vector<double> bsearchPosSorted(const vector<double>& list, const vector<double>& elem) {
    vector<double> result(elem.size());
    size_t n0, n1, m;
    double Lm, El;
    for (size_t l = 0; l < elem.size(); ++l) {
        El = elem[l];
        n0 = 1; // Start index (Mathematica uses 1-based indexing, C++ uses 0-based)
        n1 = list.size(); // End index
        m = 1; // Midpoint index
        Lm = 0.0;

        // Binary search
        if (list[m - 1] < El) {
            while (n0 <= n1) {
                m = (n0 + n1) / 2; // Compute midpoint
                Lm = list[m - 1];
                if (Lm == El) {
                    n0 = m;
                    break;
                }
                if (Lm < El) {
                    n0 = m + 1;
                }
                else {
                    n1 = m - 1;
                }
            }
        }

        n1 = list.size(); // Reset n1
        if (Lm <= El) {
            if (m == list.size()) {
                result[l] = m;
            }
            else {
                result[l] = m + (El - Lm) / (list[m] - Lm);
            }
        }
        else {
            if (m > 1) {
                result[l] = m - (El - Lm) / (list[m - 2] - Lm);
            }
            else {
                result[l] = m;
            }
        }
    }

    return move(result);
}

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
    cudaStream_t stream = 0)
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

    // cudaDeviceSynchronize(); // Optional, remove if used asynchronously
    return move(result);
}

void bsearchPosSortedGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    thrust::device_vector<double>& result,
    cudaStream_t stream = 0)
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

    // cudaDeviceSynchronize(); // Optional, remove if used asynchronously
}

vector<double> isearchPosSortedInit(const vector<double>& list, const vector<double>& elem, const vector<double>& inits) {
    vector<double> result(elem.size(), 0.0); // Initialize output vector

    for (size_t k = 0; k < inits.size()/len; ++k)
    { 
        size_t length = list.size();
        double l0, l1, Lm, El;
        size_t n0, n1, m;
        bool even;
        double temp=length;
        // Iterate over `elem` in reverse order
        for (size_t l = len; l-- > 0;) 
        {
            El = list.back() * elem[k*len+l];
            n1 = min(static_cast<size_t>(ceil(temp)), length);
            n0 = static_cast<size_t>(floor(inits[k*len+l]));
            m = min(n0 + 1, length);
            even = true;
    
            if ((Lm = list[m - 1]) > El) {
                n1 = max(static_cast<size_t>(m) - 2, (size_t)1);
            }
    
            // Perform the search
            while ((l0 = list[n0 - 1]) <= El && El <= (l1 = list[n1 - 1]) && n0 < n1) {
                even = !even;
                if (even) {
                    m = n0 + round((El - l0) / (l1 - l0) * (n1 - n0));
                }
                else {
                    m = (n0 + n1) / 2;
                }
                Lm = list[m - 1];
                if (Lm == El) {
                    n0 = m;
                    n1 = m - 1;
                }
                else if (Lm < El) {
                    n0 = m + 1;
                }
                else {
                    n1 = m - 1;
                }
            }
    
            // Compute the output value
            if (Lm <= El) {
                if (m == length) {
                    temp = m;
                }
                else {
                    temp = m + (El - Lm) / (list[m] - Lm);
                }
            }
            else {
                if (m > 1) {
                    temp = m - (El - Lm) / (list[m - 2] - Lm);
                }
                else {
                    temp = m;
                }
            }
            result[k*len+l] = temp;
        }
    } 

    return move(result);
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

void interpolate(const vector<double>& posB1xIn = {}, const vector<double>& posB2xIn = {},
    const bool same = false)
{
    // Compute posB1x
    vector<double> posB1x = !posB1xIn.empty() ?
        (same ? posB1xIn : isearchPosSortedInit(t1grid, theta, posB1xIn)) :
        bsearchPosSorted(t1grid, theta * t1grid.back());

    // Compute posB2x
    vector<double> posB2x = !posB2xIn.empty() ?
        (same ? posB2xIn : isearchPosSortedInit(t1grid, phi2, posB2xIn)) :
        bsearchPosSorted(t1grid, phi2 * t1grid.back());

    // Update old positions
    posB1xOld = posB1x;
    posB2xOld = posB2x;

    // Interpolate QKA1int and QRA1int
    if (t1grid.back() > 0) {
        indexVecLN3(weightsA1y, indsA1y, QKA1int, QRA1int);
    }
    else {
        QKA1int.assign(len * len, QKv[0]);
        QRA1int.assign(len * len, QRv[0]);
    }
    SigmaK(QKA1int, SigmaKA1int);
    SigmaR(QKA1int, QRA1int, SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (t1grid.back() > 0) {
        indexVecLN3(weightsA2y, indsA2y, QKA2int, QRA2int);
    }
    else {
        QKA2int.assign(len * len, QKv[0]);
        QRA2int.assign(len * len, QRv[0]);
    }
    SigmaR(QKA2int, QRA2int, SigmaRA2int);

    // Interpolate QKB1int and QRB1int
    // Compute `floor` vector
    double maxPosB1x = posB1x[0];
    for (size_t i = 1; i < posB1x.size(); ++i) {
        if (posB1x[i] > maxPosB1x) {
            maxPosB1x = posB1x[i];
        }
    }
    size_t maxCeil = static_cast<size_t>(ceil(maxPosB1x)) - 1;
    if (maxCeil < 1) {
        maxCeil = 1;
    }

    // Compute FLOOR vector
    vector<size_t> Floor(posB1x.size());
    for (size_t i = 0; i < posB1x.size(); ++i) {
        size_t flooredValue = static_cast<size_t>(floor(posB1x[i]));
        if (flooredValue < 1) {
            flooredValue = 1;
        }
        else if (flooredValue > maxCeil) {
            flooredValue = maxCeil;
        }
        Floor[i] = flooredValue;
    }

    // Compute `diff` vector
    vector<double> diff(posB1x.size());
    // Subtract(vector<double>(Floor.begin(), Floor.end()), posB1x, diff);
    diff = vector<double>(Floor.begin(), Floor.end()) - posB1x;

    if (t1grid.back() > 0) {
        indexVecN(len, diff, Floor, delta_t_ratio, QKB1int, QRB1int);
    }
    else {
        QKB1int.assign(len * len, QKv[0]);
        QRB1int.assign(len * len, QRv[0]);
    }
    SigmaK(QKB1int, SigmaKB1int);
    SigmaR(QKB1int, QRB1int, SigmaRB1int);

    // Interpolate QKB2int and QRB2int
    if (t1grid.back() > 0) {
        indexMatAll(posB2x, indsB2y, weightsB2y, delta_t_ratio, QKB2int, QRB2int);
    }
    else {
        QKB2int.assign(len * len, QKv[0]);
        QRB2int.assign(len * len, QRv[0]);
    }
    SigmaK(QKB2int, SigmaKB2int);
    SigmaR(QKB2int, QRB2int, SigmaRB2int);

    // Interpolate rInt
    if (t1grid.back() > 0) {
        indexVecR2(rvec, drvec, diff, Floor, delta_t_ratio, rInt);
    }
    else {
        rInt.assign(len, rvec[0]);
    }
}

__global__ void diffNfloorKernel(
    const double* __restrict__ posB1x,
    size_t* __restrict__ Floor,
    double* __restrict__ diff,
    size_t len)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    double pos = posB1x[i];

    // Floor with bounds check done purely inside the kernel
    size_t floored = static_cast<size_t>(floor(pos));

    // Instead of precomputing maxCeil, compute it on-the-fly from pos
    size_t maxCeil = static_cast<size_t>(ceil(pos)); // conservative per-thread local bound
    floored = max(size_t(1), min(floored, maxCeil - 1));

    Floor[i] = floored;
    diff[i] = static_cast<double>(floored) - pos;
}

void diffNfloor(
    const thrust::device_vector<double>& posB1x,
    thrust::device_vector<size_t>& Floor,
    thrust::device_vector<double>& diff,
    cudaStream_t stream = 0)
{
    size_t len = posB1x.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    diffNfloorKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(posB1x.data()),
        thrust::raw_pointer_cast(Floor.data()),
        thrust::raw_pointer_cast(diff.data()),
        len
    );
}

// void interpolateGPU(
//     const double* posB1xIn = nullptr,
//     const double* posB2xIn = nullptr,
//     const bool same = false,
//     StreamPool* pool = nullptr) {

//     size_t len = theta.size();
//     int threads = 64;
//     int blocks = (len*len + threads - 1) / threads;

//     if (!pool) pool = &getDefaultStreamPool();

//     // Compute sim->d_posB1x
//     if (posB1xIn) {
//         if (same) {
//             sim->d_posB1xOld = thrust::device_vector<double>(posB1xIn, posB1xIn + len);
//         } else {
//             bsearchPosSortedGPU(sim->d_t1grid, sim->d_theta, sim->d_posB1xOld, (*pool)[0]);
//         }
//     } else {
//         bsearchPosSortedGPU(sim->d_t1grid, sim->d_theta, sim->d_posB1xOld, (*pool)[0]);
//     }

//     // Compute sim->d_posB2x
//     if (posB2xIn) {
//         if (same) {
//             sim->d_posB2xOld = thrust::device_vector<double>(posB2xIn, posB2xIn + len * len);
//         } else {
//             bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);
//         }
//     } else {
//         bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);
//     }

//     // Interpolate QKA1int and QRA1int
//     if (sim->d_t1grid.back() > 0) {
//         indexVecLN3GPU(sim->d_weightsA1y, sim->d_indsA1y, sim->d_QKA1int, sim->d_QRA1int,(*pool)[2]);
//     } else {
//         sim->d_QKA1int.assign(len * len, sim->d_QKv[0]);
//         sim->d_QRA1int.assign(len * len, sim->d_QRv[0]);
//     }
//     // SigmaKGPU(sim->d_QKA1int, sim->d_SigmaKA1int,(*pool)[2]);
//     // SigmaRGPU(sim->d_QKA1int, sim->d_QRA1int, sim->d_SigmaRA1int,(*pool)[2]);

//     // Interpolate QKA2int and QRA2int
//     if (sim->d_t1grid.back() > 0) {
//         indexVecLN3GPU(sim->d_weightsA2y, sim->d_indsA2y, sim->d_QKA2int, sim->d_QRA2int,(*pool)[3]);
//     } else {
//         sim->d_QKA2int.assign(len * len, sim->d_QKv[0]);
//         sim->d_QRA2int.assign(len * len, sim->d_QRv[0]);
//     }
//     // SigmaRGPU(sim->d_QKA2int, sim->d_QRA2int, sim->d_SigmaRA2int,(*pool)[3]);

//     // Interpolate QKB1int and QRB1int
//     // thrust::device_vector<size_t> Floor(sim->d_posB1xOld.size());
//     // thrust::device_vector<double> diff(Floor.size());

//     diffNfloor(sim->d_posB1xOld, sim->Stemp0, sim->temp0, (*pool)[0]);

//     if (sim->d_t1grid.back() > 0) {
//         indexVecNGPU(sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_QKB1int, sim->d_QRB1int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[0]);
//     } else {
//         sim->d_QKB1int.assign(len * len, sim->d_QKv[0]);
//         sim->d_QRB1int.assign(len * len, sim->d_QRv[0]);
//     }
//     // SigmaKGPU(sim->d_QKB1int, sim->d_SigmaKB1int,(*pool)[0]);
//     // SigmaRGPU(sim->d_QKB1int, sim->d_QRB1int, sim->d_SigmaRB1int,(*pool)[0]);

//     // Interpolate QKB2int and QRB2int
//     if (sim->d_t1grid.back() > 0) {
//         indexMatAllGPU(sim->d_posB2xOld, sim->d_indsB2y, sim->d_weightsB2y, sim->d_delta_t_ratio, sim->d_QKB2int, sim->d_QRB2int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[1]);
//     } else {
//         sim->d_QKB2int.assign(len * len, sim->d_QKv[0]);
//         sim->d_QRB2int.assign(len * len, sim->d_QRv[0]);
//     }
//     // SigmaKGPU(sim->d_QKB2int, sim->d_SigmaKB2int,(*pool)[1]);
//     // SigmaRGPU(sim->d_QKB2int, sim->d_QRB2int, sim->d_SigmaRB2int,(*pool)[1]);

//     // Interpolate rInt
//     if (sim->d_t1grid.back() > 0) {
//         indexVecR2GPU(sim->d_rvec, sim->d_drvec, sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_rInt, (*pool)[0]);
//     } else {
//         sim->d_rInt.assign(len, sim->d_rvec[0]);
//     }

//     computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[2]>>>(
//         sim->d_QKA1int.data().get(),
//         sim->d_QRA1int.data().get(),
//         sim->d_SigmaKA1int.data().get(),
//         sim->d_SigmaRA1int.data().get(),
//         len*len);
//     computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[3]>>>(
//         sim->d_QKA2int.data().get(),
//         sim->d_QRA2int.data().get(),
//         sim->d_SigmaKA2int.data().get(),
//         sim->d_SigmaRA2int.data().get(),
//         len*len);
//     computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[0]>>>(
//         sim->d_QKB1int.data().get(),
//         sim->d_QRB1int.data().get(),
//         sim->d_SigmaKB1int.data().get(),
//         sim->d_SigmaRB1int.data().get(),
//         len*len);
//     computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[1]>>>(
//         sim->d_QKB2int.data().get(),
//         sim->d_QRB2int.data().get(),
//         sim->d_SigmaKB2int.data().get(),
//         sim->d_SigmaRB2int.data().get(),
//         len*len);
// }

void interpolateGPU(
    const double* posB1xIn = nullptr,
    const double* posB2xIn = nullptr,
    const bool same = false,
    StreamPool* pool = nullptr) {

    size_t len = theta.size();
    int threads = 64;
    int blocks = (len*len + threads - 1) / threads;

    if (!pool) pool = &getDefaultStreamPool();

    // Compute sim->d_posB1x
    bsearchPosSortedGPU(sim->d_t1grid, sim->d_theta, sim->d_posB1xOld, (*pool)[0]);

    // Compute sim->d_posB2x
    bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);

    // Interpolate QKA1int and QRA1int
    indexVecLN3GPU(sim->d_weightsA1y, sim->d_indsA1y, sim->d_QKA1int, sim->d_QRA1int, (*pool)[2]);

    // Interpolate QKA2int and QRA2int
    indexVecLN3GPU(sim->d_weightsA2y, sim->d_indsA2y, sim->d_QKA2int, sim->d_QRA2int, (*pool)[3]);

    // Interpolate QKB1int and QRB1int
    diffNfloor(sim->d_posB1xOld, sim->Stemp0, sim->temp0, (*pool)[0]);
    indexVecNGPU(sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_QKB1int, sim->d_QRB1int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[0]);

    // Interpolate QKB2int and QRB2int
    indexMatAllGPU(sim->d_posB2xOld, sim->d_indsB2y, sim->d_weightsB2y, sim->d_delta_t_ratio, sim->d_QKB2int, sim->d_QRB2int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[1]);

    // Interpolate rInt
    indexVecR2GPU(sim->d_rvec, sim->d_drvec, sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_rInt, (*pool)[0]);

    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[2]>>>(
        sim->d_QKA1int.data().get(),
        sim->d_QRA1int.data().get(),
        sim->d_SigmaKA1int.data().get(),
        sim->d_SigmaRA1int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[3]>>>(
        sim->d_QKA2int.data().get(),
        sim->d_QRA2int.data().get(),
        sim->d_SigmaKA2int.data().get(),
        sim->d_SigmaRA2int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[0]>>>(
        sim->d_QKB1int.data().get(),
        sim->d_QRB1int.data().get(),
        sim->d_SigmaKB1int.data().get(),
        sim->d_SigmaRB1int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[1]>>>(
        sim->d_QKB2int.data().get(),
        sim->d_QRB2int.data().get(),
        sim->d_SigmaKB2int.data().get(),
        sim->d_SigmaRB2int.data().get(),
        len*len);
}

auto gather = [](const std::vector<double>& v,
                 const std::vector<size_t>& idxs,
                 size_t len,
                 const std::vector<double>& scale = {}) {
    std::vector<double> out(idxs.size() * len);
    for (size_t i = 0; i < idxs.size(); ++i) {
        size_t offset = idxs[i] * len;
        double factor = (scale.empty() ? 1.0 : scale[i]);
        for (size_t j = 0; j < len; ++j) {
            out[i * len + j] = factor * v[offset + j];
        }
    }
    return move(out);
};

void sparsifyNscale(double threshold) {
    bool erased = false;
    int loop = 0;
    std::vector<size_t> inds = {0};
    inds.reserve(t1grid.size());

    for (size_t i = 2; i + 1 < t1grid.size(); ++i) {
        double tleft = t1grid[i - 2];
        double tmid  = t1grid[i];
        double tdiff1 = t1grid[i - 1] - tleft;
        double tdiff2 = tmid - tleft;
        double tdiff3 = t1grid[i + 1] - tmid;

        double val = 0.0;
        for (int j = 0; j < len; ++j) {
            double df_term1 = dQKv[(i - 1) * len + j];
            double df_term2 = dQKv[(i + 1) * len + j];
            double f_term1 = QKv[i * len + j] - QKv[(i - 2) * len + j];
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        for (int j = 0; j < len; ++j) {
            double df_term1 = dQRv[(i - 1) * len + j];
            double df_term2 = dQRv[(i + 1) * len + j];
            double f_term1 = QRv[i * len + j] - QRv[(i - 2) * len + j];
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        if (val < threshold && !erased) {
            erased = true;
            ++loop;
        } else {
            erased = false;
            ++loop;
            inds.push_back(loop);
        }
    }

    inds.push_back(t1grid.size() - 2);
    inds.push_back(t1grid.size() - 1);

    // Calculate rotated and modded indices
    std::vector<size_t> indsD(inds.size());
    indsD[0] = 0; // First index remains the same
    for (size_t i = 0; i < inds.size() - 1; ++i) {
        indsD[i + 1] = inds[i] + 1;
    }

    // Compute tfac
    std::vector<double> tfac(inds.size());
    tfac[0] = 1.0;
    for (size_t i = 1; i < inds.size(); ++i) {
        tfac[i] = (t1grid[inds[i]] - t1grid[inds[i - 1]]) / (t1grid[indsD[i]] - t1grid[indsD[i] - 1]);
    }

    // printVectorDifference(gather(rvec, inds, 1),rvec);

    QKv   = gather(QKv, inds, len);
    QRv   = gather(QRv, inds, len);
    dQKv  = gather(dQKv, indsD, len, tfac);
    dQRv  = gather(dQRv, indsD, len, tfac);
    rvec = gather(rvec, inds, 1);
    drvec = gather(drvec, indsD, 1, tfac);
    t1grid = gather(t1grid, inds, 1);

    // Δt ratio
    std::vector<double> dgrid(inds.size());
    for (size_t i = 1; i < t1grid.size(); ++i)
        dgrid[i] = t1grid[i] - t1grid[i - 1];
    for (size_t i = 2; i < t1grid.size(); ++i)
        delta_t_ratio[i] = dgrid[i] / dgrid[i - 1];

    delta_t_ratio.resize(inds.size());    

    interpolate();
}

// GPU gather kernel
__global__ void gatherKernel(const double* __restrict__ v,
                            const size_t* __restrict__ idxs,
                            const double* __restrict__ scale,
                            double* __restrict__ out,
                            size_t len, size_t n_chunks, bool use_scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks * len) return;

    size_t chunk = i / len;
    size_t j = i % len;

    size_t offset = idxs[chunk] * len;
    double factor = use_scale ? scale[chunk] : 1.0;
    out[i] = factor * v[offset + j];
}

thrust::device_vector<double> gatherGPU(const thrust::device_vector<double>& v,
                                        const thrust::device_vector<size_t>& idxs,
                                        size_t len,
                                        const thrust::device_vector<double>& scale = {},
                                        cudaStream_t stream = 0) {
    size_t n_chunks = idxs.size();
    thrust::device_vector<double> out(n_chunks * len);

    size_t threads = 64;
    size_t blocks = (n_chunks * len + threads - 1) / threads;

    bool use_scale = !scale.empty();
    gatherKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(v.data()),
        thrust::raw_pointer_cast(idxs.data()),
        use_scale ? thrust::raw_pointer_cast(scale.data()) : nullptr,
        thrust::raw_pointer_cast(out.data()),
        len, n_chunks, use_scale);

    return move(out);
}

__global__ void computeSparsifyFlags(const double* __restrict__ t1grid,
                                     const double* __restrict__ QKv,
                                     const double* __restrict__ QRv,
                                     const double* __restrict__ dQKv,
                                     const double* __restrict__ dQRv,
                                     bool* __restrict__ flags,
                                     double threshold, size_t len, size_t n,
                                     cudaStream_t stream = 0) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i + 1 >= n) return;
    if (i/2 * 2 != i) return; // Ensure i is even

    double tleft = t1grid[i - 2];
    double tmid = t1grid[i];
    double tdiff1 = t1grid[i - 1] - tleft;
    double tdiff2 = tmid - tleft;
    double tdiff3 = t1grid[i + 1] - tmid;
    double scale = tdiff2 / 12.0;

    double val = 0.0;
    for (size_t j = 0; j < len; ++j) {
        size_t idx_im2 = (i - 2) * len + j;
        size_t idx_im1 = (i - 1) * len + j;
        size_t idx_i   = i * len + j;
        size_t idx_ip1 = (i + 1) * len + j;

        double df1_qk = dQKv[idx_im1];
        double df2_qk = dQKv[idx_ip1];
        double f_qk = QKv[idx_i] - QKv[idx_im2];

        double df1_qr = dQRv[idx_im1];
        double df2_qr = dQRv[idx_ip1];
        double f_qr = QRv[idx_i] - QRv[idx_im2];

        val += fabs(scale * (2.0 * f_qk - tdiff2 * (df1_qk / tdiff1 + df2_qk / tdiff3)));
        val += fabs(scale * (2.0 * f_qr - tdiff2 * (df1_qr / tdiff1 + df2_qr / tdiff3)));
    }
    flags[i] = (val >= threshold);
}

void sparsifyNscaleGPU(double threshold, cudaStream_t stream = 0) {

    size_t t1len = sim->d_t1grid.size();
    thrust::device_vector<bool> flags(t1len, true);

    size_t threads = 64;
    size_t blocks = (t1len - 2 + threads - 1) / threads;
    computeSparsifyFlags<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        thrust::raw_pointer_cast(sim->d_QKv.data()),
        thrust::raw_pointer_cast(sim->d_QRv.data()),
        thrust::raw_pointer_cast(sim->d_dQKv.data()),
        thrust::raw_pointer_cast(sim->d_dQRv.data()),
        thrust::raw_pointer_cast(flags.data()),
        threshold, len, t1len);

    thrust::device_vector<size_t> inds(t1len);
    thrust::sequence(inds.begin(), inds.end());

    size_t n = inds.size();
    thrust::device_vector<size_t> filtered(t1len); // max possible size
    auto end_it = thrust::copy_if(
        inds.begin(), inds.end(), 
        flags.begin(), 
        filtered.begin(), 
        thrust::identity<bool>()
    );
    filtered.resize(end_it - filtered.begin()); // shrink to actual size

    // Construct d_indsD and tfac
    thrust::device_vector<size_t> indsD(filtered.size(),0);
    thrust::transform(filtered.begin(), filtered.end() - 1, indsD.begin() + 1, indsD.begin() + 1, thrust::placeholders::_1 + 1);

    thrust::device_vector<double> tfac(filtered.size(), 1.0);
    const double* t1_ptr = thrust::raw_pointer_cast(sim->d_t1grid.data());

    auto max_it = thrust::max_element(indsD.begin(), indsD.end());
    auto min_it = thrust::min_element(indsD.begin(), indsD.end());
    double max_val = *max_it;
    double min_val = *min_it;

    auto max_f = thrust::max_element(filtered.begin(), filtered.end());
    auto min_f = thrust::min_element(filtered.begin(), filtered.end());
    double max_valf = *max_f;
    double min_valf = *min_f;

    tfac[0]= 1.0;
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            filtered.begin() + 1,       // inds[i]
            filtered.begin(),           // inds[i-1]
            indsD.begin() + 1      // indsD[i]
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            filtered.end(),             // one past last valid index
            filtered.end() - 1,
            indsD.end()
        )),
        tfac.begin() + 1,
        [t1_ptr] __device__ (thrust::tuple<size_t, size_t, size_t> tup) {
            size_t inds_i    = thrust::get<0>(tup);
            size_t inds_im1  = thrust::get<1>(tup);
            size_t indsD_i   = thrust::get<2>(tup);
            return (t1_ptr[inds_i] - t1_ptr[inds_im1]) /
                (t1_ptr[indsD_i] - t1_ptr[indsD_i - 1]);
        });

    sim->d_QKv = gatherGPU(sim->d_QKv, filtered, len);
    sim->d_QRv = gatherGPU(sim->d_QRv, filtered, len);
    sim->d_dQKv = gatherGPU(sim->d_dQKv, indsD, len, tfac);
    sim->d_dQRv = gatherGPU(sim->d_dQRv, indsD, len, tfac);
    sim->d_rvec = gatherGPU(sim->d_rvec, filtered, 1);
    sim->d_drvec = gatherGPU(sim->d_drvec, indsD, 1, tfac);
    sim->d_t1grid = gatherGPU(sim->d_t1grid, filtered, 1);

    size_t new_n = sim->d_t1grid.size();
    sim->d_delta_t_ratio.resize(new_n);
    thrust::device_vector<double> dgrid(new_n);
    thrust::transform(sim->d_t1grid.begin() + 1, sim->d_t1grid.end(), sim->d_t1grid.begin(), dgrid.begin() + 1, thrust::minus<>());
    thrust::transform(dgrid.begin() + 2, dgrid.end(), dgrid.begin() + 1, sim->d_delta_t_ratio.begin() + 2, thrust::divides<>());

    // vector<double> gpu_result(tfac.size());
    // thrust::copy(tfac.begin(), tfac.end(), gpu_result.begin());

    interpolateGPU();
}

double SSPRK104()
{
    const size_t stages = 10;
    const double amat[stages][stages] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0}
    };
    const double bvec[stages] = { 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10 };
    const double b2vec[stages] = { 0., 2.0 / 9, 0, 0, 5.0 / 18, 1.0 / 3, 0., 0., 0., 1.0 / 6 };

    // Initialize variables
    vector<vector<double>> gKvec(stages + 1, vector<double>(len, 0.0));
    gKvec[0] = getLastLenEntries(QKv, len);
    vector<vector<double>> gRvec(stages + 1, vector<double>(len, 0.0));
    gRvec[0] = getLastLenEntries(QRv, len);
    vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = t1grid.back();

    vector<double> gKe(len, 0.0);
    vector<double> gRe(len, 0.0);
    double gte = 0.0;

    vector<vector<double>> hKvec(stages, vector<double>(len, 0.0));
    vector<vector<double>> hRvec(stages, vector<double>(len, 0.0));
    vector<double> htvec(stages, 0.0);

    vector<vector<double>> posB1xvec(3, vector<double>(len, 0.0));
    vector<vector<double>> posB2xvec(3, vector<double>(len * len, 0.0));
    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (QKv.size() == len || n != 0) {
            interpolate(
                (n == 0 ? vector<double>{} : (n == 5 ? posB1xvec[0] : (n == 6 ? posB1xvec[1] : (n == 7 ? posB1xvec[2] : posB1xOld)))),
                (n == 0 ? vector<double>{} : (n == 5 ? posB2xvec[0] : (n == 6 ? posB2xvec[1] : (n == 7 ? posB2xvec[2] : posB2xOld)))),
                (n == 5 || n == 6 || n == 7)
            );
        }

        // Update position vectors
        if (n == 2) {
            posB1xvec[0] = posB1xOld;
            posB2xvec[0] = posB2xOld;
        }
        else if (n == 3) {
            posB1xvec[1] = posB1xOld;
            posB2xvec[1] = posB2xOld;
        }
        else if (n == 4) {
            posB1xvec[2] = posB1xOld;
            posB2xvec[2] = posB2xOld;
        }


        // Compute k[n]
        // if (n == 1) {
        //     double sum = std::accumulate(QKB1int.begin(), QKB1int.end(), 0.0);
        //     cout << "Sum of QKB1int: " << sum - QKB1int.size() << endl;
        //     sum = std::accumulate(QKB2int.begin(), QKB2int.end(), 0.0);
        //     cout << "Sum of QKB2int: " << sum - QKB2int.size() << endl;
        //     sum = std::accumulate(QKA1int.begin(), QKA1int.end(), 0.0);
        //     cout << "Sum of QKA1int: " << sum - QKA1int.size() << endl;
        //     sum = std::accumulate(QRA1int.begin(), QRA1int.end(), 0.0);
        //     cout << "Sum of QRA1int: " << sum - QRA1int.size() << endl;
        //     sum = std::accumulate(QRA2int.begin(), QRA2int.end(), 0.0);
        //     cout << "Sum of QRA2int: " << sum - QRA2int.size() << endl;
        //     sum = std::accumulate(QRB1int.begin(), QRB1int.end(), 0.0);
        //     cout << "Sum of QRB1int: " << sum - QRB1int.size() << endl;
        //     sum = std::accumulate(indsB2y.begin(), indsB2y.end(), 0.0);
        //     cout << "Sum of indsB2y: " << sum << endl;
        //     temp_cpu = weightsB2y;
        // }
        hKvec[n] = QKstep();
        hRvec[n] = QRstep();
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            vector<double> lastQKv = getLastLenEntries(QKv, len);
            vector<double> lastQRv = getLastLenEntries(QRv, len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], t1grid.back());
            gKvec[n + 1] = gKvec[0] + hKvec[0] * (delta_t * amat[1][0]);
            gRvec[n + 1] = gRvec[0] + hRvec[0] * (delta_t * amat[1][0]);
            gtvec[n + 1] = gtvec[0] + delta_t * amat[1][0] * htvec[0];
            appendAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); //Append Update

        }
        else if (n == stages - 1) {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                gKvec[n + 1] += hKvec[j] * (delta_t * bvec[j]);
                gRvec[n + 1] += hRvec[j] * (delta_t * bvec[j]);
                gtvec[n + 1] += delta_t * bvec[j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
        else {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                gKvec[n + 1] += hKvec[j] * (delta_t * amat[n + 1][j]);
                gRvec[n + 1] += hRvec[j] * (delta_t * amat[n + 1][j]);
                gtvec[n + 1] += delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
        // double sum = std::accumulate(QKv.begin(), QKv.end(), 0.0);
        // cout << "Sum of QKv: " << sum - t1grid.size() * len << endl;
        // sum = std::accumulate(hKvec[n].begin(), hKvec[n].end(), 0.0);
        // cout << "Sum of gKvec: " << sum - 0 << endl;
    }

    // Final interpolation
    interpolate(posB1xOld, posB2xOld);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (delta_t * b2vec[j]);
        gRe += hRvec[j] * (delta_t * b2vec[j]);
        gte += delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    double error = 0.0;
    for (size_t i = 0; i < gKvec[stages].size(); ++i) {
        error += abs(gKvec[stages][i] - gKe[i]);
    }
    for (size_t i = 0; i < gRvec[stages].size(); ++i) {
        error += abs(gRvec[stages][i] - gRe[i]);
    }
    error += abs(gtvec[stages] - gte);

    //return error;
    return error;
}

double RK54()
{
    const size_t stages = 7;
    const double amat[stages][stages] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.0 / 40, 9.0 / 40, 0.0, 0.0, 0.0, 0.0, 0.0},
        {44.0 / 45, -56.0 / 15, 32.0 / 9, 0.0, 0.0, 0.0, 0.0},
        {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 0.0, 0.0, 0.0},
        {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 0.0, 0.0},
        {35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0}
    };
    const double bvec[stages] = { 35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0 };
    const double b2vec[stages] = { 5179.0/57600, 0.0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40 };

    // Initialize variables
    vector<vector<double>> gKvec(stages + 1, vector<double>(len, 0.0));
    gKvec[0] = getLastLenEntries(QKv, len);
    vector<vector<double>> gRvec(stages + 1, vector<double>(len, 0.0));
    gRvec[0] = getLastLenEntries(QRv, len);
    vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = t1grid.back();

    vector<double> gKe(len, 0.0);
    vector<double> gRe(len, 0.0);
    double gte = 0.0;

    vector<vector<double>> hKvec(stages, vector<double>(len, 0.0));
    vector<vector<double>> hRvec(stages, vector<double>(len, 0.0));
    vector<double> htvec(stages, 0.0);

    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (QKv.size() == len || n != 0) {
            interpolate(
                (n == 0 ? vector<double>{} : posB1xOld),
                (n == 0 ? vector<double>{} : posB2xOld),
                false
            );
        }

        hKvec[n] = QKstep();
        hRvec[n] = QRstep();
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            vector<double> lastQKv = getLastLenEntries(QKv, len);
            vector<double> lastQRv = getLastLenEntries(QRv, len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], t1grid.back());
            gKvec[n + 1] = gKvec[0] + hKvec[0] * (delta_t * amat[1][0]);
            gRvec[n + 1] = gRvec[0] + hRvec[0] * (delta_t * amat[1][0]);
            gtvec[n + 1] = gtvec[0] + delta_t * amat[1][0] * htvec[0];
            appendAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); //Append Update
        }
        else if (n == stages - 1) {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                gKvec[n + 1] += hKvec[j] * (delta_t * bvec[j]);
                gRvec[n + 1] += hRvec[j] * (delta_t * bvec[j]);
                gtvec[n + 1] += delta_t * bvec[j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
        else {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                gKvec[n + 1] += hKvec[j] * (delta_t * amat[n + 1][j]);
                gRvec[n + 1] += hRvec[j] * (delta_t * amat[n + 1][j]);
                gtvec[n + 1] += delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
        // double sum = std::accumulate(QKv.begin(), QKv.end(), 0.0);
        // cout << "Sum of QKv: " << sum - t1grid.size() * len << endl;
        // sum = std::accumulate(hKvec[n].begin(), hKvec[n].end(), 0.0);
        // cout << "Sum of gKvec: " << sum - 0 << endl;
    }

    // Final interpolation
    interpolate(posB1xOld, posB2xOld, true);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (delta_t * b2vec[j]);
        gRe += hRvec[j] * (delta_t * b2vec[j]);
        gte += delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    double error = 0.0;
    for (size_t i = 0; i < gKvec[stages].size(); ++i) {
        error += abs(gKvec[stages][i] - gKe[i]);
    }
    for (size_t i = 0; i < gRvec[stages].size(); ++i) {
        error += abs(gRvec[stages][i] - gRe[i]);
    }
    error += abs(gtvec[stages] - gte);

    //return error;
    return error;
}

void init_RK54GPU() {
    rk->init = 1;
    rk->stages = 7;

    rk->avec = new double[rk->stages * (rk->stages + 1) / 2 + 1] {1.0 / 5, 3.0 / 40, 9.0 / 40, 44.0 / 45, 
        -56.0 / 15, 32.0 / 9, 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 
        9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 35.0/384, 
        0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};

    rk->bvec = new double[rk->stages] { 35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0 };

    rk->b2vec = new double[rk->stages] { 5179.0/57600, 0.0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40 };

    rk->cvec = new double[rk->stages] { 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0};

    rk->d_avec.resize(rk->stages * (rk->stages + 1) / 2 + 1);
    cudaMemcpy(thrust::raw_pointer_cast(rk->d_avec.data()), rk->avec, rk->d_avec.size() * sizeof(double), cudaMemcpyHostToDevice);

    rk->gK0.resize(len, 0.0);
    rk->gR0.resize(len, 0.0);
    rk->gK.resize(len, 0.0);
    rk->gR.resize(len, 0.0);
    rk->gKe.resize(len, 0.0);
    rk->gRe.resize(len, 0.0);
    rk->gKfinal.resize(len, 0.0);
    rk->gRfinal.resize(len, 0.0);

    // rk->posB1xvec.resize(rk->posCount * len, 0.0);
    // rk->posB2xvec.resize(rk->posCount * len * len, 0.0);

    rk->hK.resize(len * rk->stages, 0.0);
    rk->hR.resize(len * rk->stages, 0.0);
    rk->hK0.resize(len, 0.0);
    rk->hR0.resize(len, 0.0);
} 

void init_SSPRK104GPU() {
    rk->stages = 10;
    rk->posCount = 3;
    rk->init = 2;

    rk->avec = new double[rk->stages] {
        1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6,
        1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6
    };

    rk->bvec = new double[rk->stages] {
        1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10,
        1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10
    };

    rk->b2vec = new double[rk->stages] {
        0.0, 2.0 / 9, 0.0, 0.0, 5.0 / 18, 1.0 / 3, 0.0, 0.0, 0.0, 1.0 / 6
    };

    rk->gK0.resize(len, 0.0);
    rk->gR0.resize(len, 0.0);
    rk->gK.resize(len, 0.0);
    rk->gR.resize(len, 0.0);
    rk->gKfinal.resize(len, 0.0);
    rk->gRfinal.resize(len, 0.0);
    rk->gKe.resize(len, 0.0);
    rk->gRe.resize(len, 0.0);

    rk->posB1xvec.resize(rk->posCount * len, 0.0);
    rk->posB2xvec.resize(rk->posCount * len * len, 0.0);

    rk->hK.resize(len, 0.0);
    rk->hR.resize(len, 0.0);
    rk->hK0.resize(len, 0.0);
    rk->hR0.resize(len, 0.0);
}

double RK54GPU(StreamPool* pool = nullptr) {

    size_t t1len = sim->d_t1grid.size();
    double t_current = sim->d_t1grid.back();
    // Initialize variables
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gK.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gR.begin());
    rk->gt = t_current;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gK0.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gR0.begin());
    rk->gt0 = t_current;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gKfinal.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gRfinal.begin());
    rk->gtfinal = t_current + delta_t;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gKe.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gRe.begin());
    rk->gte = t_current + delta_t;

    thrust::fill(sim->error_result.begin(), sim->error_result.end(), 0.0);

    double dr = 0.0;
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->stages; ++n) {
        // Interpolation
        if (sim->d_QKv.size() == len || n != 0) {
            interpolateGPU();
        }

        QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, T0, Gamma, n, *pool);
        rk->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            rk->hK0.assign(rk->hK.begin(), rk->hK.begin() + len);
            rk->hR0.assign(rk->hR.begin(), rk->hR.begin() + len);
            computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK0.data()), thrust::raw_pointer_cast(rk->hK.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gK.data()), delta_t, n + 1, len);
            computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR0.data()), thrust::raw_pointer_cast(rk->hR.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gR.data()), delta_t, n + 1, len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * len, delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * len, delta_t * rk->bvec[n], len);
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * len, delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * len, delta_t * rk->b2vec[n], len);
            }
            rk->gt = rk->gt0 + delta_t * rk->cvec[n] * rk->ht;
            dr = drstep2GPU(get_slice_ptr(sim->d_QKv, t1len - 1, len), get_slice_ptr(sim->d_QRv, t1len - 1, len), rk->hK0.data(), rk->hR0.data(), sim->d_t1grid.back(), T0, *pool);
            appendAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, len, *pool); // Append Update
        } else {
            if (n != rk->stages - 1) {
                computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK0.data()), thrust::raw_pointer_cast(rk->hK.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gK.data()), delta_t, n + 1, len);
                computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR0.data()), thrust::raw_pointer_cast(rk->hR.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gR.data()), delta_t, n + 1, len);
            }
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * len, delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * len, delta_t * rk->bvec[n], len);
            rk->gt = rk->gt0 + delta_t * rk->cvec[n] * rk->ht;
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * len, delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * len, delta_t * rk->b2vec[n], len);
            }
            if (n != rk->stages - 1) {
                replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(rk->gKfinal.data(), rk->gRfinal.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gtfinal, len, *pool); // Replace Update
            }
        }
    }

    // Final interpolation
    interpolateGPU(thrust::raw_pointer_cast(sim->d_posB1xOld.data()), thrust::raw_pointer_cast(sim->d_posB2xOld.data()), true, pool);
 
    // Compute error estimate
    computeError<<<blocks, threads, threads * sizeof(double), (*pool)[0]>>>(
        thrust::raw_pointer_cast(rk->gKfinal.data()),
        thrust::raw_pointer_cast(rk->gKe.data()),
        thrust::raw_pointer_cast(rk->gRfinal.data()),
        thrust::raw_pointer_cast(rk->gRe.data()),
        thrust::raw_pointer_cast(sim->error_result.data()),
        len
    );

    double error = sim->error_result[0];
    error += abs(rk->gtfinal - rk->gte);

    return error;
}

double SSPRK104GPU(StreamPool* pool = nullptr) {

    size_t t1len = sim->d_t1grid.size();
    double t_current = sim->d_t1grid.back();
    // Initialize variables
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gK.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gR.begin());
    rk->gt = t_current;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gK0.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gR0.begin());
    rk->gt0 = t_current;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gKfinal.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gRfinal.begin());
    rk->gtfinal = t_current + delta_t;
    thrust::copy(sim->d_QKv.end() - len, sim->d_QKv.end(), rk->gKe.begin());
    thrust::copy(sim->d_QRv.end() - len, sim->d_QRv.end(), rk->gRe.begin());
    rk->gte = t_current + delta_t;

    thrust::fill(sim->error_result.begin(), sim->error_result.end(), 0.0);
    
    double dr = 0.0;
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->stages; ++n) {
        // Interpolation
        if (sim->d_QKv.size() == len || n != 0) {
            interpolateGPU(
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->posB1xvec, 0, len).get() : (n == 6 ? get_slice_ptr(rk->posB1xvec, 1, len).get() : (n == 7 ? get_slice_ptr(rk->posB1xvec, 2, len).get() : thrust::raw_pointer_cast(sim->d_posB1xOld.data()))))),
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->posB2xvec, 0, len*len).get() : (n == 6 ? get_slice_ptr(rk->posB2xvec, 1, len*len).get() : (n == 7 ? get_slice_ptr(rk->posB2xvec, 2, len*len).get() : thrust::raw_pointer_cast(sim->d_posB2xOld.data()))))),
                (n == 5 || n == 6 || n == 7),
                pool);
            // interpolateGPU(thrust::raw_pointer_cast(sim->d_posB1xOld.data()),thrust::raw_pointer_cast(sim->d_posB2xOld.data()),true);
        }

        // Update position vectors
        if (n == 2) {
            set_slice(rk->posB1xvec,0,sim->d_posB1xOld);
            set_slice(rk->posB2xvec,0,sim->d_posB2xOld);
        } else if (n == 3) {
            set_slice(rk->posB1xvec,1,sim->d_posB1xOld);
            set_slice(rk->posB2xvec,1,sim->d_posB2xOld);
        } else if (n == 4) {
            set_slice(rk->posB1xvec,2,sim->d_posB1xOld);
            set_slice(rk->posB2xvec,2,sim->d_posB2xOld);
        }

        // Compute k[n]
        // rk->hK = QKstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, T0, Gamma, *pool);
        // rk->hR = QRstepGPU(sim->d_QRv, sim->d_rInt, sim->d_SigmaRA2int, sim->d_QRB2int, sim->d_QRA2int, sim->d_SigmaRB2int, sim->d_t1grid, sim->d_theta, *pool);
        QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, T0, Gamma, 0, *pool);
        rk->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            rk->hK0 = rk->hK;
            rk->hR0 = rk->hR;
            // rk->gK = MAGPU(rk->gK.data(), rk->hK.data(), delta_t * rk->avec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->avec[n], len);
            // rk->gR = MAGPU(rk->gR.data(), rk->hR.data(), delta_t * rk->avec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->avec[n], len);
            //rk->gKfinal = MAGPU(rk->gKfinal.data(), rk->hK.data(), delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->bvec[n], len);
            // rk->gRfinal = MAGPU(rk->gRfinal.data(), rk->hR.data(), delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->bvec[n], len);
            if(rk->b2vec[n] != 0.0) {
                // rk->gKe = MAGPU(rk->gKe.data(), rk->hK.data(), delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->b2vec[n], len);
                // rk->gRe = MAGPU(rk->gRe.data(), rk->hR.data(), delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->b2vec[n], len);
            }
            rk->gt += delta_t * rk->avec[n] * rk->ht;
            dr = drstep2GPU(get_slice_ptr(sim->d_QKv, t1len - 1, len), get_slice_ptr(sim->d_QRv, t1len - 1, len), rk->hK.data(), rk->hR.data(), t_current, T0, *pool);

            appendAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK.data(), rk->hR.data(), rk->ht * dr, rk->gt, len, *pool); // Append Update
        } else {
            if (n != rk->stages - 1) {
                // rk->gK = MAGPU(rk->gK.data(), rk->hK.data(), delta_t * rk->avec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->avec[n], len);
                // rk->gR = MAGPU(rk->gR.data(), rk->hR.data(), delta_t * rk->avec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->avec[n], len);
            }
            // rk->gKfinal = MAGPU(rk->gKfinal.data(), rk->hK.data(), delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->bvec[n], len);
            // rk->gRfinal = MAGPU(rk->gRfinal.data(), rk->hR.data(), delta_t * rk->bvec[n], len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->bvec[n], len);
            if(rk->b2vec[n] != 0.0) {
                // rk->gKe = MAGPU(rk->gKe.data(), rk->hK.data(), delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->b2vec[n], len);
                // rk->gRe = MAGPU(rk->gRe.data(), rk->hR.data(), delta_t * rk->b2vec[n], len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()), delta_t * rk->b2vec[n], len);
            }
            if (n == 4) {
                AddSubtractGPU(rk->gK, rk->gKfinal, rk->gK0, rk->gR, rk->gRfinal, rk->gR0, (*pool)[0]);
                rk->gt = rk->gt0*9/15 + (rk->gt + delta_t * rk->avec[n] * rk->ht)*6/15;
            } else if (n != rk->stages - 1) {
                rk->gt += delta_t * rk->avec[n] * rk->ht;
            }
            if (n != rk->stages - 1) {
                replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(rk->gKfinal.data(), rk->gRfinal.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, len, *pool); // Replace Update
            }
        }
        // double sum = thrust::reduce(sim->d_QKv.begin(), sim->d_QKv.end(), 0.0, thrust::plus<double>());
        // cout << "Sum of sim->d_QKv: " << sum - sim->d_t1grid.size() * len << endl;
        // sum = thrust::reduce(gKvec.begin()+(n+1)*len, gKvec.begin()+(n+2)*len, 0.0, thrust::plus<double>());
        // cout << "Sum of gKvec: " << sum - 0 << endl;

    }

    // Final interpolation
    interpolateGPU(thrust::raw_pointer_cast(sim->d_posB1xOld.data()), thrust::raw_pointer_cast(sim->d_posB2xOld.data()),pool);

    // Compute error estimate
    computeError<<<blocks, threads, threads * sizeof(double), (*pool)[0]>>>(
        thrust::raw_pointer_cast(rk->gKfinal.data()),
        thrust::raw_pointer_cast(rk->gKe.data()),
        thrust::raw_pointer_cast(rk->gRfinal.data()),
        thrust::raw_pointer_cast(rk->gRe.data()),
        thrust::raw_pointer_cast(sim->error_result.data()),
        len
    );

    double error = sim->error_result[0];
    error += abs(rk->gtfinal - rk->gte);

    return error;
}

void init()
{
    gpu = isCompatibleGPUInstalled();
    sim = new SimulationData();
    rk = new RKData();
    import();

    t1grid.resize(1, 0.0);
    delta_t_ratio.resize(1, 0.0);
    specRad = 4 * sqrt(DDflambda(1));

    delta_t = delta_t_min;
    loop = 0;
    delta = 1;
    delta_old = 0;

    posB1xOld.resize(len, 1.0);
    posB2xOld.resize(len * len, 0.0);

    SigmaKA1int.resize(len * len, 0.0);
    SigmaRA1int.resize(len * len, 0.0);
    SigmaKB1int.resize(len * len, 0.0);
    SigmaRB1int.resize(len * len, 0.0);
    SigmaKA2int.resize(len * len, 0.0);
    SigmaRA2int.resize(len * len, 0.0);
    SigmaKB2int.resize(len * len, 0.0);
    SigmaRB2int.resize(len * len, 0.0);

    QKA1int.resize(len * len, 0.0);
    QRA1int.resize(len * len, 0.0);
    QKB1int.resize(len * len, 0.0);
    QRB1int.resize(len * len, 0.0);
    QKA2int.resize(len * len, 0.0);
    QRA2int.resize(len * len, 0.0);
    QKB2int.resize(len * len, 0.0);
    QRB2int.resize(len * len, 0.0);

    QKv.resize(len, 1.0);
    QRv.resize(len, 1.0);
    dQKv.resize(len, 0.0);
    dQRv.resize(len, 0.0);
    rvec.resize(1, Gamma + Dflambda(1) / T0);
    drvec.resize(1, rstep());

    rInt.resize(len, 0.0);
    drInt.resize(len, 0.0);

    if (gpu) {
        copyVectorsToGPU();
        copyParametersToDevice(p, p2, lambda);

        if (!weightsA1y.empty()) {
            size_t depthA1 = weightsA1y.size() / (len * len);
            IndexVecLN3Optimizer::setupKernel(depthA1);
        }
        if (!weightsA2y.empty()) {
            size_t depthA2 = weightsA2y.size() / (len * len);
            IndexVecLN3Optimizer::setupKernel(depthA2);
        }

        if (!weightsB2y.empty()) {
            size_t depthB2 = weightsB2y.size() / (len * len);
            IndexMatAllOptimizer::setupKernel(depthB2);
        }

        init_RK54GPU();
    } else {
        rk->init = 1;
    }  
}

double update() {
    if (rk->init == 1) {
        return RK54();
    } else {
        return SSPRK104();
    }  
}

double updateGPU(StreamPool* pool = nullptr) {
    if (rk->init == 1) {
        return RK54GPU(pool);
    } else {
        return SSPRK104GPU(pool);
    }
}

int main() {

    StreamPool* pool = new StreamPool(20);

    // 0) Initialize
    init();

    std::cout << "Starting simulation..." << std::endl;

    // 1) Open the output file for correlation
    std::ofstream corr("correlation.txt");
    if (!corr) {
        std::cerr << "Error: Unable to open correlation.txt" << std::endl;
        return 1;
    }
    corr << std::fixed << std::setprecision(14);
    cout << std::fixed << std::setprecision(14);

    double t = (gpu ? sim->d_t1grid.back() : t1grid.back());

    // 2) Main loop
    while (t < tmax && loop < maxLoop) {
        t += delta_t;
        delta_old = delta;
        delta = (gpu ? updateGPU(pool) : update());
        // cout << "delta CPU: " << delta << endl;
        // delta = SSPRK104GPU();
        // cout << "delta GPU: " << delta << endl;
        loop++;

        if (loop % 100000 == 0) {
            (gpu ? sparsifyNscaleGPU(delta_max) : sparsifyNscale(delta_max));
            if (delta < delta_max / 2) {
                delta_t *= 0.5;
                if (gpu){
                    init_SSPRK104GPU();
                }  else {
                    rk->init = 2;
                } 
            }
        }

        // primitive time-step adaptation
        if (delta < delta_max && loop > 5 &&
            (delta < 1.1 * delta_old || delta_old == 0) &&
            rmax[rk->init] / specRad > delta_t && (gpu ? sim->d_delta_t_ratio.back() : delta_t_ratio.back()))
        {
            delta_t *= 1.01;
        }
        else if (delta > 2 * delta_max && delta_t > delta_t_min) {
            delta_t *= 0.9;
        }
        if (rk->init == 2 && delta > delta_max && rmax[1] / specRad > delta_t) {
            if (gpu) {
                init_RK54GPU();
            } else {
                rk->init = 1;
            }
            delta_t *= 0.5;
        }

        size_t current_t1len = gpu ? sim->d_t1grid.size() : t1grid.size();
        double qk0 = gpu ? sim->d_QKv[(current_t1len - 1) * len + 0] : 
                      QKv[(current_t1len - 1) * len + 0];

        // display a video
        std::cout << "loop: " << loop
            << " time: " << t
            << " time step: " << delta_t
            << " delta: " << delta
            << " method: " << (rk->init == 1 ? "RK54" : "SSPRK104")
            << " QK: " << qk0
            << " length of t1grid: " << current_t1len
            << std::endl;

        // record QK(t,0) to file
        if( gpu ) {
            double energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, T0); 
            corr << t << "\t" << energy << "\t" << qk0 << "\n";
        } else {
            vector<double> temp(len,0.0);
            SigmaK(getLastLenEntries(QKv, len),temp);
            double energy = -(ConvA(temp,getLastLenEntries(QRv, len),t)[0] + Dflambda(qk0)/T0); 
            corr << t << "\t" << energy << "\t" << qk0 << "\n";
        }
    }

    (gpu ? copyVectorsToCPU() : copyVectorsToGPU());

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        // QKstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, T0, Gamma, *pool);
        // QRstepGPU(sim->d_QRv, sim->d_rInt, sim->d_SigmaRA2int, sim->d_QRB2int, sim->d_QRA2int, sim->d_SigmaRB2int, sim->d_t1grid, sim->d_theta, *pool);
        // QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, T0, Gamma, 0, *pool);
        // rk->gK = MAGPU(rk->gK.data(), rk->hK.data(), delta_t * rk->avec[0], len);
        // computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()), delta_t * rk->avec[0], len);
        // AddSubtractGPU(rk->gK, rk->gKfinal, rk->gK0, rk->gR, rk->gRfinal, rk->gR0, (*pool)[0]);
        // drstep2GPU(get_slice_ptr(sim->d_QKv, sim->d_t1grid.size() - 1, len), get_slice_ptr(sim->d_QRv, sim->d_t1grid.size() - 1, len), rk->hK.data(), rk->hR.data(), sim->d_t1grid.back(), T0, *pool);
        // rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, Gamma, T0, *pool);
        interpolateGPU();  
        // replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht, rk->gt, len, *pool); // Replace Update
        // bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);
    }

    // Synchronize all streams before timing ends

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;
    std::cout << "Average wall time: " << avg_ms << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        updateGPU(pool);
    }

    // Synchronize all streams before timing ends

    end = std::chrono::high_resolution_clock::now();

    total = end - start;
    avg_ms = total.count() / 100;
    std::cout << "Average wall time: " << avg_ms << " ms" << std::endl;

    // 3) Print the final results
    std::cout << "final delta_t: " << delta_t << std::endl;
    std::cout << "final delta:   " << delta << std::endl;
    std::cout << "final loop:    " << loop << std::endl;
    std::cout << "final t1grid:  " << t1grid.back() << std::endl;
    std::cout << "final rvec:    " << rvec.back() << std::endl;
    std::cout << "final drvec:   " << drvec.back() << std::endl;
    std::cout << "final QKv:     " << QKv[(t1grid.size() - 1) * len] - 1 << std::endl;
    std::cout << "final QRv:     " << QRv[(t1grid.size() - 1) * len] - 1 << std::endl;
    std::cout << "Simulation finished." << std::endl;
 
    // 4) Close the file
    corr.close();

    clearAllVectors();

    delete pool;
    return 0;
}