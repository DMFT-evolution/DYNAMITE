//Compile with nvcc -ccbin clang++ --extended-lambda --use_fast_math -Xcompiler "-O3 -march=native -ffast-math" -o minimalLoop minimalLoop.cu

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
        return result;
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
        return result;
    }

    // Convert back to raw thrust vector
    const thrust::device_vector<double>& vec() const { return data; }
    thrust::device_vector<double>& vec() { return data; }
};


constexpr int p = 2;
constexpr int p2 = 10;
constexpr double lambda = 0.6;
constexpr double TMCT = 0.805166;
constexpr double T0 = 1.001*TMCT;
// double T0=1e50;
constexpr double Gamma = 0.0;
constexpr int maxLoop = 10000;

constexpr double tmax = 1e7; //time to evolve to
constexpr double delta_t_min = 1e-2; //initial and minimal time step
constexpr double delta_max = 1e-8; //maximal error per step
constexpr double rmax = 13; // stability range of SSPRK(10,4)

double delta;
double delta_old;
int loop;
double specRad;
double delta_t;
size_t len = 512;
int ord;
bool gpu = true; // Use GPU if true, CPU if false

vector<double> theta, phi1, phi2, posA1y, posA2y, posB2y, weightsA1y, weightsA2y, weightsB2y, posB1xOld, posB2xOld, integ;
vector<size_t> indsA1y, indsA2y, indsB2y;

vector<double> t1grid, delta_t_ratio;

vector<double> QKv, QRv, dQKv, dQRv, rInt, drInt, rvec, drvec;

vector<double> SigmaKA1int, SigmaRA1int, SigmaKB1int, SigmaRB1int, SigmaKA2int, SigmaRA2int, SigmaKB2int, SigmaRB2int;
vector<double> QKA1int, QRA1int, QKB1int, QRB1int, QKA2int, QRA2int, QKB2int, QRB2int;

// Device pointers for GPU memory
// double *d_theta, *d_phi1, *d_phi2, *d_posA1y, *d_posA2y, *d_posB2y, *d_weightsA1y, *d_weightsA2y, *d_weightsB2y, *d_posB1xOld, *d_posB2xOld, *d_integ;
// size_t *d_indsA1y, *d_indsA2y, *d_indsB2y;

// double *d_t1grid, *d_delta_t_ratio;

// double *d_QKv, *d_QRv, *d_dQKv, *d_dQRv, *d_rInt, *d_drInt, *d_rvec, *d_drvec;

// double *d_SigmaKA1int, *d_SigmaRA1int, *d_SigmaKB1int, *d_SigmaRB1int, *d_SigmaKA2int, *d_SigmaRA2int, *d_SigmaKB2int, *d_SigmaRB2int;
// double *d_QKA1int, *d_QRA1int, *d_QKB1int, *d_QRB1int, *d_QKA2int, *d_QRA2int, *d_QKB2int, *d_QRB2int;

thrust::device_vector<double> d_theta, d_phi1, d_phi2, d_posA1y, d_posA2y, d_posB2y, d_weightsA1y, d_weightsA2y, d_weightsB2y, d_posB1xOld, d_posB2xOld, d_integ;
thrust::device_vector<size_t> d_indsA1y, d_indsA2y, d_indsB2y;

thrust::device_vector<double> d_t1grid, d_delta_t_ratio;

thrust::device_vector<double> d_QKv, d_QRv, d_dQKv, d_dQRv, d_rInt, d_drInt, d_rvec, d_drvec;

thrust::device_vector<double> d_SigmaKA1int, d_SigmaRA1int, d_SigmaKB1int, d_SigmaRB1int, d_SigmaKA2int, d_SigmaRA2int, d_SigmaKB2int, d_SigmaRB2int;
thrust::device_vector<double> d_QKA1int, d_QRA1int, d_QKB1int, d_QRB1int, d_QKA2int, d_QRA2int, d_QKB2int, d_QRB2int;

vector<double> temp_gpu;
vector<double> temp_cpu(weightsB2y.size(), 0.0);

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

    return result;
}

inline vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for addition.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::minus<>());

    return result;
}

// Overload the * operator for the element-wise product of two vectors
inline vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for element-wise multiplication.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise multiplication
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::multiplies<>());

    return result;
}

// Overload the * operator for the product of a vector and a scalar
inline vector<double> operator*(const vector<double>& vec, double scalar) {
    vector<double> result(vec.size());

    // Use std::transform to multiply each element by the scalar
    std::transform(vec.begin(), vec.end(), result.begin(), [scalar](double val) { return val * scalar; });

    return result;
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
    return result;
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
    return result;
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
    
    return result;
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
    return result;
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
    return result;
}

// Example: Multiply a thrust::device_vector by a scalar
thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& vec, double scalar, size_t len) {
    thrust::device_vector<double> result(len);
    thrust::transform(
        vec, vec + len,
        result.begin(),
        [scalar] __device__(double x) { return x * scalar; }
    );
    return result;
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
    double alpha,
    double beta,
    size_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double result = alpha * a[i] + beta * b[i];

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
    double alpha,
    double beta,
    const thrust::device_vector<double>* delta = nullptr,
    const thrust::device_vector<double>* extra1 = nullptr,
    const thrust::device_vector<double>* extra2 = nullptr,
    const thrust::device_vector<double>* extra3 = nullptr,
    const thrust::device_ptr<double>& subtract = nullptr
) {
    size_t N = out.size();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    FusedUpdateKernel<<<blocks, threads>>>(
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
    cudaDeviceSynchronize();
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

void QRstepFused(const thrust::device_vector<double>& qR,
                 const thrust::device_vector<double>& theta,
                 const thrust::device_vector<double>& conv1,
                 const thrust::device_vector<double>& conv2,
                 const thrust::device_vector<double>& r,
                 thrust::device_vector<double>& out) {
    size_t len = theta.size();
    int threads = 256;
    int blocks = (len + threads - 1) / threads;

    FusedQRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qR.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(conv1.data()),
        thrust::raw_pointer_cast(conv2.data()),
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(out.data()), len
    );
    cudaDeviceSynchronize();
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

    d_theta = theta; // Copying to thrust::device_vector
    d_phi1 = phi1;   
    d_phi2 = phi2;   
    d_posA1y = posA1y; 
    d_posA2y = posA2y; 
    d_posB2y = posB2y; 
    d_indsA1y = indsA1y; 
    d_indsA2y = indsA2y; 
    d_indsB2y = indsB2y; 
    d_weightsA1y = weightsA1y; 
    d_weightsA2y = weightsA2y; 
    d_weightsB2y = weightsB2y; 
    d_integ = integ; 

    d_posB1xOld = posB1xOld; 
    d_posB2xOld = posB2xOld; 

    d_SigmaKA1int = SigmaKA1int; 
    d_SigmaRA1int = SigmaRA1int; 
    d_SigmaKB1int = SigmaKB1int; 
    d_SigmaRB1int = SigmaRB1int; 
    d_SigmaKA2int = SigmaKA2int; 
    d_SigmaRA2int = SigmaRA2int; 
    d_SigmaKB2int = SigmaKB2int; 
    d_SigmaRB2int = SigmaRB2int; 

    d_QKA1int = QKA1int; 
    d_QRA1int = QRA1int; 
    d_QKB1int = QKB1int; 
    d_QRB1int = QRB1int; 
    d_QKA2int = QKA2int; 
    d_QRA2int = QRA2int; 
    d_QKB2int = QKB2int; 
    d_QRB2int = QRB2int; 

    d_QKv = QKv; 
    d_QRv = QRv; 
    d_dQKv = dQKv; 
    d_dQRv = dQRv; 
    d_rvec = rvec; 
    d_drvec = drvec; 

    d_rInt = rInt; 
    d_drInt = drInt; 
    d_t1grid = t1grid; 
    d_delta_t_ratio = delta_t_ratio; 

    std::cout << "All vectors copied to GPU memory." << std::endl;
}

void copyVectorsToCPU() {
    QKv.resize(d_QKv.size());
    thrust::copy(d_QKv.begin(), d_QKv.end(), QKv.begin());
    QRv.resize(d_QRv.size());
    thrust::copy(d_QRv.begin(), d_QRv.end(), QRv.begin());
    dQKv.resize(d_dQKv.size());
    thrust::copy(d_dQKv.begin(), d_dQKv.end(), dQKv.begin());
    dQRv.resize(d_dQRv.size());
    thrust::copy(d_dQRv.begin(), d_dQRv.end(), dQRv.begin());
    rvec.resize(d_rvec.size());
    thrust::copy(d_rvec.begin(), d_rvec.end(), rvec.begin());
    drvec.resize(d_drvec.size());
    thrust::copy(d_drvec.begin(), d_drvec.end(), drvec.begin());
    rInt.resize(d_rInt.size());
    thrust::copy(d_rInt.begin(), d_rInt.end(), rInt.begin());
    drInt.resize(d_drInt.size());
    thrust::copy(d_drInt.begin(), d_drInt.end(), drInt.begin());
    t1grid.resize(d_t1grid.size());
    thrust::copy(d_t1grid.begin(), d_t1grid.end(), t1grid.begin());
    delta_t_ratio.resize(d_delta_t_ratio.size());
    thrust::copy(d_delta_t_ratio.begin(), d_delta_t_ratio.end(), delta_t_ratio.begin());

    // Stuff below this line is not necessary and can be remove after testing.
    SigmaKA1int.resize(d_SigmaKA1int.size());
    thrust::copy(d_SigmaKA1int.begin(), d_SigmaKA1int.end(), SigmaKA1int.begin());
    SigmaRA1int.resize(d_SigmaRA1int.size());
    thrust::copy(d_SigmaRA1int.begin(), d_SigmaRA1int.end(), SigmaRA1int.begin());
    SigmaKB1int.resize(d_SigmaKB1int.size());
    thrust::copy(d_SigmaKB1int.begin(), d_SigmaKB1int.end(), SigmaKB1int.begin());
    SigmaRB1int.resize(d_SigmaRB1int.size());
    thrust::copy(d_SigmaRB1int.begin(), d_SigmaRB1int.end(), SigmaRB1int.begin());
    SigmaKA2int.resize(d_SigmaKA2int.size());
    thrust::copy(d_SigmaKA2int.begin(), d_SigmaKA2int.end(), SigmaKA2int.begin());
    SigmaRA2int.resize(d_SigmaRA2int.size());
    thrust::copy(d_SigmaRA2int.begin(), d_SigmaRA2int.end(), SigmaRA2int.begin());
    SigmaKB2int.resize(d_SigmaKB2int.size());
    thrust::copy(d_SigmaKB2int.begin(), d_SigmaKB2int.end(), SigmaKB2int.begin());
    SigmaRB2int.resize(d_SigmaRB2int.size());
    thrust::copy(d_SigmaRB2int.begin(), d_SigmaRB2int.end(), SigmaRB2int.begin());

    QKA1int.resize(d_QKA1int.size());
    thrust::copy(d_QKA1int.begin(), d_QKA1int.end(), QKA1int.begin());
    QRA1int.resize(d_QRA1int.size());
    thrust::copy(d_QRA1int.begin(), d_QRA1int.end(), QRA1int.begin());
    QKB1int.resize(d_QKB1int.size());
    thrust::copy(d_QKB1int.begin(), d_QKB1int.end(), QKB1int.begin());
    QRB1int.resize(d_QRB1int.size());
    thrust::copy(d_QRB1int.begin(), d_QRB1int.end(), QRB1int.begin());
    QKA2int.resize(d_QKA2int.size());
    thrust::copy(d_QKA2int.begin(), d_QKA2int.end(), QKA2int.begin());
    QRA2int.resize(d_QRA2int.size());
    thrust::copy(d_QRA2int.begin(), d_QRA2int.end(), QRA2int.begin());
    QKB2int.resize(d_QKB2int.size());
    thrust::copy(d_QKB2int.begin(), d_QKB2int.end(), QKB2int.begin());
    QRB2int.resize(d_QRB2int.size());
    thrust::copy(d_QRB2int.begin(), d_QRB2int.end(), QRB2int.begin());

    posA1y.resize(d_posA1y.size());
    thrust::copy(d_posA1y.begin(), d_posA1y.end(), posA1y.begin());
    posA2y.resize(d_posA2y.size());
    thrust::copy(d_posA2y.begin(), d_posA2y.end(), posA2y.begin());
    posB2y.resize(d_posB2y.size());
    thrust::copy(d_posB2y.begin(), d_posB2y.end(), posB2y.begin());
    posB1xOld.resize(d_posB1xOld.size());
    thrust::copy(d_posB1xOld.begin(), d_posB1xOld.end(), posB1xOld.begin());
    posB2xOld.resize(d_posB2xOld.size());
    thrust::copy(d_posB2xOld.begin(), d_posB2xOld.end(), posB2xOld.begin());
}

void clearAllVectors() {
    d_QKv.clear();
    d_QKv.shrink_to_fit();
    d_QRv.clear();
    d_QRv.shrink_to_fit();
    d_dQKv.clear();
    d_dQKv.shrink_to_fit();
    d_dQRv.clear();
    d_dQRv.shrink_to_fit();

    d_rInt.clear();
    d_rInt.shrink_to_fit();
    d_drInt.clear();
    d_drInt.shrink_to_fit();
    d_rvec.clear();
    d_rvec.shrink_to_fit();
    d_drvec.clear();
    d_drvec.shrink_to_fit();

    d_SigmaKA1int.clear();
    d_SigmaKA1int.shrink_to_fit();
    d_SigmaRA1int.clear();
    d_SigmaRA1int.shrink_to_fit();
    d_SigmaKB1int.clear();
    d_SigmaKB1int.shrink_to_fit();
    d_SigmaRB1int.clear();
    d_SigmaRB1int.shrink_to_fit();
    d_SigmaKA2int.clear();
    d_SigmaKA2int.shrink_to_fit();
    d_SigmaRA2int.clear();
    d_SigmaRA2int.shrink_to_fit();
    d_SigmaKB2int.clear();
    d_SigmaKB2int.shrink_to_fit();
    d_SigmaRB2int.clear();
    d_SigmaRB2int.shrink_to_fit();

    d_QKA1int.clear();
    d_QKA1int.shrink_to_fit();
    d_QRA1int.clear();
    d_QRA1int.shrink_to_fit();
    d_QKB1int.clear();
    d_QKB1int.shrink_to_fit();
    d_QRB1int.clear();
    d_QRB1int.shrink_to_fit();
    d_QKA2int.clear();
    d_QKA2int.shrink_to_fit();
    d_QRA2int.clear();
    d_QRA2int.shrink_to_fit();
    d_QKB2int.clear();
    d_QKB2int.shrink_to_fit();
    d_QRB2int.clear();
    d_QRB2int.shrink_to_fit();

    d_theta.clear();
    d_theta.shrink_to_fit();
    d_phi1.clear();
    d_phi1.shrink_to_fit();
    d_phi2.clear();
    d_phi2.shrink_to_fit();

    d_posA1y.clear();
    d_posA1y.shrink_to_fit();
    d_posA2y.clear();
    d_posA2y.shrink_to_fit();
    d_posB2y.clear();
    d_posB2y.shrink_to_fit();

    d_weightsA1y.clear();
    d_weightsA1y.shrink_to_fit();
    d_weightsA2y.clear();
    d_weightsA2y.shrink_to_fit();
    d_weightsB2y.clear();
    d_weightsB2y.shrink_to_fit();

    d_posB1xOld.clear();
    d_posB1xOld.shrink_to_fit();
    d_posB2xOld.clear();
    d_posB2xOld.shrink_to_fit();

    d_indsA1y.clear();
    d_indsA1y.shrink_to_fit();
    d_indsA2y.clear();
    d_indsA2y.shrink_to_fit();
    d_indsB2y.clear();
    d_indsB2y.shrink_to_fit();

    d_integ.clear();
    d_integ.shrink_to_fit();
    d_t1grid.clear();
    d_t1grid.shrink_to_fit();
    d_delta_t_ratio.clear();
    d_delta_t_ratio.shrink_to_fit();
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

__device__ double DDflambdaGPU(const double q) 
{
    return lambda * p * (p - 1) * pow_const_device<p - 2>(q) + (1 - lambda) * p2 * (p2 - 1) * pow_const_device<p2 - 2>(q);
}

__device__ double DDDflambdaGPU(const double q) 
{
    return lambda * p * (p - 1) * (p - 2) * pow_const_device<p - 3>(q) + (1 - lambda) * p2 * (p2 - 1) * (p2 - 2) * pow_const_device<p2 - 3>(q);
}

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
__global__ void indexVecLN3_kernel(const double* __restrict__ weights,
                                   const size_t* __restrict__ inds,
                                   const double* __restrict__ QK,
                                   const double* __restrict__ QR,
                                   double* __restrict__ qk_out,
                                   double* __restrict__ qr_out,
                                   size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    const double* w = weights + j * DEPTH;
    size_t idx = inds[j];

    double sum_k = 0.0, sum_r = 0.0;

    #pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        double weight = w[d];
        sum_k += weight * __ldg(QK + idx + d);
        sum_r += weight * __ldg(QR + idx + d);
    }

    qk_out[j] = sum_k;
    qr_out[j] = sum_r;
}

template <int DEPTH>
void indexVecLN3GPU_fast(const thrust::device_vector<double>& weights,
                         const thrust::device_vector<size_t>& inds,
                         thrust::device_vector<double>& qk_result,
                         thrust::device_vector<double>& qr_result)
{
    size_t prod = inds.size();
    size_t length = d_QKv.size() - len;

    const double* QK_start = thrust::raw_pointer_cast(d_QKv.data()) + length;
    const double* QR_start = thrust::raw_pointer_cast(d_QRv.data()) + length;
    const double* weights_ptr = thrust::raw_pointer_cast(weights.data());
    const size_t* inds_ptr = thrust::raw_pointer_cast(inds.data());
    double* qk_out = thrust::raw_pointer_cast(qk_result.data());
    double* qr_out = thrust::raw_pointer_cast(qr_result.data());

    int threads = 256;
    int blocks = (prod + threads - 1) / threads;

    indexVecLN3_kernel<DEPTH><<<blocks, threads>>>(
        weights_ptr, inds_ptr, QK_start, QR_start, qk_out, qr_out, prod
    );

    cudaDeviceSynchronize(); // Optional if used synchronously
}

void indexVecLN3GPU(
    const thrust::device_vector<double>& weights,
    const thrust::device_vector<size_t>& inds,
    thrust::device_vector<double>& qk_result,
    thrust::device_vector<double>& qr_result
) {
    size_t prod = inds.size();
    size_t length = d_QKv.size() - len;
    size_t depth = weights.size() / prod;

    const double* __restrict__ QK_start = thrust::raw_pointer_cast(d_QKv.data()) + length;
    const double* __restrict__ QR_start = thrust::raw_pointer_cast(d_QRv.data()) + length;
    const double* __restrict__ weights_start = thrust::raw_pointer_cast(weights.data());
    const size_t* __restrict__ inds_start = thrust::raw_pointer_cast(inds.data());
    double* __restrict__ qk_result_start = thrust::raw_pointer_cast(qk_result.data());
    double* __restrict__ qr_result_start = thrust::raw_pointer_cast(qr_result.data());

    auto kernel = [=] __device__ (size_t j) {
        size_t index = inds_start[j];
        const double* w = weights_start + j * depth;

        double qk = 0.0, qr = 0.0;

        // Stack-allocated weights for better register usage
        double w_local[32];  // works for depth ≤ 32

        // Preload weights into local registers
        #pragma unroll 32
        for (int d = 0; d < 32; ++d) {
            w_local[d] = (d < depth) ? w[d] : 0.0;
        }

        // Now accumulate QK/QR
        #pragma unroll 32
        for (int d = 0; d < 32; ++d) {
            if (d < depth) {
                double wk = QK_start[index + d];
                double wr = QR_start[index + d];
                double wj = w_local[d];
                qk += wj * wk;
                qr += wj * wr;
            }
        }

        qk_result_start[j] = qk;
        qr_result_start[j] = qr;
    };

    thrust::for_each_n(thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        prod,
        kernel
    );
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
                  size_t len) {
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
        thrust::device,
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

void indexVecR2GPU(const thrust::device_vector<double>& in1, 
                   const thrust::device_vector<double>& in2, 
                   const thrust::device_vector<double>& in3, 
                   const thrust::device_vector<size_t>& inds, 
                   const thrust::device_vector<double>& dtratio, 
                   thrust::device_vector<double>& result) {
    size_t dims = inds.size();
    size_t t1len = dtratio.size();

    const double* in1_ptr = thrust::raw_pointer_cast(in1.data());
    const double* in2_ptr = thrust::raw_pointer_cast(in2.data());
    const double* in3_ptr = thrust::raw_pointer_cast(in3.data());
    const size_t* inds_ptr = thrust::raw_pointer_cast(inds.data());
    const double* dtratio_ptr = thrust::raw_pointer_cast(dtratio.data());
    double* result_ptr = thrust::raw_pointer_cast(result.data());

    thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        dims,
        [=] __device__(size_t i) {
            double in3_squared = in3_ptr[i] * in3_ptr[i];
            double in3_cubed = in3_squared * in3_ptr[i];

            if (inds_ptr[i] < t1len - 1) {
                result_ptr[i] = (1 - 3 * in3_squared - 2 * in3_cubed) * in1_ptr[inds_ptr[i] - 1] +
                                (3 * in3_squared + 2 * in3_cubed) * in1_ptr[inds_ptr[i]] -
                                (in3_ptr[i] + 2 * in3_squared + in3_cubed) * in2_ptr[inds_ptr[i]] -
                                (in3_squared + in3_cubed) * in2_ptr[inds_ptr[i] + 1] / dtratio_ptr[inds_ptr[i] + 1];
            } else {
                result_ptr[i] = (1 - in3_squared) * in1_ptr[inds_ptr[i] - 1] +
                                in3_squared * in1_ptr[inds_ptr[i]] -
                                (in3_ptr[i] + in3_squared) * in2_ptr[inds_ptr[i]];
            }
        });
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

__global__ void indexMatAllKernel(const double* __restrict__ posx,
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

    int threads = 256;
    int blocks = (prod + threads - 1) / threads;

    indexMatAllKernel<<<blocks, threads>>>(
        posx_ptr, indsy_ptr, weightsy_ptr, dtratio_ptr,
        qK_result_ptr, qR_result_ptr,
        QKv_ptr, QRv_ptr, dQKv_ptr, dQRv_ptr,
        len, depth, t1len, prod
    );

    cudaDeviceSynchronize();
}

void SigmaK(const vector<double>& qk, vector<double>& result)
{
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}

void SigmaKGPU(const thrust::device_vector<double>& qk, thrust::device_vector<double>& result) {
    assert(qk.size() == result.size());

    thrust::transform(
        qk.begin(), qk.end(),
        result.begin(),
        [] __device__(double qk_val) {
            return DflambdaGPU(qk_val);
        }
    );
}

// __global__ void SigmaKGPU_kernel(const double* __restrict__ qk,
//                                  double* __restrict__ result,
//                                  size_t N)
// {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         double val = qk[i];
//         result[i] = DflambdaGPU(val);  // must be __device__ __forceinline__
//     }
// }

// void SigmaKGPU_fast(const thrust::device_vector<double>& qk,
//                     thrust::device_vector<double>& result) {
//     size_t N = qk.size();
//     assert(result.size() == N);

//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;

//     SigmaKGPU_kernel<<<blocks, threads>>>(
//         thrust::raw_pointer_cast(qk.data()),
//         thrust::raw_pointer_cast(result.data()),
//         N
//     );

//     // Optional: remove if used in async stream
//     cudaDeviceSynchronize();
// }

void SigmaR(const vector<double>& qk, const vector<double>& qr, vector<double>& result)
{
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]) * qr[i];
    }
}

void SigmaRGPU(const thrust::device_vector<double>& qk, 
               const thrust::device_vector<double>& qr, 
               thrust::device_vector<double>& result) {
    assert(qk.size() == qr.size() && qk.size() == result.size());

    thrust::transform(
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
    return result;
}

vector<double> SigmaR10(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDDflambda(qk[i]) * qr[i];
    }
    return result;
}

vector<double> SigmaK01(const vector<double>& qk)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = 0.0;
    }
    return result;
}

vector<double> SigmaR01(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return result;
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
    return out;
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

    return out;
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
                                       const thrust::device_vector<double>& theta) {
    size_t length = integ.size();
    size_t depth = f.size() / length;

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = (theta.size() == depth) ? thrust::raw_pointer_cast(theta.data()) : nullptr;
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 128;
    size_t shmem = threads * sizeof(double);

    ConvAGPUKernel<<<depth, threads, shmem>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    cudaDeviceSynchronize();

    return out;
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
    return out;
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

    return out;
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

thrust::device_vector<double> ConvRGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta) {
    size_t length = integ.size();              // rows
    size_t depth = f.size() / length;          // output entries

    thrust::device_vector<double> out(depth, 0.0);

    const double* f_ptr = thrust::raw_pointer_cast(f.data());
    const double* g_ptr = thrust::raw_pointer_cast(g.data());
    const double* integ_ptr = thrust::raw_pointer_cast(integ.data());
    const double* theta_ptr = thrust::raw_pointer_cast(theta.data());
    double* out_ptr = thrust::raw_pointer_cast(out.data());

    int threads = 128;
    size_t shmem = length * sizeof(double) + threads * sizeof(double);

    ConvRKernel<<<depth, threads, shmem>>>(
        f_ptr, g_ptr, integ_ptr, theta_ptr, out_ptr, t, length, depth
    );
    cudaDeviceSynchronize();

    return out;
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
    const thrust::device_vector<double>& d_QKv,
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_QKB1int,
    const thrust::device_vector<double>& d_QKB2int,
    const thrust::device_vector<double>& d_QKA1int,
    const thrust::device_vector<double>& d_QRA1int,
    const thrust::device_vector<double>& d_QRA2int,
    const thrust::device_vector<double>& d_QRB1int,
    const thrust::device_vector<double>& d_SigmaRA1int,
    const thrust::device_vector<double>& d_SigmaRA2int,
    const thrust::device_vector<double>& d_SigmaKB1int,
    const thrust::device_vector<double>& d_SigmaKB2int,
    const thrust::device_vector<double>& d_SigmaKA1int,
    const thrust::device_vector<double>& d_SigmaRB1int,
    const thrust::device_vector<double>& d_theta,
    const thrust::device_vector<double>& d_t1grid,
    const thrust::device_vector<double>& d_rInt,
    double T0,
    double Gamma) {
    
    size_t len = d_theta.size();
    thrust::device_ptr<double> d_qK = get_slice_ptr(d_QKv,d_t1grid.size()-1,len);
    thrust::device_ptr<double> d_qR = get_slice_ptr(d_QRv,d_t1grid.size()-1,len);
    thrust::device_vector<double> d_temp(len);
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + len;

    double scale = Dflambda(d_QKv[d_QKv.size() - len]) / T0;

    thrust::transform(
        idx_first, idx_last,
        d_temp.begin(),
        [len, ptr = d_QKB1int.data()] __device__ (size_t i) {
            return ptr[i * len];
        }
    );

    // Step 1: Run reductions
    auto convR  = ConvRGPU(d_SigmaRA2int, d_QKB2int, d_t1grid.back(), d_integ, d_theta);
    auto convA1 = ConvAGPU(d_SigmaRA1int, d_QKB1int, d_t1grid.back(), d_integ, d_theta);
    auto convA2 = ConvAGPU(d_SigmaKA1int, d_QRB1int, d_t1grid.back(), d_integ, d_theta);

    thrust::device_vector<double> d_d1qK(len);
    thrust::device_vector<double> d_d2qK(len);

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(d_temp.data()), d_qK, d_d1qK, scale, -d_rInt.back(), nullptr, &convR, &convA1, &convA2, nullptr);

    // // Compute d1qK
    // thrust::device_vector<double> d_d1qK = SumGPU(
    //     scalarMultiply(d_temp, scale),
    //     SumGPU(
    //         scalarMultiply(d_qK, -d_rInt.back()),
    //         SumGPU(
    //             ConvRGPU(d_SigmaRA2int, d_QKB2int, d_t1grid.back(), d_integ, d_theta),
    //             SumGPU(
    //                 ConvAGPU(d_SigmaRA1int, d_QKB1int, d_t1grid.back(), d_integ, d_theta),
    //                 ConvAGPU(d_SigmaKA1int, d_QRB1int, d_t1grid.back(), d_integ, d_theta)
    //             )
    //         )
    //     ) 
    // );

    thrust::transform(
        idx_first, idx_last,
        d_temp.begin(),
        [len, ptr = d_QKB1int.data()] __device__ (size_t i) {
            return DflambdaGPU(ptr[i * len]);
        }
    );

    convR  = ConvRGPU(d_QRA2int, d_SigmaKB2int, d_t1grid.back(), d_integ, d_theta);
    convA1 = ConvAGPU(d_QRA1int, d_SigmaKB1int, d_t1grid.back(), d_integ, d_theta);
    convA2 = ConvAGPU(d_QKA1int, d_SigmaRB1int, d_t1grid.back(), d_integ, d_theta);

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(d_temp.data()), d_qR, d_d2qK, d_QKv[d_QKv.size() - len] / T0, 2.0 * Gamma, &d_rInt, &convR, &convA1, &convA2, d_qK);

    // // Compute d2qK
    // thrust::device_vector<double> d_d2qK = SumGPU(scalarMultiply(d_temp, d_QKv[d_QKv.size() - len] / T0),
    //     SumGPU(scalarMultiply(d_qR, 2 * Gamma),
    //         SubtractGPU(
    //             SumGPU(
    //                 ConvRGPU(d_QRA2int, d_SigmaKB2int, d_t1grid.back(), d_integ, d_theta),
    //                 SumGPU(
    //                     ConvAGPU(d_QRA1int, d_SigmaKB1int, d_t1grid.back(), d_integ, d_theta),
    //                     ConvAGPU(d_QKA1int, d_SigmaRB1int, d_t1grid.back(), d_integ, d_theta)
    //                 )
    //             ),
    //             ProductGPU(d_qK, d_rInt)
    //         )
    //     )
    // );

    // Combine d1qK and d2qK
    thrust::device_vector<double> d_result = SumGPU(d_d1qK, ProductGPU(d_d2qK,d_theta));

    return d_result;
}

vector<double> QRstep()
{
    vector<double> qR = getLastLenEntries(QRv, len);
    vector<double> d1qR = (qR * (-rInt.back())) + ConvR(SigmaRA2int, QRB2int, t1grid.back());
    vector<double> d2qR = (qR * rInt) - ConvR(QRA2int, SigmaRB2int, t1grid.back());
    return d1qR + (d2qR * theta);
}

thrust::device_vector<double> QRstepGPU(
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_rInt,
    const thrust::device_vector<double>& d_SigmaRA2int,
    const thrust::device_vector<double>& d_QRB2int,
    const thrust::device_vector<double>& d_QRA2int,
    const thrust::device_vector<double>& d_SigmaRB2int,
    const thrust::device_vector<double>& d_t1grid) {
    
    size_t len = d_theta.size();
    thrust::device_vector<double> d_qR = getLastLenEntriesGPU(d_QRv,len);

    // // Compute d1qR
    // thrust::device_vector<double> d_d1qR = SumGPU(
    //     scalarMultiply(d_qR, -d_rInt.back()),
    //     ConvRGPU(d_SigmaRA2int, d_QRB2int, d_t1grid.back(), d_integ, d_theta)
    // );

    // // Compute d2qR
    // thrust::device_vector<double> d_d2qR = SubtractGPU(
    //     ProductGPU(d_qR, d_rInt),
    //     ConvRGPU(d_QRA2int, d_SigmaRB2int, d_t1grid.back(), d_integ, d_theta)
    // );

    // thrust::device_vector<double> d_result = SumGPU(d_d1qR, ProductGPU(d_d2qR,d_theta));

    auto conv1 = ConvRGPU(d_SigmaRA2int, d_QRB2int, d_t1grid.back(), d_integ, d_theta);
    auto conv2 = ConvRGPU(d_QRA2int, d_SigmaRB2int, d_t1grid.back(), d_integ, d_theta);

    thrust::device_vector<double> d_result(len);
    QRstepFused(d_qR, d_theta, conv1, conv2, d_rInt, d_result);


    return d_result;
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
    const thrust::device_vector<double>& d_QKv,
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_t1grid,
    const thrust::device_vector<double>& d_integ,
    const thrust::device_vector<double>& d_theta,
    double Gamma,
    double T0) {
    
    size_t len = d_theta.size();
    thrust::device_vector<double> d_sigmaK(len, 0.0);
    thrust::device_vector<double> d_sigmaR(len, 0.0);

    thrust::device_vector<double> d_qK = getLastLenEntriesGPU(d_QKv, len);
    thrust::device_vector<double> d_qR = getLastLenEntriesGPU(d_QRv, len);

    // Compute sigmaK
    thrust::transform(
        d_qK.begin(), d_qK.end(),
        d_sigmaK.begin(),
        [] __device__ (double qk) { return DflambdaGPU(qk); }
    );

    // Compute sigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_qK.begin(), d_qR.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_qK.end(),   d_qR.end())),
        d_sigmaR.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaR_qK = ConvAGPU(d_sigmaR, d_qK, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(d_sigmaK, d_qR, d_t1grid.back(), d_integ, d_theta);

    // Compute final result
    double result = Gamma +
                    convA_sigmaR_qK.front() +
                    convA_sigmaK_qR.front() +
                    d_sigmaK.front() * d_qK.front() / T0;

    

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
    const thrust::device_vector<double>& d_QKv,
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_dQKv,
    const thrust::device_vector<double>& d_dQRv,
    const thrust::device_vector<double>& d_t1grid,
    const thrust::device_vector<double>& d_integ,
    const thrust::device_vector<double>& d_theta,
    double T0) {
    
    size_t len = d_QKv.size();
    thrust::device_vector<double> d_sigmaK(len, 0.0);
    thrust::device_vector<double> d_sigmaR(len, 0.0);
    thrust::device_vector<double> d_dsigmaK(len, 0.0);
    thrust::device_vector<double> d_dsigmaR(len, 0.0);

    // Compute sigmaK
    thrust::transform(
        d_QKv.begin(), d_QKv.end(),
        d_sigmaK.begin(),
        [] __device__(double qk) { return DflambdaGPU(qk); }
    );

    // Compute sigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_QRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_QRv.end())),
        d_sigmaR.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );

    // Compute dsigmaK
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_dQKv.begin(), d_dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_dQKv.end(), d_dQRv.end())),
        d_dsigmaK.begin(),
        [] __device__(thrust::tuple<double, double, double> qk_dqk_dqr) {
            double qk = thrust::get<0>(qk_dqk_dqr);
            double dqk = thrust::get<1>(qk_dqk_dqr);
            double dqr = thrust::get<2>(qk_dqk_dqr);
            return DDflambdaGPU(qk) * dqk + DflambdaGPU(qk) * dqr;
        }
    );

    // Compute dsigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_QRv.begin(), d_dQKv.begin(), d_dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_QRv.end(), d_dQKv.end(), d_dQRv.end())),
        d_dsigmaR.begin(),
        [] __device__(thrust::tuple<double, double, double, double> qk_qr_dqk_dqr) {
            double qk = thrust::get<0>(qk_qr_dqk_dqr);
            double qr = thrust::get<1>(qk_qr_dqk_dqr);
            double dqk = thrust::get<2>(qk_qr_dqk_dqr);
            double dqr = thrust::get<3>(qk_qr_dqk_dqr);
            return DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaR_qK = ConvAGPU(d_sigmaR, d_QKv, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(d_sigmaK, d_QRv, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_dsigmaR_qK = ConvAGPU(d_dsigmaR, d_QKv, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_dsigmaK_qR = ConvAGPU(d_dsigmaK, d_QRv, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaR_dqK = ConvAGPU(d_sigmaR, d_dQKv, d_t1grid.back(), d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaK_dqR = ConvAGPU(d_sigmaK, d_dQRv, d_t1grid.back(), d_integ, d_theta);

    // Compute final result
    double result = convA_sigmaR_qK.front() +
                    convA_sigmaK_qR.front() +
                    convA_dsigmaR_qK.front() +
                    convA_dsigmaK_qR.front() +
                    convA_sigmaR_dqK.front() +
                    convA_sigmaK_dqR.front() +
                    (d_dsigmaK.front() * d_QKv.front() + d_sigmaK.front() * d_dQKv.front()) / T0;

    return result;
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
    const thrust::device_vector<double>& d_QKv,
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_dQKv,
    const thrust::device_vector<double>& d_dQRv,
    const double t,
    double T0) {
    
    size_t len = d_QKv.size();
    thrust::device_vector<double> d_sigmaK(len, 0.0);
    thrust::device_vector<double> d_sigmaR(len, 0.0);
    thrust::device_vector<double> d_dsigmaK(len, 0.0);
    thrust::device_vector<double> d_dsigmaR(len, 0.0);

    // Compute sigmaK
    thrust::transform(
        d_QKv.begin(), d_QKv.end(),
        d_sigmaK.begin(),
        [] __device__(double qk) { return DflambdaGPU(qk); }
    );

    // Compute sigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_QRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_QRv.end())),
        d_sigmaR.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );

    // Compute dsigmaK
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_dQKv.begin(), d_dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_dQKv.end(), d_dQRv.end())),
        d_dsigmaK.begin(),
        [] __device__(thrust::tuple<double, double, double> qk_dqk_dqr) {
            double qk = thrust::get<0>(qk_dqk_dqr);
            double dqk = thrust::get<1>(qk_dqk_dqr);
            double dqr = thrust::get<2>(qk_dqk_dqr);
            return DDflambdaGPU(qk) * dqk;
        }
    );

    // Compute dsigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.begin(), d_QRv.begin(), d_dQKv.begin(), d_dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_QKv.end(), d_QRv.end(), d_dQKv.end(), d_dQRv.end())),
        d_dsigmaR.begin(),
        [] __device__(thrust::tuple<double, double, double, double> qk_qr_dqk_dqr) {
            double qk = thrust::get<0>(qk_qr_dqk_dqr);
            double qr = thrust::get<1>(qk_qr_dqk_dqr);
            double dqk = thrust::get<2>(qk_qr_dqk_dqr);
            double dqr = thrust::get<3>(qk_qr_dqk_dqr);
            return DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaR_qK = ConvAGPU(d_sigmaR, d_QKv, 1.0, d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(d_sigmaK, d_QRv, 1.0, d_integ, d_theta);
    thrust::device_vector<double> convA_dsigmaR_qK = ConvAGPU(d_dsigmaR, d_QKv, t, d_integ, d_theta);
    thrust::device_vector<double> convA_dsigmaK_qR = ConvAGPU(d_dsigmaK, d_QRv, t, d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaR_dqK = ConvAGPU(d_sigmaR, d_dQKv, t, d_integ, d_theta);
    thrust::device_vector<double> convA_sigmaK_dqR = ConvAGPU(d_sigmaK, d_dQRv, t, d_integ, d_theta);

    // Compute final result
    double result = convA_sigmaR_qK.front() +
                    convA_sigmaK_qR.front() +
                    convA_dsigmaR_qK.front() +
                    convA_dsigmaK_qR.front() +
                    convA_sigmaR_dqK.front() +
                    convA_sigmaK_dqR.front() +
                    (d_dsigmaK.front() * d_QKv.front() + d_sigmaK.front() * d_dQKv.front()) / T0;

    return result;
}

double energyGPU(
    const thrust::device_vector<double>& d_QKv,
    const thrust::device_vector<double>& d_QRv,
    const thrust::device_vector<double>& d_t1grid,
    const thrust::device_vector<double>& d_integ,
    const thrust::device_vector<double>& d_theta,
    double T0) {
    
    size_t len = d_theta.size();
    thrust::device_vector<double> d_sigmaK(len, 0.0);

    thrust::device_vector<double> d_qK = getLastLenEntriesGPU(d_QKv, len);
    thrust::device_vector<double> d_qR = getLastLenEntriesGPU(d_QRv, len);

    // Compute sigmaK
    thrust::transform(
        d_qK.begin(), d_qK.end(),
        d_sigmaK.begin(),
        [] __device__ (double qk) { return DflambdaGPU(qk); }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(d_sigmaK, d_qR, d_t1grid.back(), d_integ, d_theta);

    // Compute final result
    double result = - (convA_sigmaK_qR.front() + Dflambda(d_qK.front()) / T0);

    

    return result;
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

    // cout << endl << "dqK: " << dqK[dqK.size()-510] << endl;
    // double sum = std::accumulate(dqK.begin(), dqK.end(), 0.0);
    // cout << "Sum of dqK: " << sum - len << endl;
    // sum = std::accumulate(dQKv.begin(), dQKv.end(), 0.0);
    // cout << "Append: Sum of dQKv: " << sum << endl;
    for (size_t i = 0; i < length; i++)
    {
        QKv.push_back(qK[i]);
        QRv.push_back(qR[i]);
        dQKv.push_back(tdiff * dqK[i]);
        dQRv.push_back(tdiff * dqR[i]);
    }

    // sum = std::accumulate(dQKv.begin(), dQKv.end(), 0.0);
    // cout << "Append: Sum of dQKv: " << sum << endl;

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
                                        const thrust::device_ptr<double>& src, double size, double scale = 1.0) {
    size_t required_size = dest.size() + size;

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
        thrust::copy(src, src + size, dest.begin() + insert_pos);
    }
    else {
        thrust::transform(
        src, src + size,
        dest.begin() + insert_pos,
        [scale] __device__ (double val) {
            return val * scale;
        }
    );
    }
}

void appendAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update d_t1grid and d_delta_t_ratio
    d_t1grid.push_back(t);
    size_t idx = d_t1grid.size() - 1;
    double tdiff = d_t1grid[idx] - d_t1grid[idx - 1];
    if (idx > 1) {
        double prev = d_t1grid[idx - 1] - d_t1grid[idx - 2];
        d_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        d_delta_t_ratio.push_back(0.0);
    }

    appendGPU(d_QKv,qK);
    appendGPU(d_QRv,qR);
    // d_QKv.reserve(d_QKv.size() + length);
    // thrust::copy(qK.begin(), qK.end(), d_QKv.end() - qK.size());
    // d_QRv.reserve(d_QRv.size() + length);
    // d_QRv.insert(d_QRv.end(), qR.begin(), qR.end());

    // cout << endl << "d_dqK: " << dqK[dqK.size()-510] << endl;
    // double sum = thrust::reduce(dqK.begin(), dqK.end(), 0.0, thrust::plus<double>());
    // cout << "Sum of d_dqK: " << sum - len << endl;
    // sum = thrust::reduce(d_dQKv.begin(), d_dQKv.end(), 0.0, thrust::plus<double>());
    // cout << "AppendGPU: Sum of dQKv: " << sum << endl;
    appendGPU(d_dQKv, dqK, tdiff);
    appendGPU(d_dQRv, dqR, tdiff);

    // sum = thrust::reduce(d_dQKv.begin(), d_dQKv.end(), 0.0, thrust::plus<double>());
    // cout << "AppendGPU: Sum of dQKv: " << sum << endl;

    // 2) finally update drvec and rvec
    d_drvec.push_back(tdiff * dr);
    d_rvec.push_back(rstepGPU(d_QKv, d_QRv, d_t1grid, d_integ, d_theta, Gamma, T0));
}

void appendAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len)
{
    // 1) update d_t1grid and d_delta_t_ratio
    d_t1grid.push_back(t);
    size_t idx = d_t1grid.size() - 1;
    double tdiff = d_t1grid[idx] - d_t1grid[idx - 1];
    if (idx > 1) {
        double prev = d_t1grid[idx - 1] - d_t1grid[idx - 2];
        d_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        d_delta_t_ratio.push_back(0.0);
    }

    appendGPU_ptr(d_QKv, qK, len);
    appendGPU_ptr(d_QRv, qR, len);
    // d_QKv.reserve(d_QKv.size() + length);
    // thrust::copy(qK.begin(), qK.end(), d_QKv.end() - qK.size());
    // d_QRv.reserve(d_QRv.size() + length);
    // d_QRv.insert(d_QRv.end(), qR.begin(), qR.end());

    // cout << endl << "d_dqK: " << dqK[dqK.size()-510] << endl;
    // double sum = thrust::reduce(dqK.begin(), dqK.end(), 0.0, thrust::plus<double>());
    // cout << "Sum of d_dqK: " << sum - len << endl;
    // sum = thrust::reduce(d_dQKv.begin(), d_dQKv.end(), 0.0, thrust::plus<double>());
    // cout << "AppendGPU: Sum of dQKv: " << sum << endl;
    appendGPU_ptr(d_dQKv, dqK, len, tdiff);
    appendGPU_ptr(d_dQRv, dqR, len, tdiff);

    // sum = thrust::reduce(d_dQKv.begin(), d_dQKv.end(), 0.0, thrust::plus<double>());
    // cout << "AppendGPU: Sum of dQKv: " << sum << endl;

    // 2) finally update drvec and rvec
    d_drvec.push_back(tdiff * dr);
    d_rvec.push_back(rstepGPU(d_QKv, d_QRv, d_t1grid, d_integ, d_theta, Gamma, T0));
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
    const double t)
{   
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = d_QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        d_t1grid.back() = t;
        double tdiff = (d_t1grid[d_t1grid.size() - 1] - d_t1grid[d_t1grid.size() - 2]);

        if (d_t1grid.size() > 2) {
            d_delta_t_ratio.back() = tdiff /
                (d_t1grid[d_t1grid.size() - 2] - d_t1grid[d_t1grid.size() - 3]);
        }
        else {
            d_delta_t_ratio.back() = 0.0;
        }

        thrust::copy(qK.begin(), qK.end(), d_QKv.begin() + length);
        thrust::copy(qR.begin(), qR.end(), d_QRv.begin() + length);
        thrust::transform(dqK.begin(), dqK.end(), d_dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
        thrust::transform(dqR.begin(), dqR.end(), d_dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

        d_drvec.back() = tdiff * dr;
        d_rvec.back() = rstepGPU(d_QKv, d_QRv, d_t1grid, d_integ, d_theta, Gamma, T0);
    }
}

void replaceAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len)
{   
    // Replace the existing values in the vectors with the new values
    size_t length = d_QKv.size() - len;
    d_t1grid.back() = t;
    double tdiff = (d_t1grid[d_t1grid.size() - 1] - d_t1grid[d_t1grid.size() - 2]);

    if (d_t1grid.size() > 2) {
        d_delta_t_ratio.back() = tdiff /
            (d_t1grid[d_t1grid.size() - 2] - d_t1grid[d_t1grid.size() - 3]);
    }
    else {
        d_delta_t_ratio.back() = 0.0;
    }

    thrust::copy(qK, qK + len, d_QKv.begin() + length);
    thrust::copy(qR, qR + len, d_QRv.begin() + length);
    thrust::transform(dqK, dqK + len, d_dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
    thrust::transform(dqR, dqR + len, d_dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

    d_drvec.back() = tdiff * dr;
    d_rvec.back() = rstepGPU(d_QKv, d_QRv, d_t1grid, d_integ, d_theta, Gamma, T0);
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

    return result;
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

    return result;
}

__global__ void bsearch_interp_kernel(
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
    const thrust::device_vector<double>& elem)
{
    size_t list_size = list.size();
    size_t elem_size = elem.size();

    thrust::device_vector<double> result(elem_size);

    int threads = 256;
    int blocks = (elem_size + threads - 1) / threads;

    bsearch_interp_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(list.data()),
        thrust::raw_pointer_cast(elem.data()),
        thrust::raw_pointer_cast(result.data()),
        list_size,
        elem_size
    );

    cudaDeviceSynchronize(); // Optional, remove if used asynchronously
    return result;
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

    return result;
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

            return out;
        }
    );

    return result;
}

// __global__ void isearchPosSortedInitGPU_kernel(
//     const double* __restrict__ list,
//     const double* __restrict__ elem,
//     const double* __restrict__ inits,
//     double* __restrict__ result,
//     size_t length,
//     double last,
//     size_t N)
// {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     double El = last * elem[i];
//     size_t n0 = static_cast<size_t>(floor(inits[i]));
//     size_t n1 = length;
//     size_t m = min(n0 + 1, length);
//     bool even = true;

//     double Lm = list[m - 1];

//     if (Lm > El)
//         n1 = max(m - 2, size_t(1));

//     double l0, l1;

//     // Manual loop with a maximum number of iterations to avoid divergence
//     #pragma unroll 8
//     for (int iter = 0; iter < 10; ++iter) {
//         if (n0 >= n1) break;

//         l0 = list[n0 - 1];
//         l1 = list[n1 - 1];

//         if (!(l0 <= El && El <= l1)) break;

//         even = !even;
//         if (even) {
//             double frac = (El - l0) / (l1 - l0);
//             m = n0 + static_cast<size_t>(round(frac * (n1 - n0)));
//         } else {
//             m = (n0 + n1) >> 1;
//         }

//         Lm = list[m - 1];

//         if (Lm == El) {
//             n0 = m;
//             n1 = m - 1;
//         } else if (Lm < El) {
//             n0 = m + 1;
//         } else {
//             n1 = m - 1;
//         }
//     }

//     double out;
//     if (Lm <= El) {
//         if (m == length) {
//             out = static_cast<double>(m);
//         } else {
//             double Lmp1 = list[m];
//             out = m + (El - Lm) / (Lmp1 - Lm);
//         }
//     } else {
//         if (m > 1) {
//             double Lmm1 = list[m - 2];
//             out = m - (El - Lm) / (Lmm1 - Lm);
//         } else {
//             out = static_cast<double>(m);
//         }
//     }

//     result[i] = out;
// }

// thrust::device_vector<double> isearchPosSortedInitGPU_fast(
//     const thrust::device_vector<double>& list,
//     const thrust::device_vector<double>& elem,
//     const thrust::device_vector<double>& inits)
// {
//     size_t N = elem.size();
//     size_t length = list.size();
//     double last = list.back();  // could be passed as a separate parameter if list is constant

//     thrust::device_vector<double> result(N);

//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;

//     isearchPosSortedInitGPU_kernel<<<blocks, threads>>>(
//         thrust::raw_pointer_cast(list.data()),
//         thrust::raw_pointer_cast(elem.data()),
//         thrust::raw_pointer_cast(inits.data()),
//         thrust::raw_pointer_cast(result.data()),
//         length,
//         last,
//         N
//     );

//     cudaDeviceSynchronize();  // or enqueue to a stream
//     return result;
// }

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

    // double sum = std::accumulate(dQKv.begin(), dQKv.end(), 0.0);
    // cout << "Sum of dQKv: " << sum << endl;

    // for (size_t i = 0; i < delta_t_ratio.size(); ++i) {
    //     std::cout << "delta_t_ratio[" << i << "] = " << delta_t_ratio[i] << std::endl;
    // }

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

void diffNfloor(
    const thrust::device_vector<double>& posB1x,
    thrust::device_vector<size_t>& Floor,
    thrust::device_vector<double>& diff) {
    double maxPosB1x = *thrust::max_element(posB1x.begin(), posB1x.end());
    size_t maxCeil = std::max(static_cast<size_t>(ceil(maxPosB1x)) - 1, size_t(1));


    thrust::transform(
        posB1x.begin(), posB1x.end(),
        Floor.begin(),
        [maxCeil] __device__ (double pos) {
            size_t floored = static_cast<size_t>(floor(pos));
            if (floored < 1) return size_t(1);
            if (floored > maxCeil) return maxCeil;
            return floored;
        }
    );

    thrust::transform(
        thrust::device,
        Floor.begin(), Floor.end(),
        posB1x.begin(),
        diff.begin(),
        [] __device__ (size_t floor_val, double pos_val) {
            return static_cast<double>(floor_val) - pos_val;
        }
    );
}

void interpolateGPU(
    const double* posB1xIn = nullptr,
    const double* posB2xIn = nullptr,
    const bool same = false) {

    // Compute d_posB1x
    d_posB1xOld = (posB1xIn ?
        (same ? thrust::device_vector<double>(posB1xIn,posB1xIn+len) : bsearchPosSortedGPU(d_t1grid, d_theta)) :
        bsearchPosSortedGPU(d_t1grid, d_theta));

    // Compute d_posB2x
    d_posB2xOld = (posB2xIn ?
        (same ? thrust::device_vector<double>(posB2xIn,posB2xIn+len*len) : bsearchPosSortedGPU(d_t1grid, d_phi2)) :
        bsearchPosSortedGPU(d_t1grid, d_phi2));

    // Interpolate QKA1int and QRA1int
    if (d_t1grid.back() > 0) {
        indexVecLN3GPU(d_weightsA1y, d_indsA1y, d_QKA1int, d_QRA1int);
    } else {
        d_QKA1int.assign(len * len, d_QKv[0]);
        d_QRA1int.assign(len * len, d_QRv[0]);
    }
    SigmaKGPU(d_QKA1int, d_SigmaKA1int);
    SigmaRGPU(d_QKA1int, d_QRA1int, d_SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (d_t1grid.back() > 0) {
        indexVecLN3GPU(d_weightsA2y, d_indsA2y, d_QKA2int, d_QRA2int);
    } else {
        d_QKA2int.assign(len * len, d_QKv[0]);
        d_QRA2int.assign(len * len, d_QRv[0]);
    }
    SigmaRGPU(d_QKA2int, d_QRA2int, d_SigmaRA2int);

    // Interpolate QKB1int and QRB1int
    thrust::device_vector<size_t> Floor(d_posB1xOld.size());
    thrust::device_vector<double> diff(Floor.size());

    diffNfloor(d_posB1xOld, Floor, diff);

    // double sum = thrust::reduce(d_dQKv.begin(), d_dQKv.end(), 0.0, thrust::plus<double>());
    // cout << "Sum of d_dQKv: " << sum << endl;

    // std::vector<double> diff_host(d_delta_t_ratio.size());
    // thrust::copy(d_delta_t_ratio.begin(), d_delta_t_ratio.end(), diff_host.begin());

    // for (size_t i = 0; i < d_delta_t_ratio.size(); ++i) {
    //     std::cout << "d_delta_t_ratio[" << i << "] = " << diff_host[i] << std::endl;
    // }

    if (d_t1grid.back() > 0) {
        indexVecNGPU(diff, Floor, d_delta_t_ratio, d_QKB1int, d_QRB1int, d_QKv, d_QRv, d_dQKv, d_dQRv, len);
    } else {
        d_QKB1int.assign(len * len, d_QKv[0]);
        d_QRB1int.assign(len * len, d_QRv[0]);
    }
    SigmaKGPU(d_QKB1int, d_SigmaKB1int);
    SigmaRGPU(d_QKB1int, d_QRB1int, d_SigmaRB1int);

    // Interpolate QKB2int and QRB2int
    if (d_t1grid.back() > 0) {
        indexMatAllGPU(d_posB2xOld, d_indsB2y, d_weightsB2y, d_delta_t_ratio, d_QKB2int, d_QRB2int, d_QKv, d_QRv, d_dQKv, d_dQRv, len);
    } else {
        d_QKB2int.assign(len * len, d_QKv[0]);
        d_QRB2int.assign(len * len, d_QRv[0]);
    }
    SigmaKGPU(d_QKB2int, d_SigmaKB2int);
    SigmaRGPU(d_QKB2int, d_QRB2int, d_SigmaRB2int);

    // Interpolate rInt
    if (d_t1grid.back() > 0) {
        indexVecR2GPU(d_rvec, d_drvec, diff, Floor, d_delta_t_ratio, d_rInt);
    } else {
        d_rInt.assign(len, d_rvec[0]);
    }
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
    return out;
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

    // QKv.resize(inds.size()*len);  
    // QRv.resize(inds.size()*len);
    // dQKv.resize(inds.size()*len);
    // dQRv.resize(inds.size()*len);
    // rvec.resize(inds.size());
    // drvec.resize(inds.size());
    // t1grid.resize(inds.size());
    delta_t_ratio.resize(inds.size());    

    // for (int i = 0; i < tfac.size(); ++i) {
    //     std::cerr << "tfac at index " << i << ": " << tfac[i] << std::endl;
    // }

    // for (int i = 0; i < drvec.size(); ++i) {
    //     std::cerr << "drvec at index " << i << ": " << drvec[i] << std::endl;
    // }

    // for (int i = 0; i < t1grid.size(); ++i) {
    //      std::cerr << "dQRv at index " << i << ": " << dQRv[i*len] << std::endl;
    // }

    // for (int i = 0; i < tfac.size(); ++i) {
    //     std::cerr << "tfac at index " << i << ": " << indsD[i] << std::endl;
    // }

    // for (int i = 0; i < inds.size(); ++i) {
    //     std::cerr << "inds at index " << i << ": " << inds[i] << std::endl;
    // }

    // std::cerr << "Size of dQKv: " << dQKv.size() << std::endl;

    interpolate();
}

// GPU gather kernel
__global__ void gatherKernel(const double* __restrict__ d_v,
                            const size_t* __restrict__ d_idxs,
                            const double* __restrict__ d_scale,
                            double* __restrict__ d_out,
                            size_t len, size_t n_chunks, bool use_scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks * len) return;

    size_t chunk = i / len;
    size_t j = i % len;

    size_t offset = d_idxs[chunk] * len;
    double factor = use_scale ? d_scale[chunk] : 1.0;
    d_out[i] = factor * d_v[offset + j];
}

thrust::device_vector<double> gatherGPU(const thrust::device_vector<double>& d_v,
                                        const thrust::device_vector<size_t>& d_idxs,
                                        size_t len,
                                        const thrust::device_vector<double>& d_scale = {}) {
    size_t n_chunks = d_idxs.size();
    thrust::device_vector<double> d_out(n_chunks * len);

    size_t threads = 256;
    size_t blocks = (n_chunks * len + threads - 1) / threads;

    bool use_scale = !d_scale.empty();
    gatherKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_v.data()),
        thrust::raw_pointer_cast(d_idxs.data()),
        use_scale ? thrust::raw_pointer_cast(d_scale.data()) : nullptr,
        thrust::raw_pointer_cast(d_out.data()),
        len, n_chunks, use_scale);

    return d_out;
}

__global__ void computeSparsifyFlags(const double* __restrict__ d_t1grid,
                                     const double* __restrict__ d_QKv,
                                     const double* __restrict__ d_QRv,
                                     const double* __restrict__ d_dQKv,
                                     const double* __restrict__ d_dQRv,
                                     bool* __restrict__ d_flags,
                                     double threshold, size_t len, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i + 1 >= n) return;
    if (i/2 * 2 != i) return; // Ensure i is even

    double tleft = d_t1grid[i - 2];
    double tmid = d_t1grid[i];
    double tdiff1 = d_t1grid[i - 1] - tleft;
    double tdiff2 = tmid - tleft;
    double tdiff3 = d_t1grid[i + 1] - tmid;
    double scale = tdiff2 / 12.0;

    double val = 0.0;
    for (size_t j = 0; j < len; ++j) {
        size_t idx_im2 = (i - 2) * len + j;
        size_t idx_im1 = (i - 1) * len + j;
        size_t idx_i   = i * len + j;
        size_t idx_ip1 = (i + 1) * len + j;

        double df1_qk = d_dQKv[idx_im1];
        double df2_qk = d_dQKv[idx_ip1];
        double f_qk = d_QKv[idx_i] - d_QKv[idx_im2];

        double df1_qr = d_dQRv[idx_im1];
        double df2_qr = d_dQRv[idx_ip1];
        double f_qr = d_QRv[idx_i] - d_QRv[idx_im2];

        val += fabs(scale * (2.0 * f_qk - tdiff2 * (df1_qk / tdiff1 + df2_qk / tdiff3)));
        val += fabs(scale * (2.0 * f_qr - tdiff2 * (df1_qr / tdiff1 + df2_qr / tdiff3)));
    }
    d_flags[i] = (val >= threshold);
}

void sparsifyNscaleGPU(double threshold) {

    size_t t1len = d_t1grid.size();
    thrust::device_vector<bool> d_flags(t1len, true);

    size_t threads = 256;
    size_t blocks = (t1len - 2 + threads - 1) / threads;
    computeSparsifyFlags<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_t1grid.data()),
        thrust::raw_pointer_cast(d_QKv.data()),
        thrust::raw_pointer_cast(d_QRv.data()),
        thrust::raw_pointer_cast(d_dQKv.data()),
        thrust::raw_pointer_cast(d_dQRv.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        threshold, len, t1len);

    thrust::device_vector<size_t> d_inds(t1len);
    thrust::sequence(d_inds.begin(), d_inds.end());

    size_t n = d_inds.size();
    thrust::device_vector<size_t> d_filtered(t1len); // max possible size
    auto end_it = thrust::copy_if(
        d_inds.begin(), d_inds.end(), 
        d_flags.begin(), 
        d_filtered.begin(), 
        thrust::identity<bool>()
    );
    d_filtered.resize(end_it - d_filtered.begin()); // shrink to actual size

    // Construct d_indsD and tfac
    thrust::device_vector<size_t> d_indsD(d_filtered.size(),0);
    thrust::transform(d_filtered.begin(), d_filtered.end() - 1, d_indsD.begin() + 1, d_indsD.begin() + 1, thrust::placeholders::_1 + 1);

    thrust::device_vector<double> d_tfac(d_filtered.size(), 1.0);
    const double* t1_ptr = thrust::raw_pointer_cast(d_t1grid.data());

    d_tfac[0]= 1.0;
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            d_filtered.begin() + 1,       // inds[i]
            d_filtered.begin(),           // inds[i-1]
            d_indsD.begin() + 1      // indsD[i]
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_filtered.end(),             // one past last valid index
            d_filtered.end() - 1,
            d_indsD.end()
        )),
        d_tfac.begin() + 1,
        [t1_ptr] __device__ (thrust::tuple<size_t, size_t, size_t> tup) {
            size_t inds_i    = thrust::get<0>(tup);
            size_t inds_im1  = thrust::get<1>(tup);
            size_t indsD_i   = thrust::get<2>(tup);
            return (t1_ptr[inds_i] - t1_ptr[inds_im1]) /
                (t1_ptr[indsD_i] - t1_ptr[indsD_i - 1]);
        });

    d_QKv = gatherGPU(d_QKv, d_filtered, len);
    d_QRv = gatherGPU(d_QRv, d_filtered, len);
    d_dQKv = gatherGPU(d_dQKv, d_indsD, len, d_tfac);
    d_dQRv = gatherGPU(d_dQRv, d_indsD, len, d_tfac);
    d_rvec = gatherGPU(d_rvec, d_filtered, 1);
    d_drvec = gatherGPU(d_drvec, d_indsD, 1, d_tfac);
    d_t1grid = gatherGPU(d_t1grid, d_filtered, 1);

    size_t new_n = d_t1grid.size();
    d_delta_t_ratio.resize(new_n);
    thrust::device_vector<double> d_dgrid(new_n);
    thrust::transform(d_t1grid.begin() + 1, d_t1grid.end(), d_t1grid.begin(), d_dgrid.begin() + 1, thrust::minus<>());
    thrust::transform(d_dgrid.begin() + 2, d_dgrid.end(), d_dgrid.begin() + 1, d_delta_t_ratio.begin() + 2, thrust::divides<>());

    vector<double> gpu_result(d_tfac.size());
    thrust::copy(d_tfac.begin(), d_tfac.end(), gpu_result.begin());

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

double SSPRK104GPU() {
    constexpr size_t stages = 10;
    constexpr size_t posCount = 3;
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
    const double avec[stages] = {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6};
    const double bvec[stages] = {1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10};
    const double b2vec[stages] = {0., 2.0 / 9, 0, 0, 5.0 / 18, 1.0 / 3, 0., 0., 0., 1.0 / 6};

    // Initialize variables
    thrust::device_vector<double> gKvec((stages + 1) * len, 0.0);
    set_slice(gKvec,0,getLastLenEntriesGPU(d_QKv, len));
    thrust::device_vector<double> gRvec((stages + 1) * len, 0.0);
    set_slice(gRvec,0,getLastLenEntriesGPU(d_QRv, len));
    thrust::device_vector<double> hKvec(stages * len, 0.0);
    thrust::device_vector<double> hRvec(stages * len, 0.0);
    thrust::device_vector<double> posB1xvec(posCount * len, 0.0);
    thrust::device_vector<double> posB2xvec(posCount * len * len, 0.0);

    thrust::device_vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = d_t1grid.back();
    thrust::device_vector<double> htvec(stages, 0.0);

    thrust::device_vector<double> gKe(len, 0.0);
    thrust::device_vector<double> gRe(len, 0.0);
    double gte = 0.0;
    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (d_QKv.size() == len || n != 0) {
            interpolateGPU(
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(posB1xvec, 0, len).get() : (n == 6 ? get_slice_ptr(posB1xvec, 1, len).get() : (n == 7 ? get_slice_ptr(posB1xvec, 2, len).get() : thrust::raw_pointer_cast(d_posB1xOld.data()))))),
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(posB2xvec, 0, len*len).get() : (n == 6 ? get_slice_ptr(posB2xvec, 1, len*len).get() : (n == 7 ? get_slice_ptr(posB2xvec, 2, len*len).get() : thrust::raw_pointer_cast(d_posB2xOld.data()))))),
                (n == 5 || n == 6 || n == 7)
            );
            // interpolateGPU(thrust::raw_pointer_cast(d_posB1xOld.data()),thrust::raw_pointer_cast(d_posB2xOld.data()),true);
        }

        // Update position vectors
        if (n == 2) {
            set_slice(posB1xvec,0,d_posB1xOld);
            set_slice(posB2xvec,0,d_posB2xOld);
        } else if (n == 3) {
            set_slice(posB1xvec,1,d_posB1xOld);
            set_slice(posB2xvec,1,d_posB2xOld);
        } else if (n == 4) {
            set_slice(posB1xvec,2,d_posB1xOld);
            set_slice(posB2xvec,2,d_posB2xOld);
        }

        // Compute k[n]
        // if (n == 1) {
        //     double sum = thrust::reduce(d_QKB1int.begin(), d_QKB1int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QKB1int: " << sum - d_QKB1int.size() << endl;
        //     sum = thrust::reduce(d_QKB2int.begin(), d_QKB2int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QKB2int: " << sum - d_QKB2int.size() << endl;
        //     sum = thrust::reduce(d_QKA1int.begin(), d_QKA1int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QKA1int: " << sum - d_QKA1int.size() << endl;
        //     sum = thrust::reduce(d_QRA1int.begin(), d_QRA1int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QRA1int: " << sum - d_QRA1int.size() << endl;
        //     sum = thrust::reduce(d_QRA2int.begin(), d_QRA2int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QRA2int: " << sum - d_QRA2int.size() << endl;
        //     sum = thrust::reduce(d_QRB1int.begin(), d_QRB1int.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_QRB1int: " << sum - d_QRB1int.size() << endl;
        //     sum = thrust::reduce(d_indsB2y.begin(), d_indsB2y.end(), 0.0, thrust::plus<double>());
        //     cout << "Sum of d_indsB2y: " << sum << endl;
        //     temp_gpu.resize(d_weightsB2y.size());
        //     thrust::copy(d_weightsB2y.begin(), d_weightsB2y.end(), temp_gpu.begin());
        // }
        set_slice(hKvec,n,QKstepGPU(d_QKv, d_QRv, d_QKB1int, d_QKB2int, d_QKA1int, d_QRA1int, d_QRA2int, d_QRB1int, d_SigmaRA1int, d_SigmaRA2int, d_SigmaKB1int, d_SigmaKB2int, d_SigmaKA1int, d_SigmaRB1int, d_theta, d_t1grid, d_rInt, T0, Gamma));
        set_slice(hRvec,n,QRstepGPU(d_QRv, d_rInt, d_SigmaRA2int, d_QRB2int, d_QRA2int, d_SigmaRB2int, d_t1grid));
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            thrust::device_vector<double> tempK = get_slice(hKvec,0,len);
            thrust::device_vector<double> tempR = get_slice(hRvec,0,len);
            thrust::device_ptr<double> tempK_ptr = get_slice_ptr(hKvec,0,len);
            thrust::device_ptr<double> tempR_ptr = get_slice_ptr(hRvec,0,len);
            thrust::device_vector<double> lastQKv(d_QKv.end() - len, d_QKv.end());
            thrust::device_vector<double> lastQRv(d_QRv.end() - len, d_QRv.end());
            dr = drstep2GPU(lastQKv, lastQRv, tempK, tempR, d_t1grid.back(), T0);
            set_slice(gKvec,1,MAGPU(get_slice_ptr(gKvec,0,len), tempK_ptr, delta_t * amat[1][0], len));
            set_slice(gRvec,1,MAGPU(get_slice_ptr(gRvec,0,len), tempR_ptr, delta_t * amat[1][0], len));
            gtvec[n + 1] = gtvec[0] + delta_t * amat[1][0] * htvec[0];
            appendAllGPU_ptr(get_slice_ptr(gKvec,1,len), get_slice_ptr(gRvec,1,len), tempK_ptr, tempR_ptr, htvec[0] * dr, gtvec[n + 1], len); // Append Update
        } else if (n == stages - 1) {
            set_slice_ptr(gKvec,n + 1,get_slice_ptr(gKvec, 0, len),len);
            set_slice_ptr(gRvec,n + 1,get_slice_ptr(gRvec, 0, len),len);
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                set_slice(gKvec,n + 1, MAGPU(get_slice_ptr(gKvec,n + 1,len), get_slice_ptr(hKvec,j,len), delta_t * bvec[j], len));
                set_slice(gRvec,n + 1, MAGPU(get_slice_ptr(gRvec,n + 1,len), get_slice_ptr(hRvec,j,len), delta_t * bvec[j], len));
                gtvec[n + 1] += delta_t * bvec[j] * htvec[j];
            }
            replaceAllGPU_ptr(get_slice_ptr(gKvec,n + 1,len), get_slice_ptr(gRvec,n + 1,len), get_slice_ptr(hKvec,0,len), get_slice_ptr(hRvec,0,len), htvec[0] * dr, gtvec[n + 1], len); // Replace Update
        } else if (n==4) {
            set_slice_ptr(gKvec,n + 1,get_slice_ptr(gKvec, 0, len),len);
            set_slice_ptr(gRvec,n + 1,get_slice_ptr(gRvec, 0, len),len);
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                set_slice(gKvec,n + 1, MAGPU(get_slice_ptr(gKvec,n + 1,len), get_slice_ptr(hKvec,j,len), delta_t * amat[n + 1][j], len));
                set_slice(gRvec,n + 1, MAGPU(get_slice_ptr(gRvec,n + 1,len), get_slice_ptr(hRvec,j,len), delta_t * amat[n + 1][j], len));
                gtvec[n + 1] += delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAllGPU_ptr(get_slice_ptr(gKvec,n + 1,len), get_slice_ptr(gRvec,n + 1,len), get_slice_ptr(hKvec,0,len), get_slice_ptr(hRvec,0,len), htvec[0] * dr, gtvec[n + 1], len); // Replace Update
        } else {
            set_slice(gKvec,n + 1, MAGPU(get_slice_ptr(gKvec,n ,len), get_slice_ptr(hKvec,n,len), delta_t * avec[n], len));
            set_slice(gRvec,n + 1, MAGPU(get_slice_ptr(gRvec,n ,len), get_slice_ptr(hRvec,n,len), delta_t * avec[n], len));
            gtvec[n + 1] = gtvec[n] + delta_t * avec[n] * htvec[n];
            replaceAllGPU_ptr(get_slice_ptr(gKvec,n + 1,len), get_slice_ptr(gRvec,n + 1,len), get_slice_ptr(hKvec,0,len), get_slice_ptr(hRvec,0,len), htvec[0] * dr, gtvec[n + 1], len); // Replace Update
        }
        // double sum = thrust::reduce(d_QKv.begin(), d_QKv.end(), 0.0, thrust::plus<double>());
        // cout << "Sum of d_QKv: " << sum - d_t1grid.size() * len << endl;
        // sum = thrust::reduce(gKvec.begin()+(n+1)*len, gKvec.begin()+(n+2)*len, 0.0, thrust::plus<double>());
        // cout << "Sum of gKvec: " << sum - 0 << endl;

    }

    // Final interpolation
    interpolateGPU(thrust::raw_pointer_cast(d_posB1xOld.data()), thrust::raw_pointer_cast(d_posB2xOld.data()));

    // Compute ge
    gKe = get_slice(gKvec,0,len);
    gRe = get_slice(gRvec,0,len);
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        // gKe = SumGPU(gKe, scalarMultiply_ptr(get_slice_ptr(hKvec,j,len), delta_t * b2vec[j], len));
        MAGPU_ptr(thrust::device_ptr<double>(thrust::raw_pointer_cast(gKe.data())), get_slice_ptr(hKvec,j,len), delta_t * b2vec[j], gKe, len);
        MAGPU_ptr(thrust::device_ptr<double>(thrust::raw_pointer_cast(gRe.data())), get_slice_ptr(hRvec,j,len), delta_t * b2vec[j], gRe, len);
        gte += delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    const thrust::device_ptr<double>& gKvecEnd = get_slice_ptr(gKvec,stages,len); 
    const thrust::device_ptr<double>& gRvecEnd = get_slice_ptr(gRvec,stages,len); 
    double error = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(gKvecEnd, gKe.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(gKvecEnd + len, gKe.end())),
        AbsDiff{},
        0.0,
        thrust::plus<double>()
    );

    error += thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(gRvecEnd, gRe.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(gRvecEnd + len, gRe.end())),
        AbsDiff{},
        0.0,
        thrust::plus<double>()
    );

    error += abs(gtvec[stages] - gte);

    return error;
}

void init()
{
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

    copyVectorsToGPU();
}

int main() {

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


    // 2) Main loop
    while (t1grid.back() < tmax && loop < maxLoop) {

        delta_old = delta;
        delta = (gpu ? SSPRK104GPU() : SSPRK104());
        // cout << "delta CPU: " << delta << endl;
        // delta = SSPRK104GPU();
        // cout << "delta GPU: " << delta << endl;
        loop++;

        if (loop % 100000 == 0) {
            (gpu ? sparsifyNscaleGPU(delta_max) : sparsifyNscale(delta_max));
        }

        // primitive time-step adaptation
        if (false && delta < delta_max && loop > 5 &&
            (delta < 1.1 * delta_old || delta_old == 0) &&
            rmax / specRad > delta_t && (gpu ? d_delta_t_ratio.back() : delta_t_ratio.back()))
        {
            delta_t *= 1.01;
        }
        else if (delta > 2 * delta_max && delta_t > delta_t_min) {
            delta_t *= 0.9;
        }

        // display a video
        std::cout << "loop: " << loop
            << " time: " << (gpu ? d_t1grid.back() : t1grid.back())
            << " time step: " << delta_t
            << " delta: " << delta
            << " specRad: " << specRad
            << std::endl;

        // record QK(t,0) to file
        if( gpu ) {
            double t = d_t1grid.back();
            double qk0 = d_QKv[(t1grid.size() - 1) * len + 0];
            double energy = energyGPU(d_QKv, d_QRv, d_t1grid, d_integ, d_theta, T0); 
            corr << t << "\t" << energy << "\t" << qk0 << "\n";
        } else {
            double t = t1grid.back();
            double qk0 = QKv[(t1grid.size() - 1) * len + 0];
            vector<double> temp(len,0.0);
            SigmaK(getLastLenEntries(QKv, len),temp);
            double energy = -(ConvA(temp,getLastLenEntries(QRv, len),t)[0] + Dflambda(qk0)/T0); 
            corr << t << "\t" << energy << "\t" << qk0 << "\n";
        }
    }

    (gpu ? copyVectorsToCPU() : copyVectorsToGPU());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
    //    set_slice(d_QKv,1,get_slice(d_QKv,0,len));
    //    appendAllGPU_ptr(get_slice_ptr(d_QKv,1,len), get_slice_ptr(d_QRv,1,len), get_slice_ptr(d_dQKv,1,len), get_slice_ptr(d_dQRv,1,len), 0.1, 0.1, len); // Append Update
        indexVecLN3GPU(d_weightsA2y, d_indsA2y, d_QKA2int, d_QRA2int);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;

    std::cout << "Average time: " << avg_ms << " ms" << std::endl;

    // vector<double> cpu_result = ConvR(QKA1int, SigmaRB1int, t1grid.back());
    // thrust::device_vector<double> d_gpu_result = ConvRGPU_Shared2D(d_QKA1int, d_SigmaRB1int, d_t1grid.back(), d_integ, d_theta);

    // vector<double> gpu_result(d_gpu_result.size());
    // thrust::copy(d_gpu_result.begin(), d_gpu_result.end(), gpu_result.begin());
    
    // printVectorDifference(cpu_result, gpu_result);
    
    // interpolateGPU(d_posB1xOld, d_posB2xOld);
    // interpolate(posB1xOld, posB2xOld);

    // vector<double> cpu_result = QKstep();

    // thrust::device_vector<double> d_gpu_result = QKstepGPU(d_QKv, d_QRv, d_QKB1int, d_QKB2int, d_QKA1int, d_QRA1int, d_QRA2int, d_QRB1int, d_SigmaRA1int, d_SigmaRA2int, d_SigmaKB1int, d_SigmaKB2int, d_SigmaKA1int, d_SigmaRB1int, d_theta, d_t1grid, d_rInt, T0, Gamma);

    // vector<double> gpu_result(d_gpu_result.size());
    // thrust::copy(d_gpu_result.begin(), d_gpu_result.end(), gpu_result.begin());
    
    // printVectorDifference(cpu_result, gpu_result);

    // // Compute `floor` vector
    // vector<double> posB1x = posB1xOld;
    // thrust::device_vector<double> d_posB1x = d_posB1xOld;
    // double maxPosB1x = posB1x[0];
    // for (size_t i = 1; i < posB1x.size(); ++i) {
    //     if (posB1x[i] > maxPosB1x) {
    //         maxPosB1x = posB1x[i];
    //     }
    // }
    // size_t maxCeil = static_cast<size_t>(ceil(maxPosB1x)) - 1;
    // if (maxCeil < 1) {
    //     maxCeil = 1;
    // }
    // // Compute `Floor` vector
    // vector<size_t> Floor(posB1x.size());
    // for (size_t i = 0; i < posB1x.size(); ++i) {
    //     size_t flooredValue = static_cast<size_t>(floor(posB1x[i]));
    //     if (flooredValue < 1) {
    //         flooredValue = 1;
    //     }
    //     else if (flooredValue > maxCeil) {
    //         flooredValue = maxCeil;
    //     }
    //     Floor[i] = flooredValue;
    // }

    // // Compute `diff` vector
    // vector<double> diff(posB1x.size());
    // Subtract(vector<double>(Floor.begin(), Floor.end()), posB1xOld, diff);

    // // indexVecN(len, diff, Floor, delta_t_ratio, QKB1int, QRB1int);

    // thrust::device_vector<size_t> d_Floor(d_posB1x.size());
    // thrust::transform(
    //     d_posB1x.begin(), d_posB1x.end(),
    //     d_Floor.begin(),
    //     [maxCeil = static_cast<size_t>(ceil(d_posB1x.back())) - 1] __device__(double pos) {
    //         size_t flooredValue = static_cast<size_t>(floor(pos));
    //         return min_device(max_device(1, flooredValue), max_device(flooredValue, maxCeil));
    //     });

    // thrust::device_vector<double> d_diff = SubtractGPU(
    //     scalarMultiply(d_Floor, 1.0), d_posB1xOld);

    // // indexVecNGPU(d_diff, d_Floor, d_delta_t_ratio, d_QKB1int, d_QRB1int, d_QKv, d_QRv, d_dQKv, d_dQRv, len);
    // thrust::fill(d_QRB2int.begin(), d_QRB2int.end(), 0.0);
    // std::fill(QRB2int.begin(), QRB2int.end(), 0.0);

    // SSPRK104();
    // SSPRK104GPU();

    // cout << "Difference between CPU and GPU results:" << endl;
    // cout << "CPU result: " << SSPRK104() << endl;
    // cout << "GPU result: " << SSPRK104GPU() << endl;

    // printVectorDifference(temp_cpu, temp_gpu);

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

    return 0;
}