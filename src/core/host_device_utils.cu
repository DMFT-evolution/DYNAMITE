#include "core/host_device_utils.hpp"

void Product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) {
    if (a.size() != b.size()) throw std::invalid_argument("Product: size mismatch");
    out.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
}

void scaleVec(const std::vector<double>& in, double s, std::vector<double>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] * s;
}

void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Vectors must be of the same length.\n";
        std::cerr << "Size of vector a: " << a.size() << ", Size of vector b: " << b.size() << "\n";
        return;
    }
    double total = 0.0;
    std::cout << "Differences between vectors:\n";
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        std::cout << "Index " << i << ": |" << a[i] << " - " << b[i] << "| = " << diff << "\n";
        total += diff;
    }
    std::cout << "Total Differences between vectors: " << total << "\n";
}

thrust::device_vector<double> SubtractGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b) {
    assert(a.size() == b.size());
    thrust::device_vector<double> result(a.size());
    thrust::transform(a.begin(), a.end(), b.begin(), result.begin(), thrust::minus<double>());
    return result;
}

thrust::device_vector<double> scalarMultiply(const thrust::device_vector<double>& v, double s) {
    thrust::device_vector<double> r(v.size());
    thrust::transform(v.begin(), v.end(), r.begin(), [s] __device__(double x){ return x * s; });
    return r;
}

thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& v, double s, size_t len) {
    thrust::device_vector<double> r(len);
    thrust::transform(v, v + len, r.begin(), [s] __device__(double x){ return x * s; });
    return r;
}

__global__ void update_gK_gR_kernel(
    double* gK, const double* gKfinal, const double* gK0,
    double* gR, const double* gRfinal, const double* gR0, int N) {
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
                    cudaStream_t stream) {
    int N = gK.size();
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    update_gK_gR_kernel<<<numBlocks, blockSize, 0, stream>>>(
        thrust::raw_pointer_cast(gK.data()),
        thrust::raw_pointer_cast(gKfinal.data()),
        thrust::raw_pointer_cast(gK0.data()),
        thrust::raw_pointer_cast(gR.data()),
        thrust::raw_pointer_cast(gRfinal.data()),
        thrust::raw_pointer_cast(gR0.data()),
        N);
}

__global__ void FusedUpdateKernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ extra1,
    const double* __restrict__ extra2,
    const double* __restrict__ extra3,
    const double* __restrict__ delta,
    const double* __restrict__ subtract,
    double* __restrict__ out,
    const double* alpha,
    const double* beta,
    size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double result = alpha[0] * a[i] + beta[0] * b[i];
    if (extra1) result += extra1[i];
    if (extra2) result += extra2[i];
    if (extra3) result += extra3[i];
    if (subtract) result -= delta[i] * subtract[i];
    out[i] = result;
}

void FusedUpdate(const thrust::device_ptr<double>& a,
                 const thrust::device_ptr<double>& b,
                 const thrust::device_vector<double>& out,
                 const double* alpha,
                 const double* beta,
                 const thrust::device_vector<double>* delta,
                 const thrust::device_vector<double>* extra1,
                 const thrust::device_vector<double>* extra2,
                 const thrust::device_vector<double>* extra3,
                 const thrust::device_ptr<double>& subtract,
                 cudaStream_t stream) {
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
        subtract.get(),
        thrust::raw_pointer_cast(const_cast<thrust::device_vector<double>&>(out).data()),
        alpha,
        beta,
        N);
}
