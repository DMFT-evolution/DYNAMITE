#include "EOMs/runge_kutta.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include "math/math_ops.hpp"
#include "core/vector_utils.hpp"
#include "convolution/convolution.hpp"
#include "core/compute_utils.hpp"
#include "EOMs/time_steps.hpp"
#include "interpolation/interpolation_core.hpp"
#include "core/device_utils.cuh"
#include "math/math_sigma.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "core/console.hpp"
#include <cmath>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// Kernel implementations
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
        ATOMIC_ADD_DBL(result, sdata[0]);
    }
}

// GPU Runge-Kutta initialization functions
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

    rk->gK0.resize(config.len, 0.0);
    rk->gR0.resize(config.len, 0.0);
    rk->gK.resize(config.len, 0.0);
    rk->gR.resize(config.len, 0.0);
    rk->gKe.resize(config.len, 0.0);
    rk->gRe.resize(config.len, 0.0);
    rk->gKfinal.resize(config.len, 0.0);
    rk->gRfinal.resize(config.len, 0.0);

    rk->hK.resize(config.len * rk->stages, 0.0);
    rk->hR.resize(config.len * rk->stages, 0.0);
    rk->hK0.resize(config.len, 0.0);
    rk->hR0.resize(config.len, 0.0);

    if (config.debug) {
        // Basic size sanity inits
        const size_t expected = static_cast<size_t>(config.len) * static_cast<size_t>(rk->stages);
        if (rk->hK.size() != expected) {
            std::cerr << dmfe::console::ERR() << "RK54GPU init: hK size mismatch (" << rk->hK.size() << " vs " << expected << ")" << std::endl;
            abort();
        }
    }
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

    rk->gK0.resize(config.len, 0.0);
    rk->gR0.resize(config.len, 0.0);
    rk->gK.resize(config.len, 0.0);
    rk->gR.resize(config.len, 0.0);
    rk->gKfinal.resize(config.len, 0.0);
    rk->gRfinal.resize(config.len, 0.0);
    rk->gKe.resize(config.len, 0.0);
    rk->gRe.resize(config.len, 0.0);

    rk->posB1xvec.resize(rk->posCount * config.len, 0.0);
    rk->posB2xvec.resize(rk->posCount * config.len * config.len, 0.0);

    rk->hK.resize(config.len, 0.0);
    rk->hR.resize(config.len, 0.0);
    rk->hK0.resize(config.len, 0.0);
    rk->hR0.resize(config.len, 0.0);

    if (config.debug) {
        const size_t expected = static_cast<size_t>(config.len);
        if (rk->hK.size() != expected) {
            std::cerr << dmfe::console::ERR() << "SSPRK104GPU init: hK size mismatch (" << rk->hK.size() << " vs " << expected << ")" << std::endl;
            abort();
        }
    }
}

// Chebyshev polynomial of the first kind T_n(x) with long double precision
long double chebyshevT_ld(int n, long double x) {
    if (n == 0) return 1.0L;
    if (n == 1) return x;
    long double t0 = 1.0L, t1 = x;
    for (int i = 2; i <= n; ++i) {
        long double t2 = 2.0L * x * t1 - t0;
        t0 = t1;
        t1 = t2;
    }
    return t1;
}

// Chebyshev polynomial of the second kind U_n(x) with long double precision
long double chebyshevU_ld(int n, long double x) {
    if (n == 0) return 1.0L;
    if (n == 1) return 2.0L * x;
    long double u0 = 1.0L, u1 = 2.0L * x;
    for (int i = 2; i <= n; ++i) {
        long double u2 = 2.0L * x * u1 - u0;
        u0 = u1;
        u1 = u2;
    }
    return u1;
}

// Gaussian elimination solver with long double precision
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b) {
    int n = A.size();
    std::vector<std::vector<long double>> augmented = A;
    for (int i = 0; i < n; ++i) {
        augmented[i].push_back(b[i]);
    }
    
    // Forward elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        // Swap rows
        std::swap(augmented[i], augmented[maxRow]);
        
        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            long double factor = augmented[k][i] / augmented[i][i];
            for (int j = i; j <= n; ++j) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    
    // Back substitution
    std::vector<long double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = augmented[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= augmented[i][j] * x[j];
        }
        x[i] /= augmented[i][i];
    }
    
    return x;
}

// Main SERKcoeffs function with enhanced precision
std::vector<double> SERKcoeffs(int q) {
    const int m = 2;
    const long double μp = 19.0L / 20.0L;
    const int s = m * q;
    const long double α2 = static_cast<long double>(q) / (static_cast<long double>(s) * s);
    const long double w0 = 1.0L + μp / (static_cast<long double>(s) * s);
    const long double w1 = chebyshevT_ld(s, w0) / (static_cast<long double>(s) * chebyshevU_ld(s - 1, w0));
    
    const int num_eq = s + 1;
    const int num_var = s + 1;
    
    std::vector<std::vector<long double>> A(num_eq, std::vector<long double>(num_var, 0.0L));
    std::vector<long double> R_values(num_eq);
    
    for (int z_idx = 0; z_idx < num_eq; ++z_idx) {
        long double z = static_cast<long double>(z_idx + 1);
        long double R_z = chebyshevT_ld(s, w0 + w1 * z) / chebyshevT_ld(s, w0);
        R_values[z_idx] = R_z;
        
        // Coefficient for b[0] (bvec[[1]] in Mathematica)
        long double T0 = chebyshevT_ld(0, 1.0L + α2 * z);
        A[z_idx][0] = T0;
        
        // Coefficients for the sum terms
        for (int j = 1; j <= q; ++j) {
            long double Tm_j = std::pow(chebyshevT_ld(m, 1.0L + α2 * z), j - 1);
            for (int i = 1; i <= m; ++i) {
                long double Ti = chebyshevT_ld(i, 1.0L + α2 * z);
                int k = i + m * (j - 1);
                A[z_idx][k] = Ti * Tm_j;
            }
        }
    }
    
    // Solve the system A * bvec = R_values using long double precision
    std::vector<long double> bvec_ld = gaussianElimination_ld(A, R_values);
    
    // Convert back to double for output
    std::vector<double> bvec(bvec_ld.size());
    for (size_t i = 0; i < bvec_ld.size(); ++i) {
        bvec[i] = static_cast<double>(bvec_ld[i]);
    }
    
    return bvec;
}

void init_SERK2(int q) {
    std::vector<double> coeffs = SERKcoeffs(q);
    rk->stages = coeffs.size() - 1;
    rk->init = 2 + q / 2;

    rk->bvec = new double[rk->stages + 1];
    for (size_t i = 0; i <= rk->stages; ++i) {
        rk->bvec[i] = coeffs[i];
    }

    rk->gK0.resize(config.len, 0.0);
    rk->gR0.resize(config.len, 0.0);
    rk->gK.resize(config.len, 0.0);
    rk->gR.resize(config.len, 0.0);
    rk->gKfinal.resize(config.len, 0.0);
    rk->gRfinal.resize(config.len, 0.0);
    rk->gKe.resize(config.len, 0.0);
    rk->gRe.resize(config.len, 0.0);

    rk->hK.resize(config.len, 0.0);
    rk->hR.resize(config.len, 0.0);
    rk->hK0.resize(config.len, 0.0);
    rk->hR0.resize(config.len, 0.0);
}

// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool) {

    size_t t1len = sim->d_t1grid.size();
    double t_current = sim->d_t1grid.back();
    // Initialize variables
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR.begin());
    rk->gt = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK0.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR0.begin());
    rk->gt0 = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKfinal.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRfinal.begin());
    rk->gtfinal = t_current + config.delta_t;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKe.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRe.begin());
    rk->gte = t_current + config.delta_t;

    thrust::fill(sim->error_result.begin(), sim->error_result.end(), 0.0);

    double dr = 0.0;
    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->stages; ++n) {
        // Interpolation
        if (sim->d_QKv.size() == config.len || n != 0) {
            interpolateGPU();
        }

        QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, config.T0, config.Gamma, n, *pool);
        rk->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            rk->hK0.assign(rk->hK.begin(), rk->hK.begin() + config.len);
            rk->hR0.assign(rk->hR.begin(), rk->hR.begin() + config.len);
            computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK0.data()), thrust::raw_pointer_cast(rk->hK.data()), thrust::raw_pointer_cast(rk->d_avec.data()), thrust::raw_pointer_cast(rk->gK.data()), config.delta_t, n + 1, config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeWeightedSum gK0");
            computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR0.data()), thrust::raw_pointer_cast(rk->hR.data()), thrust::raw_pointer_cast(rk->d_avec.data()), thrust::raw_pointer_cast(rk->gR.data()), config.delta_t, n + 1, config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeWeightedSum gR0");
            if(rk->bvec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * rk->bvec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA gKfinal");
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * rk->bvec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA gRfinal");
            }
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * rk->b2vec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA gKe");
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * rk->b2vec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA gRe");
            }
            rk->gt = rk->gt0 + config.delta_t * rk->cvec[n] * rk->ht;
            dr = drstep2GPU(get_slice_ptr(sim->d_QKv, t1len - 1, config.len), get_slice_ptr(sim->d_QRv, t1len - 1, config.len), rk->hK0.data(), rk->hR0.data(), sim->d_t1grid.back(), config.T0, *pool);
            appendAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Append Update
        } else {
            if(rk->bvec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * rk->bvec[n], config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * rk->bvec[n], config.len);
            }
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * rk->b2vec[n], config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * rk->b2vec[n], config.len);
            }
            if (n != rk->stages - 1) {
                computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK0.data()), thrust::raw_pointer_cast(rk->hK.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gK.data()), config.delta_t, n + 1, config.len);
                computeWeightedSum<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR0.data()), thrust::raw_pointer_cast(rk->hR.data()), thrust::raw_pointer_cast(rk->d_avec.data()) + n * (n + 1) / 2, thrust::raw_pointer_cast(rk->gR.data()), config.delta_t, n + 1, config.len);
                rk->gt = rk->gt0 + config.delta_t * rk->cvec[n] * rk->ht;
                replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(rk->gKfinal.data(), rk->gRfinal.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gtfinal, config.len, *pool); // Replace Update
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
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeError");

    double error = sim->error_result[0];
    error += abs(rk->gtfinal - rk->gte);

    return error;
}

double SERK2GPU(int q, StreamPool* pool){

    double alpha = 1.0 / (4 * q);

    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    size_t t1len = sim->d_t1grid.size();
    double t_current = sim->d_t1grid.back();
    // Initialize variables
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR.begin());
    rk->gt = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK0.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR0.begin());
    rk->gt0 = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKfinal.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRfinal.begin());
    rk->gtfinal = t_current + config.delta_t;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKe.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRe.begin());
    rk->gte = t_current;
    computeScale<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), rk->bvec[0], config.len);
    computeScale<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), rk->bvec[0], config.len);

    thrust::fill(sim->error_result.begin(), sim->error_result.end(), 0.0);

    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < rk->stages; ++n) {
        // Interpolation
        interpolateGPU(thrust::raw_pointer_cast(sim->d_posB1xOld.data()), thrust::raw_pointer_cast(sim->d_posB2xOld.data()),pool);

        QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, config.T0, config.Gamma, n, *pool);
        rk->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            rk->hK0.assign(rk->hK.begin(), rk->hK.begin() + config.len);
            rk->hR0.assign(rk->hR.begin(), rk->hR.begin() + config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * alpha, config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gK");
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * alpha, config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gR");
            if(rk->bvec[n + 1] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->gK.data()), rk->bvec[n + 1], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gKfinal");
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->gR.data()), rk->bvec[n + 1], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gRfinal");
            }
            rk->gt = rk->gt0 + config.delta_t * alpha * rk->ht;
            dr = drstep2GPU(get_slice_ptr(sim->d_QKv, t1len - 1, config.len), get_slice_ptr(sim->d_QRv, t1len - 1, config.len), rk->hK0.data(), rk->hR0.data(), sim->d_t1grid.back(), config.T0, *pool);
            appendAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Append Update
        } else {
            if (n % 2 == 0){
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, config.delta_t * alpha, config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, config.delta_t * alpha, config.len);
                rk->gt +=  config.delta_t * alpha * rk->ht;
            } else {
                computeMAD<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()) + n * config.len, 2 * config.delta_t * alpha, config.len);
                computeMAD<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()) + n * config.len, 2 * config.delta_t * alpha, config.len);
                rk->gt = 2 * rk->gt - rk->gte + 2 * config.delta_t * alpha * rk->ht;
                rk->gKe.assign(rk->gK.begin(), rk->gK.begin() + config.len);
                rk->gRe.assign(rk->gR.begin(), rk->gR.begin() + config.len);
                rk->gte = rk->gt;
            }
            if(rk->bvec[n + 1] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->gK.data()), rk->bvec[n + 1], config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->gR.data()), rk->bvec[n + 1], config.len);
            }
            replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Replace Update
        }
    }
 
    // Compute error estimate
    computeError<<<blocks, threads, threads * sizeof(double), (*pool)[0]>>>(
        thrust::raw_pointer_cast(rk->gKfinal.data()),
        thrust::raw_pointer_cast(rk->gK.data()),
        thrust::raw_pointer_cast(rk->gRfinal.data()),
        thrust::raw_pointer_cast(rk->gR.data()),
        thrust::raw_pointer_cast(sim->error_result.data()),
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeError");

    double error = sim->error_result[0];
    error += abs(rk->gtfinal - rk->gt);

    return error;
}

double SSPRK104GPU(StreamPool* pool) {

    size_t t1len = sim->d_t1grid.size();
    double t_current = sim->d_t1grid.back();
    // Initialize variables
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR.begin());
    rk->gt = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gK0.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gR0.begin());
    rk->gt0 = t_current;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKfinal.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRfinal.begin());
    rk->gtfinal = t_current + config.delta_t;
    thrust::copy(sim->d_QKv.end() - config.len, sim->d_QKv.end(), rk->gKe.begin());
    thrust::copy(sim->d_QRv.end() - config.len, sim->d_QRv.end(), rk->gRe.begin());
    rk->gte = t_current + config.delta_t;

    thrust::fill(sim->error_result.begin(), sim->error_result.end(), 0.0);
    
    double dr = 0.0;
    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->stages; ++n) {
        // Interpolation
        if (sim->d_QKv.size() == config.len || n != 0) {
            interpolateGPU(
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->posB1xvec, 0, config.len).get() : (n == 6 ? get_slice_ptr(rk->posB1xvec, 1, config.len).get() : (n == 7 ? get_slice_ptr(rk->posB1xvec, 2, config.len).get() : thrust::raw_pointer_cast(sim->d_posB1xOld.data()))))),
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->posB2xvec, 0, config.len*config.len).get() : (n == 6 ? get_slice_ptr(rk->posB2xvec, 1, config.len*config.len).get() : (n == 7 ? get_slice_ptr(rk->posB2xvec, 2, config.len*config.len).get() : thrust::raw_pointer_cast(sim->d_posB2xOld.data()))))),
                (n == 5 || n == 6 || n == 7),
                pool);
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

        QKRstepGPU(sim->d_QKv, sim->d_QRv, sim->d_QKB1int, sim->d_QKB2int, sim->d_QKA1int, sim->d_QRA1int, sim->d_QRA2int, sim->d_QRB1int, sim->d_QRB2int, sim->d_SigmaRA1int, sim->d_SigmaRA2int, sim->d_SigmaKB1int, sim->d_SigmaKB2int, sim->d_SigmaKA1int, sim->d_SigmaRB1int, sim->d_SigmaRB2int, sim->d_integ, sim->d_theta, sim->d_t1grid, sim->d_rInt, rk->hK, rk->hR, config.T0, config.Gamma, 0, *pool);
        rk->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            rk->hK0 = rk->hK;
            rk->hR0 = rk->hR;
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->avec[n], config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gK");
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->avec[n], config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gR");
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->bvec[n], config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gKfinal");
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->bvec[n], config.len);
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gRfinal");
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->b2vec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gKe");
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->b2vec[n], config.len);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA gRe");
            }
            rk->gt += config.delta_t * rk->avec[n] * rk->ht;
            dr = drstep2GPU(get_slice_ptr(sim->d_QKv, t1len - 1, config.len), get_slice_ptr(sim->d_QRv, t1len - 1, config.len), rk->hK.data(), rk->hR.data(), t_current, config.T0, *pool);

            appendAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK.data(), rk->hR.data(), rk->ht * dr, rk->gt, config.len, *pool); // Append Update
        } else {
            if (n != rk->stages - 1) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gK.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->avec[n], config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gR.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->avec[n], config.len);
            }
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKfinal.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->bvec[n], config.len);
            computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRfinal.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->bvec[n], config.len);
            if(rk->b2vec[n] != 0.0) {
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gKe.data()), thrust::raw_pointer_cast(rk->hK.data()), config.delta_t * rk->b2vec[n], config.len);
                computeMA<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->gRe.data()), thrust::raw_pointer_cast(rk->hR.data()), config.delta_t * rk->b2vec[n], config.len);
            }
            if (n == 4) {
                AddSubtractGPU(rk->gK, rk->gKfinal, rk->gK0, rk->gR, rk->gRfinal, rk->gR0, (*pool)[0]);
                rk->gt = rk->gt0*9/15 + (rk->gt + config.delta_t * rk->avec[n] * rk->ht)*6/15;
            } else if (n != rk->stages - 1) {
                rk->gt += config.delta_t * rk->avec[n] * rk->ht;
            }
            if (n != rk->stages - 1) {
                replaceAllGPU_ptr(rk->gK.data(), rk->gR.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(rk->gKfinal.data(), rk->gRfinal.data(), rk->hK0.data(), rk->hR0.data(), rk->ht * dr, rk->gt, config.len, *pool); // Replace Update
            }
        }
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
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeError");

    double error = sim->error_result[0];
    error += abs(rk->gtfinal - rk->gte);

    return error;
}

// GPU method selection function
double updateGPU(StreamPool* pool) {
    if (rk->init == 1) {
        return RK54GPU(pool);
    } else if (rk->init == 2) {
        return SSPRK104GPU(pool);
    } else {
        return SERK2GPU(2 * (rk->init - 2), pool);
    }
}
