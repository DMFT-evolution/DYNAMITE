#include "core/gpu_memory_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "simulation/device_simulation_data.hpp"
#include "core/device_constants.hpp"
#include <thrust/copy.h>
#include "core/device_vector.hpp"
#include "core/config.hpp"
#include <iostream>
#include <stdexcept>
#include "core/console.hpp"

// d_p, d_p2, d_lambda declared in device_constants.hpp and defined in device_constants.cu

// External declarations
extern SimulationConfig config;

void copyVectorsToGPU(SimulationData& sim, size_t len) {
	// Scalars / simple lists
	sim.device->theta       = sim.host->theta;
	sim.device->phi1        = sim.host->phi1;
	sim.device->phi2        = sim.host->phi2;
	sim.device->posA1y      = sim.host->posA1y;
	sim.device->posA2y      = sim.host->posA2y;
	sim.device->posB2y      = sim.host->posB2y;
	sim.device->weightsA1y  = sim.host->weightsA1y;
	sim.device->weightsA2y  = sim.host->weightsA2y;
	sim.device->weightsB2y  = sim.host->weightsB2y;
	sim.device->posB1xOld   = sim.host->posB1xOld;
	sim.device->posB2xOld   = sim.host->posB2xOld;
	sim.device->integ       = sim.host->integ;

	sim.device->indsA1y     = sim.host->indsA1y;
	sim.device->indsA2y     = sim.host->indsA2y;
	sim.device->indsB2y     = sim.host->indsB2y;

	sim.device->t1grid        = sim.host->t1grid;
	sim.device->delta_t_ratio = sim.host->delta_t_ratio;

	sim.device->QKv   = sim.host->QKv;
	sim.device->QRv   = sim.host->QRv;
	sim.device->dQKv  = sim.host->dQKv;
	sim.device->dQRv  = sim.host->dQRv;
	sim.device->rInt  = sim.host->rInt;
	sim.device->drInt = sim.host->drInt;
	sim.device->rvec  = sim.host->rvec;
	sim.device->drvec = sim.host->drvec;

	sim.device->SigmaKA1int = sim.host->SigmaKA1int;
	sim.device->SigmaRA1int = sim.host->SigmaRA1int;
	sim.device->SigmaKB1int = sim.host->SigmaKB1int;
	sim.device->SigmaRB1int = sim.host->SigmaRB1int;
	sim.device->SigmaKA2int = sim.host->SigmaKA2int;
	sim.device->SigmaRA2int = sim.host->SigmaRA2int;
	sim.device->SigmaKB2int = sim.host->SigmaKB2int;
	sim.device->SigmaRB2int = sim.host->SigmaRB2int;

	sim.device->QKA1int = sim.host->QKA1int;
	sim.device->QRA1int = sim.host->QRA1int;
	sim.device->QKB1int = sim.host->QKB1int;
	sim.device->QRB1int = sim.host->QRB1int;
	sim.device->QKA2int = sim.host->QKA2int;
	sim.device->QRA2int = sim.host->QRA2int;
	sim.device->QKB2int = sim.host->QKB2int;
	sim.device->QRB2int = sim.host->QRB2int;

	// Workspace allocate (no host mirrors needed except optional debug)
	sim.device->convA1_1.resize(len);
	sim.device->convA2_1.resize(len);
	sim.device->convA1_2.resize(len);
	sim.device->convA2_2.resize(len);
	sim.device->convR_1.resize(len);
	sim.device->convR_2.resize(len);
	sim.device->convR_3.resize(len);
	sim.device->convR_4.resize(len);

	sim.device->temp0.resize(len);
	sim.device->temp1.resize(len);
	sim.device->temp2.resize(len);
	sim.device->temp3.resize(len);
	sim.device->temp4.resize(len);
	sim.device->temp5.resize(len);
	sim.device->temp6.resize(len);
	sim.device->temp7.resize(len);
	sim.device->temp8.resize(len);
	sim.device->temp9.resize(len);
	sim.device->temp10.resize(len);
	sim.device->temp11.resize(len);
	sim.device->temp12.resize(len);

	sim.device->Stemp0.resize(len);
	sim.device->Stemp1.resize(len);
	sim.device->Stemp2.resize(len);

	sim.device->error_result.resize(1, 0.0);

	if (config.debug) {
		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA error during host->device copy: ") + cudaGetErrorString(err));
		}
	}

	std::cout << dmfe::console::INFO() << "Host -> Device vector copy complete." << std::endl;
}

void copyVectorsToCPU(SimulationData& sim) {
	auto copyBack = [](auto& host, const auto& dev){ host.resize(dev.size()); thrust::copy(dev.begin(), dev.end(), host.begin()); };

	copyBack(sim.host->QKv, sim.device->QKv);
	copyBack(sim.host->QRv, sim.device->QRv);
	copyBack(sim.host->dQKv, sim.device->dQKv);
	copyBack(sim.host->dQRv, sim.device->dQRv);
	copyBack(sim.host->rvec, sim.device->rvec);
	copyBack(sim.host->drvec, sim.device->drvec);
	copyBack(sim.host->rInt, sim.device->rInt);
	copyBack(sim.host->drInt, sim.device->drInt);
	copyBack(sim.host->t1grid, sim.device->t1grid);
	copyBack(sim.host->delta_t_ratio, sim.device->delta_t_ratio);

	copyBack(sim.host->SigmaKA1int, sim.device->SigmaKA1int);
	copyBack(sim.host->SigmaRA1int, sim.device->SigmaRA1int);
	copyBack(sim.host->SigmaKB1int, sim.device->SigmaKB1int);
	copyBack(sim.host->SigmaRB1int, sim.device->SigmaRB1int);
	copyBack(sim.host->SigmaKA2int, sim.device->SigmaKA2int);
	copyBack(sim.host->SigmaRA2int, sim.device->SigmaRA2int);
	copyBack(sim.host->SigmaKB2int, sim.device->SigmaKB2int);
	copyBack(sim.host->SigmaRB2int, sim.device->SigmaRB2int);

	copyBack(sim.host->QKA1int, sim.device->QKA1int);
	copyBack(sim.host->QRA1int, sim.device->QRA1int);
	copyBack(sim.host->QKB1int, sim.device->QKB1int);
	copyBack(sim.host->QRB1int, sim.device->QRB1int);
	copyBack(sim.host->QKA2int, sim.device->QKA2int);
	copyBack(sim.host->QRA2int, sim.device->QRA2int);
	copyBack(sim.host->QKB2int, sim.device->QKB2int);
	copyBack(sim.host->QRB2int, sim.device->QRB2int);

	copyBack(sim.host->posA1y, sim.device->posA1y);
	copyBack(sim.host->posA2y, sim.device->posA2y);
	copyBack(sim.host->posB2y, sim.device->posB2y);
	copyBack(sim.host->posB1xOld, sim.device->posB1xOld);
	copyBack(sim.host->posB2xOld, sim.device->posB2xOld);

	// Optional: copy theta/phi only if needed
	copyBack(sim.host->theta, sim.device->theta);
	copyBack(sim.host->phi1, sim.device->phi1);
	copyBack(sim.host->phi2, sim.device->phi2);

	copyBack(sim.host->weightsA1y, sim.device->weightsA1y);
	copyBack(sim.host->weightsA2y, sim.device->weightsA2y);
	copyBack(sim.host->weightsB2y, sim.device->weightsB2y);
	copyBack(sim.host->integ, sim.device->integ);

	copyBack(sim.host->indsA1y, sim.device->indsA1y);
	copyBack(sim.host->indsA2y, sim.device->indsA2y);
	copyBack(sim.host->indsB2y, sim.device->indsB2y);

	std::cout << dmfe::console::INFO() << "Device -> Host vector copy complete." << std::endl;
}

void clearAllDeviceVectors(SimulationData& sim) {
	auto cl = [](auto& v){ v.clear(); v.shrink_to_fit(); };
	cl(sim.device->QKv); cl(sim.device->QRv); cl(sim.device->dQKv); cl(sim.device->dQRv);
	cl(sim.device->rInt); cl(sim.device->drInt); cl(sim.device->rvec); cl(sim.device->drvec);
	cl(sim.device->SigmaKA1int); cl(sim.device->SigmaRA1int); cl(sim.device->SigmaKB1int); cl(sim.device->SigmaRB1int);
	cl(sim.device->SigmaKA2int); cl(sim.device->SigmaRA2int); cl(sim.device->SigmaKB2int); cl(sim.device->SigmaRB2int);
	cl(sim.device->QKA1int); cl(sim.device->QRA1int); cl(sim.device->QKB1int); cl(sim.device->QRB1int);
	cl(sim.device->QKA2int); cl(sim.device->QRA2int); cl(sim.device->QKB2int); cl(sim.device->QRB2int);
	cl(sim.device->theta); cl(sim.device->phi1); cl(sim.device->phi2);
	cl(sim.device->posA1y); cl(sim.device->posA2y); cl(sim.device->posB2y);
	cl(sim.device->weightsA1y); cl(sim.device->weightsA2y); cl(sim.device->weightsB2y);
	cl(sim.device->posB1xOld); cl(sim.device->posB2xOld);
	cl(sim.device->indsA1y); cl(sim.device->indsA2y); cl(sim.device->indsB2y);
	cl(sim.device->integ); cl(sim.device->t1grid); cl(sim.device->delta_t_ratio);
	cl(sim.device->convA1_1); cl(sim.device->convA2_1); cl(sim.device->convA1_2); cl(sim.device->convA2_2);
	cl(sim.device->convR_1); cl(sim.device->convR_2); cl(sim.device->convR_3); cl(sim.device->convR_4);
	cl(sim.device->temp0); cl(sim.device->temp1); cl(sim.device->temp2); cl(sim.device->temp3); cl(sim.device->temp4);
	cl(sim.device->temp5); cl(sim.device->temp6); cl(sim.device->temp7); cl(sim.device->temp8); cl(sim.device->temp9);
	cl(sim.device->temp10); cl(sim.device->temp11); cl(sim.device->temp12);
	cl(sim.device->Stemp0); cl(sim.device->Stemp1); cl(sim.device->Stemp2);
	cl(sim.device->error_result);
	std::cout << dmfe::console::INFO() << "Cleared device vectors." << std::endl;
}

void clearAllHostVectors(SimulationData& sim) {
	auto cl = [](auto& v){ v.clear(); v.shrink_to_fit(); };
	cl(sim.host->theta); cl(sim.host->phi1); cl(sim.host->phi2);
	cl(sim.host->posA1y); cl(sim.host->posA2y); cl(sim.host->posB2y);
	cl(sim.host->weightsA1y); cl(sim.host->weightsA2y); cl(sim.host->weightsB2y);
	cl(sim.host->posB1xOld); cl(sim.host->posB2xOld); cl(sim.host->integ);
	cl(sim.host->indsA1y); cl(sim.host->indsA2y); cl(sim.host->indsB2y);
	cl(sim.host->t1grid); cl(sim.host->delta_t_ratio);
	cl(sim.host->QKv); cl(sim.host->QRv); cl(sim.host->dQKv); cl(sim.host->dQRv);
	cl(sim.host->rInt); cl(sim.host->drInt); cl(sim.host->rvec); cl(sim.host->drvec);
	cl(sim.host->SigmaKA1int); cl(sim.host->SigmaRA1int); cl(sim.host->SigmaKB1int); cl(sim.host->SigmaRB1int);
	cl(sim.host->SigmaKA2int); cl(sim.host->SigmaRA2int); cl(sim.host->SigmaKB2int); cl(sim.host->SigmaRB2int);
	cl(sim.host->QKA1int); cl(sim.host->QRA1int); cl(sim.host->QKB1int); cl(sim.host->QRB1int);
	cl(sim.host->QKA2int); cl(sim.host->QRA2int); cl(sim.host->QKB2int); cl(sim.host->QRB2int);
	cl(sim.host->convA1_1); cl(sim.host->convA2_1); cl(sim.host->convA1_2); cl(sim.host->convA2_2);
	cl(sim.host->convR_1); cl(sim.host->convR_2); cl(sim.host->convR_3); cl(sim.host->convR_4);
	cl(sim.host->temp0); cl(sim.host->temp1); cl(sim.host->temp2); cl(sim.host->temp3); cl(sim.host->temp4);
	cl(sim.host->temp5); cl(sim.host->temp6); cl(sim.host->temp7); cl(sim.host->temp8); cl(sim.host->temp9);
	cl(sim.host->Stemp0); cl(sim.host->Stemp1);
	cl(sim.host->error_result);
	std::cout << dmfe::console::INFO() << "Cleared host vectors." << std::endl;
}

void copyParametersToDevice(int p_host, int p2_host, double lambda_host) {
	cudaError_t err = cudaMemcpyToSymbol(d_p, &p_host, sizeof(int));
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("cudaMemcpyToSymbol(d_p) failed: ") + cudaGetErrorString(err));
	}
	err = cudaMemcpyToSymbol(d_p2, &p2_host, sizeof(int));
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("cudaMemcpyToSymbol(d_p2) failed: ") + cudaGetErrorString(err));
	}
	err = cudaMemcpyToSymbol(d_lambda, &lambda_host, sizeof(double));
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("cudaMemcpyToSymbol(d_lambda) failed: ") + cudaGetErrorString(err));
	}
}

double* copyVectorToDeviceRaw(const std::vector<double>& host_vec) {
	double* device_ptr = nullptr;
	size_t bytes = host_vec.size() * sizeof(double);
	cudaMalloc(&device_ptr, bytes);
	cudaMemcpy(device_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice);
	return device_ptr;
}

size_t* copyVectorToDeviceRaw(const std::vector<size_t>& host_vec) {
	size_t* device_ptr = nullptr;
	size_t bytes = host_vec.size() * sizeof(size_t);
	cudaMalloc(&device_ptr, bytes);
	cudaMemcpy(device_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice);
	return device_ptr;
}

