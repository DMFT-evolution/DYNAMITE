#include "core/gpu_memory_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/device_constants.hpp"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <iostream>

// d_p, d_p2, d_lambda declared in device_constants.hpp and defined in device_constants.cu

void copyVectorsToGPU(SimulationData& sim, size_t len) {
	// Scalars / simple lists
	sim.d_theta       = sim.h_theta;
	sim.d_phi1        = sim.h_phi1;
	sim.d_phi2        = sim.h_phi2;
	sim.d_posA1y      = sim.h_posA1y;
	sim.d_posA2y      = sim.h_posA2y;
	sim.d_posB2y      = sim.h_posB2y;
	sim.d_weightsA1y  = sim.h_weightsA1y;
	sim.d_weightsA2y  = sim.h_weightsA2y;
	sim.d_weightsB2y  = sim.h_weightsB2y;
	sim.d_posB1xOld   = sim.h_posB1xOld;
	sim.d_posB2xOld   = sim.h_posB2xOld;
	sim.d_integ       = sim.h_integ;

	sim.d_indsA1y     = sim.h_indsA1y;
	sim.d_indsA2y     = sim.h_indsA2y;
	sim.d_indsB2y     = sim.h_indsB2y;

	sim.d_t1grid        = sim.h_t1grid;
	sim.d_delta_t_ratio = sim.h_delta_t_ratio;

	sim.d_QKv   = sim.h_QKv;
	sim.d_QRv   = sim.h_QRv;
	sim.d_dQKv  = sim.h_dQKv;
	sim.d_dQRv  = sim.h_dQRv;
	sim.d_rInt  = sim.h_rInt;
	sim.d_drInt = sim.h_drInt;
	sim.d_rvec  = sim.h_rvec;
	sim.d_drvec = sim.h_drvec;

	sim.d_SigmaKA1int = sim.h_SigmaKA1int;
	sim.d_SigmaRA1int = sim.h_SigmaRA1int;
	sim.d_SigmaKB1int = sim.h_SigmaKB1int;
	sim.d_SigmaRB1int = sim.h_SigmaRB1int;
	sim.d_SigmaKA2int = sim.h_SigmaKA2int;
	sim.d_SigmaRA2int = sim.h_SigmaRA2int;
	sim.d_SigmaKB2int = sim.h_SigmaKB2int;
	sim.d_SigmaRB2int = sim.h_SigmaRB2int;

	sim.d_QKA1int = sim.h_QKA1int;
	sim.d_QRA1int = sim.h_QRA1int;
	sim.d_QKB1int = sim.h_QKB1int;
	sim.d_QRB1int = sim.h_QRB1int;
	sim.d_QKA2int = sim.h_QKA2int;
	sim.d_QRA2int = sim.h_QRA2int;
	sim.d_QKB2int = sim.h_QKB2int;
	sim.d_QRB2int = sim.h_QRB2int;

	// Workspace allocate (no host mirrors needed except optional debug)
	sim.convA1_1.resize(len);
	sim.convA2_1.resize(len);
	sim.convA1_2.resize(len);
	sim.convA2_2.resize(len);
	sim.convR_1.resize(len);
	sim.convR_2.resize(len);
	sim.convR_3.resize(len);
	sim.convR_4.resize(len);

	sim.temp0.resize(len);
	sim.temp1.resize(len);
	sim.temp2.resize(len);
	sim.temp3.resize(len);
	sim.temp4.resize(len);
	sim.temp5.resize(len);
	sim.temp6.resize(len);
	sim.temp7.resize(len);
	sim.temp8.resize(len);
	sim.temp9.resize(len);

	sim.Stemp0.resize(len);
	sim.Stemp1.resize(len);

	sim.error_result.resize(1, 0.0);

	std::cout << "Host -> Device vector copy complete." << std::endl;
}

void copyVectorsToCPU(SimulationData& sim) {
	auto copyBack = [](auto& host, const auto& dev){ host.resize(dev.size()); thrust::copy(dev.begin(), dev.end(), host.begin()); };

	copyBack(sim.h_QKv, sim.d_QKv);
	copyBack(sim.h_QRv, sim.d_QRv);
	copyBack(sim.h_dQKv, sim.d_dQKv);
	copyBack(sim.h_dQRv, sim.d_dQRv);
	copyBack(sim.h_rvec, sim.d_rvec);
	copyBack(sim.h_drvec, sim.d_drvec);
	copyBack(sim.h_rInt, sim.d_rInt);
	copyBack(sim.h_drInt, sim.d_drInt);
	copyBack(sim.h_t1grid, sim.d_t1grid);
	copyBack(sim.h_delta_t_ratio, sim.d_delta_t_ratio);

	copyBack(sim.h_SigmaKA1int, sim.d_SigmaKA1int);
	copyBack(sim.h_SigmaRA1int, sim.d_SigmaRA1int);
	copyBack(sim.h_SigmaKB1int, sim.d_SigmaKB1int);
	copyBack(sim.h_SigmaRB1int, sim.d_SigmaRB1int);
	copyBack(sim.h_SigmaKA2int, sim.d_SigmaKA2int);
	copyBack(sim.h_SigmaRA2int, sim.d_SigmaRA2int);
	copyBack(sim.h_SigmaKB2int, sim.d_SigmaKB2int);
	copyBack(sim.h_SigmaRB2int, sim.d_SigmaRB2int);

	copyBack(sim.h_QKA1int, sim.d_QKA1int);
	copyBack(sim.h_QRA1int, sim.d_QRA1int);
	copyBack(sim.h_QKB1int, sim.d_QKB1int);
	copyBack(sim.h_QRB1int, sim.d_QRB1int);
	copyBack(sim.h_QKA2int, sim.d_QKA2int);
	copyBack(sim.h_QRA2int, sim.d_QRA2int);
	copyBack(sim.h_QKB2int, sim.d_QKB2int);
	copyBack(sim.h_QRB2int, sim.d_QRB2int);

	copyBack(sim.h_posA1y, sim.d_posA1y);
	copyBack(sim.h_posA2y, sim.d_posA2y);
	copyBack(sim.h_posB2y, sim.d_posB2y);
	copyBack(sim.h_posB1xOld, sim.d_posB1xOld);
	copyBack(sim.h_posB2xOld, sim.d_posB2xOld);

	// Optional: copy theta/phi only if needed
	copyBack(sim.h_theta, sim.d_theta);
	copyBack(sim.h_phi1, sim.d_phi1);
	copyBack(sim.h_phi2, sim.d_phi2);

	copyBack(sim.h_weightsA1y, sim.d_weightsA1y);
	copyBack(sim.h_weightsA2y, sim.d_weightsA2y);
	copyBack(sim.h_weightsB2y, sim.d_weightsB2y);
	copyBack(sim.h_integ, sim.d_integ);

	copyBack(sim.h_indsA1y, sim.d_indsA1y);
	copyBack(sim.h_indsA2y, sim.d_indsA2y);
	copyBack(sim.h_indsB2y, sim.d_indsB2y);

	std::cout << "Device -> Host vector copy complete." << std::endl;
}

void clearAllDeviceVectors(SimulationData& sim) {
	auto cl = [](auto& v){ v.clear(); v.shrink_to_fit(); };
	cl(sim.d_QKv); cl(sim.d_QRv); cl(sim.d_dQKv); cl(sim.d_dQRv);
	cl(sim.d_rInt); cl(sim.d_drInt); cl(sim.d_rvec); cl(sim.d_drvec);
	cl(sim.d_SigmaKA1int); cl(sim.d_SigmaRA1int); cl(sim.d_SigmaKB1int); cl(sim.d_SigmaRB1int);
	cl(sim.d_SigmaKA2int); cl(sim.d_SigmaRA2int); cl(sim.d_SigmaKB2int); cl(sim.d_SigmaRB2int);
	cl(sim.d_QKA1int); cl(sim.d_QRA1int); cl(sim.d_QKB1int); cl(sim.d_QRB1int);
	cl(sim.d_QKA2int); cl(sim.d_QRA2int); cl(sim.d_QKB2int); cl(sim.d_QRB2int);
	cl(sim.d_theta); cl(sim.d_phi1); cl(sim.d_phi2);
	cl(sim.d_posA1y); cl(sim.d_posA2y); cl(sim.d_posB2y);
	cl(sim.d_weightsA1y); cl(sim.d_weightsA2y); cl(sim.d_weightsB2y);
	cl(sim.d_posB1xOld); cl(sim.d_posB2xOld);
	cl(sim.d_indsA1y); cl(sim.d_indsA2y); cl(sim.d_indsB2y);
	cl(sim.d_integ); cl(sim.d_t1grid); cl(sim.d_delta_t_ratio);
	cl(sim.convA1_1); cl(sim.convA2_1); cl(sim.convA1_2); cl(sim.convA2_2);
	cl(sim.convR_1); cl(sim.convR_2); cl(sim.convR_3); cl(sim.convR_4);
	cl(sim.temp0); cl(sim.temp1); cl(sim.temp2); cl(sim.temp3); cl(sim.temp4);
	cl(sim.temp5); cl(sim.temp6); cl(sim.temp7); cl(sim.temp8); cl(sim.temp9);
	cl(sim.Stemp0); cl(sim.Stemp1);
	cl(sim.error_result);
	std::cout << "Cleared device vectors." << std::endl;
}

void clearAllHostVectors(SimulationData& sim) {
	auto cl = [](auto& v){ v.clear(); v.shrink_to_fit(); };
	cl(sim.h_theta); cl(sim.h_phi1); cl(sim.h_phi2);
	cl(sim.h_posA1y); cl(sim.h_posA2y); cl(sim.h_posB2y);
	cl(sim.h_weightsA1y); cl(sim.h_weightsA2y); cl(sim.h_weightsB2y);
	cl(sim.h_posB1xOld); cl(sim.h_posB2xOld); cl(sim.h_integ);
	cl(sim.h_indsA1y); cl(sim.h_indsA2y); cl(sim.h_indsB2y);
	cl(sim.h_t1grid); cl(sim.h_delta_t_ratio);
	cl(sim.h_QKv); cl(sim.h_QRv); cl(sim.h_dQKv); cl(sim.h_dQRv);
	cl(sim.h_rInt); cl(sim.h_drInt); cl(sim.h_rvec); cl(sim.h_drvec);
	cl(sim.h_SigmaKA1int); cl(sim.h_SigmaRA1int); cl(sim.h_SigmaKB1int); cl(sim.h_SigmaRB1int);
	cl(sim.h_SigmaKA2int); cl(sim.h_SigmaRA2int); cl(sim.h_SigmaKB2int); cl(sim.h_SigmaRB2int);
	cl(sim.h_QKA1int); cl(sim.h_QRA1int); cl(sim.h_QKB1int); cl(sim.h_QRB1int);
	cl(sim.h_QKA2int); cl(sim.h_QRA2int); cl(sim.h_QKB2int); cl(sim.h_QRB2int);
	cl(sim.h_convA1_1); cl(sim.h_convA2_1); cl(sim.h_convA1_2); cl(sim.h_convA2_2);
	cl(sim.h_convR_1); cl(sim.h_convR_2); cl(sim.h_convR_3); cl(sim.h_convR_4);
	cl(sim.h_temp0); cl(sim.h_temp1); cl(sim.h_temp2); cl(sim.h_temp3); cl(sim.h_temp4);
	cl(sim.h_temp5); cl(sim.h_temp6); cl(sim.h_temp7); cl(sim.h_temp8); cl(sim.h_temp9);
	cl(sim.h_Stemp0); cl(sim.h_Stemp1);
	cl(sim.h_error_result);
	std::cout << "Cleared host vectors." << std::endl;
}

void copyParametersToDevice(int p_host, int p2_host, double lambda_host) {
	cudaMemcpyToSymbol(d_p, &p_host, sizeof(int));
	cudaMemcpyToSymbol(d_p2, &p2_host, sizeof(int));
	cudaMemcpyToSymbol(d_lambda, &lambda_host, sizeof(double));
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

