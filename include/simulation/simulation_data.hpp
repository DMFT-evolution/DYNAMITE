#pragma once
#include "core/config_build.hpp"
#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#endif
#include <cstddef>
#include <vector>

// Holds both host (h_*) and device (d_*) simulation state vectors.
// Host vectors were previously global; adding them here centralizes ownership
// and paves the way to remove global variables from translation units.
struct SimulationData {
#if DMFE_WITH_CUDA
    // Device data -----------------------------------------------------------
    thrust::device_vector<double> d_theta, d_phi1, d_phi2, d_posA1y, d_posA2y, d_posB2y, d_weightsA1y, d_weightsA2y, d_weightsB2y, d_posB1xOld, d_posB2xOld, d_integ;
    thrust::device_vector<size_t> d_indsA1y, d_indsA2y, d_indsB2y;

    thrust::device_vector<double> d_t1grid, d_delta_t_ratio;

    thrust::device_vector<double> d_QKv, d_QRv, d_dQKv, d_dQRv, d_rInt, d_drInt, d_rvec, d_drvec;

    thrust::device_vector<double> d_SigmaKA1int, d_SigmaRA1int, d_SigmaKB1int, d_SigmaRB1int, d_SigmaKA2int, d_SigmaRA2int, d_SigmaKB2int, d_SigmaRB2int;
    thrust::device_vector<double> d_QKA1int, d_QRA1int, d_QKB1int, d_QRB1int, d_QKA2int, d_QRA2int, d_QKB2int, d_QRB2int;
    thrust::device_vector<double> convA1_1, convA2_1, convA1_2, convA2_2, convR_1, convR_2, convR_3, convR_4;

    thrust::device_vector<double> temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12;

    thrust::device_vector<size_t> Stemp0, Stemp1, Stemp2;

    thrust::device_vector<double> error_result;
#endif

    // Host data (mirrors of device data) -----------------------------------
    std::vector<double> h_theta, h_phi1, h_phi2, h_posA1y, h_posA2y, h_posB2y, h_weightsA1y, h_weightsA2y, h_weightsB2y, h_posB1xOld, h_posB2xOld, h_integ;
    std::vector<size_t> h_indsA1y, h_indsA2y, h_indsB2y;

    std::vector<double> h_t1grid, h_delta_t_ratio;

    std::vector<double> h_QKv, h_QRv, h_dQKv, h_dQRv, h_rInt, h_drInt, h_rvec, h_drvec;

    std::vector<double> h_SigmaKA1int, h_SigmaRA1int, h_SigmaKB1int, h_SigmaRB1int, h_SigmaKA2int, h_SigmaRA2int, h_SigmaKB2int, h_SigmaRB2int;
    std::vector<double> h_QKA1int, h_QRA1int, h_QKB1int, h_QRB1int, h_QKA2int, h_QRA2int, h_QKB2int, h_QRB2int;
    std::vector<double> h_convA1_1, h_convA2_1, h_convA1_2, h_convA2_2, h_convR_1, h_convR_2, h_convR_3, h_convR_4;

    std::vector<double> h_temp0, h_temp1, h_temp2, h_temp3, h_temp4, h_temp5, h_temp6, h_temp7, h_temp8, h_temp9;

    std::vector<size_t> h_Stemp0, h_Stemp1;

    std::vector<double> h_error_result;

    // Debug/telemetry timelines ------------------------------------------------
    // Captures (simulation time, wall-clock runtime) pairs when config.debug is true.
    std::vector<double> h_debug_step_times;
    std::vector<double> h_debug_step_runtimes;
};
