#pragma once
#include "core/config_build.hpp"
#include <cstddef>
#include <vector>

// Host data (mirrors of device data) -----------------------------------
struct HostSimulationData {
    std::vector<double> theta, phi1, phi2, posA1y, posA2y, posB2y, weightsA1y, weightsA2y, weightsB2y, posB1xOld, posB2xOld, integ;
    std::vector<size_t> indsA1y, indsA2y, indsB2y;

    std::vector<double> t1grid, delta_t_ratio;

    std::vector<double> QKv, QRv, dQKv, dQRv, rInt, drInt, rvec, drvec;

    std::vector<double> SigmaKA1int, SigmaRA1int, SigmaKB1int, SigmaRB1int, SigmaKA2int, SigmaRA2int, SigmaKB2int, SigmaRB2int;
    std::vector<double> QKA1int, QRA1int, QKB1int, QRB1int, QKA2int, QRA2int, QKB2int, QRB2int;
    std::vector<double> convA1_1, convA2_1, convA1_2, convA2_2, convR_1, convR_2, convR_3, convR_4;

    std::vector<double> temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

    std::vector<size_t> Stemp0, Stemp1;

    std::vector<double> error_result;

    // Debug/telemetry timelines ------------------------------------------------
    // Captures (simulation time, wall-clock runtime) pairs when config.debug is true.
    std::vector<double> debug_step_times;
    std::vector<double> debug_step_runtimes;
    std::vector<double> debug_step_memory;
};