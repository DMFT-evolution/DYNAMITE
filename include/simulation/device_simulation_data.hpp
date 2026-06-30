#pragma once
#include "core/device_vector.hpp"


// Host vectors were previously global; adding them here centralizes ownership
// and paves the way to remove global variables from translation units.
// Device data -----------------------------------------------------------
struct DeviceSimulationData {
    dmfe::device_vector<double> theta, phi1, phi2, posA1y, posA2y, posB2y, weightsA1y, weightsA2y, weightsB2y, posB1xOld, posB2xOld, integ;
    dmfe::device_vector<size_t> indsA1y, indsA2y, indsB2y;

    dmfe::device_vector<double> t1grid, delta_t_ratio;

    dmfe::device_vector<double> QKv, QRv, dQKv, dQRv, rInt, drInt, rvec, drvec;

    dmfe::device_vector<double> SigmaKA1int, SigmaRA1int, SigmaKB1int, SigmaRB1int, SigmaKA2int, SigmaRA2int, SigmaKB2int, SigmaRB2int;
    dmfe::device_vector<double> QKA1int, QRA1int, QKB1int, QRB1int, QKA2int, QRA2int, QKB2int, QRB2int;
    dmfe::device_vector<double> convA1_1, convA2_1, convA1_2, convA2_2, convR_1, convR_2, convR_3, convR_4;

    mutable dmfe::device_vector<double> temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12;

    mutable dmfe::device_vector<size_t> Stemp0, Stemp1, Stemp2;

    dmfe::device_vector<double> error_result;
}; 
