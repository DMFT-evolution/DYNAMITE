#include "interpolation/interpolation_core.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include "search/search_utils.hpp"
#include "interpolation/index_vec.hpp"
#include "interpolation/index_mat.hpp"
#include "math/math_sigma.hpp"
#include "core/vector_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

extern SimulationConfig config;
extern SimulationData* sim;

// CPU implementation for interpolate (extracted from interpolation_core.cu)
void interpolate(const vector<double>& posB1xIn, const vector<double>& posB2xIn, const bool same)
{
    // Compute posB1x
    vector<double> posB1x = !posB1xIn.empty() ?
        (same ? posB1xIn : isearchPosSortedInit(sim->h_t1grid, sim->h_theta, posB1xIn)) :
        bsearchPosSorted(sim->h_t1grid, sim->h_theta * sim->h_t1grid.back());

    // Compute posB2x
    vector<double> posB2x = !posB2xIn.empty() ?
        (same ? posB2xIn : isearchPosSortedInit(sim->h_t1grid, sim->h_phi2, posB2xIn)) :
        bsearchPosSorted(sim->h_t1grid, sim->h_phi2 * sim->h_t1grid.back());

    // Update old positions
    sim->h_posB1xOld = posB1x;
    sim->h_posB2xOld = posB2x;

    // Interpolate QKA1int and QRA1int
    if (sim->h_t1grid.back() > 0) {
        indexVecLN3(sim->h_weightsA1y, sim->h_indsA1y, sim->h_QKA1int, sim->h_QRA1int, config.len);
    }
    else {
        sim->h_QKA1int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRA1int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKA1int, sim->h_SigmaKA1int);
    SigmaR(sim->h_QKA1int, sim->h_QRA1int, sim->h_SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (sim->h_t1grid.back() > 0) {
        indexVecLN3(sim->h_weightsA2y, sim->h_indsA2y, sim->h_QKA2int, sim->h_QRA2int, config.len);
    }
    else {
        sim->h_QKA2int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRA2int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaR(sim->h_QKA2int, sim->h_QRA2int, sim->h_SigmaRA2int);

    // Interpolate QKB1int and QRB1int (indexVecN handles log-space if enabled)
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
    diff = vector<double>(Floor.begin(), Floor.end()) - posB1x;

    if (sim->h_t1grid.back() > 0) {
        indexVecN(config.len, diff, Floor, sim->h_delta_t_ratio, sim->h_QKB1int, sim->h_QRB1int, config.len);
    }
    else {
        sim->h_QKB1int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRB1int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKB1int, sim->h_SigmaKB1int);
    SigmaR(sim->h_QKB1int, sim->h_QRB1int, sim->h_SigmaRB1int);

    // Interpolate QKB2int and QRB2int (indexMatAll handles log-space if enabled)
    if (sim->h_t1grid.back() > 0) {
        indexMatAll(sim->h_posB2xOld, sim->h_indsB2y, sim->h_weightsB2y, sim->h_delta_t_ratio, sim->h_QKB2int, sim->h_QRB2int);
    }
    else {
        sim->h_QKB2int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRB2int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKB2int, sim->h_SigmaKB2int);
    SigmaR(sim->h_QKB2int, sim->h_QRB2int, sim->h_SigmaRB2int);

    // Interpolate rInt
    if (sim->h_t1grid.back() > 0) {
        indexVecR2(sim->h_rvec, sim->h_drvec, diff, Floor, sim->h_delta_t_ratio, sim->h_rInt);
    }
    else {
        sim->h_rInt.assign(config.len, sim->h_rvec[0]);
    }
}
