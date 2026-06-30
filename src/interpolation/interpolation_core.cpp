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
        (same ? posB1xIn : isearchPosSortedInit(sim->host->t1grid, sim->host->theta, posB1xIn)) :
        bsearchPosSorted(sim->host->t1grid, sim->host->theta * sim->host->t1grid.back());

    // Compute posB2x
    vector<double> posB2x = !posB2xIn.empty() ?
        (same ? posB2xIn : isearchPosSortedInit(sim->host->t1grid, sim->host->phi2, posB2xIn)) :
        bsearchPosSorted(sim->host->t1grid, sim->host->phi2 * sim->host->t1grid.back());

    // Update old positions
    sim->host->posB1xOld = posB1x;
    sim->host->posB2xOld = posB2x;

    // Interpolate QKA1int and QRA1int
    if (sim->host->t1grid.back() > 0) {
        indexVecLN3(sim->host->weightsA1y, sim->host->indsA1y, sim->host->QKA1int, sim->host->QRA1int, config.len);
    }
    else {
        sim->host->QKA1int.assign(config.len * config.len, sim->host->QKv[0]);
        sim->host->QRA1int.assign(config.len * config.len, sim->host->QRv[0]);
    }
    SigmaK(sim->host->QKA1int, sim->host->SigmaKA1int);
    SigmaR(sim->host->QKA1int, sim->host->QRA1int, sim->host->SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (sim->host->t1grid.back() > 0) {
        indexVecLN3(sim->host->weightsA2y, sim->host->indsA2y, sim->host->QKA2int, sim->host->QRA2int, config.len);
    }
    else {
        sim->host->QKA2int.assign(config.len * config.len, sim->host->QKv[0]);
        sim->host->QRA2int.assign(config.len * config.len, sim->host->QRv[0]);
    }
    SigmaR(sim->host->QKA2int, sim->host->QRA2int, sim->host->SigmaRA2int);
    
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
    if (sim->host->t1grid.back() > 0) {
        indexVecN(config.len, diff, Floor, sim->host->delta_t_ratio, sim->host->QKB1int, sim->host->QRB1int, config.len);
    }
    else {
        sim->host->QKB1int.assign(config.len * config.len, sim->host->QKv[0]);
        sim->host->QRB1int.assign(config.len * config.len, sim->host->QRv[0]);
    }
    SigmaK(sim->host->QKB1int, sim->host->SigmaKB1int);
    SigmaR(sim->host->QKB1int, sim->host->QRB1int, sim->host->SigmaRB1int);

    // Interpolate QKB2int and QRB2int (indexMatAll handles log-space if enabled)
    if (sim->host->t1grid.back() > 0) {
        indexMatAll(sim->host->posB2xOld, sim->host->indsB2y, sim->host->weightsB2y, sim->host->delta_t_ratio, sim->host->QKB2int, sim->host->QRB2int);
    }
    else {
        sim->host->QKB2int.assign(config.len * config.len, sim->host->QKv[0]);
        sim->host->QRB2int.assign(config.len * config.len, sim->host->QRv[0]);
    }
    SigmaK(sim->host->QKB2int, sim->host->SigmaKB2int);
    SigmaR(sim->host->QKB2int, sim->host->QRB2int, sim->host->SigmaRB2int);

    // Interpolate rInt
    if (sim->host->t1grid.back() > 0) {
        indexVecR2(sim->host->rvec, sim->host->drvec, diff, Floor, sim->host->delta_t_ratio, sim->host->rInt);
    }
    else {
        sim->host->rInt.assign(config.len, sim->host->rvec[0]);
    }
}
