// Central extern declarations for global simulation state used across modules.
#pragma once
#include <vector>
#include <string>
#include "simulation_data.hpp"
#include "rk_data.hpp"

extern int p; 
extern int p2; 
extern double lambda; 
extern double TMCT; 
extern double T0; 
extern double Gamma; 
extern int maxLoop; 
extern std::string resultsDir; 
extern std::string outputDir; 
extern bool debug; 
extern bool save_output; 

extern double tmax; 
extern double delta_t_min; 
extern double delta_max; 
extern double rmax[11];

extern double delta; 
extern double delta_old; 
extern int loop; 
extern double specRad; 
extern double delta_t; 
extern size_t len; 
extern int ord; 
extern bool gpu; 

extern SimulationData* sim; 
extern RKData* rk; 
