#include "grid/phi_grid.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include "grid/grid_io.hpp"

namespace {
// writing moved to grid_io.cpp; readers are provided by grid_io.hpp
} // namespace

void generate_phi_grids(const std::vector<double>& theta,
						std::vector<double>& phi1,
						std::vector<double>& phi2) {
	const std::size_t N = theta.size();
	phi1.assign(N * N, 0.0);
	phi2.assign(N * N, 0.0);
	for (std::size_t i = 0; i < N; ++i) {
		const double ti = theta[i];
		const double one_minus_ti = 1.0 - ti;
		double* row1 = &phi1[i * N];
		double* row2 = &phi2[i * N];
		for (std::size_t j = 0; j < N; ++j) {
			const double tj = theta[j];
			row1[j] = ti * tj;                      // KroneckerProduct(theta, theta)
			row2[j] = ti + one_minus_ti * tj;       // theta + (1 - theta) * theta
		}
	}
}

// write_phi_grids moved to grid_io.cpp

bool validate_against_saved(const std::vector<double>& theta,
							const std::vector<double>& phi1,
							const std::vector<double>& phi2,
							std::size_t len,
							const std::string& subdir,
							double tol,
							double& maxAbsDiff,
							std::size_t& mismatches) {
	maxAbsDiff = 0.0;
	mismatches = 0;
	std::string base = "Grid_data/" + subdir;

	// Load reference theta
	std::vector<double> ref_theta;
	if (!read_vector_file(base + "/theta.dat", ref_theta)) return false;
	if (ref_theta.size() != len) return false;

	// Load reference phi1/phi2
	std::vector<double> ref_phi1, ref_phi2;
	if (!read_matrix_tsv(base + "/phi1.dat", ref_phi1, len)) return false;
	if (!read_matrix_tsv(base + "/phi2.dat", ref_phi2, len)) return false;

	auto update = [&](double a, double b) {
		double d = std::abs(a - b);
		if (d > maxAbsDiff) maxAbsDiff = d;
		if (d > tol) ++mismatches;
	};

	for (std::size_t i = 0; i < len; ++i) update(theta[i], ref_theta[i]);
	for (std::size_t k = 0; k < phi1.size(); ++k) update(phi1[k], ref_phi1[k]);
	for (std::size_t k = 0; k < phi2.size(); ++k) update(phi2[k], ref_phi2[k]);

	return mismatches == 0;
}
