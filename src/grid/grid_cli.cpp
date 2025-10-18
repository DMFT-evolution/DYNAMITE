#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include "core/console.hpp"

#include "grid/theta_grid.hpp"
#include "grid/phi_grid.hpp"
#include "grid/integration.hpp"
#include "grid/pos_grid.hpp"
#include "grid/grid_io.hpp"
#include "grid/interpolation_weights.hpp"

namespace dmfe {

static void print_grid_usage(const char* prog) {
    std::cout << dmfe::console::INFO() << "Usage: " << prog << " grid [options]\n"
              << "Options:\n"
              << "  -L, --len N                  Grid length (default: 512)\n"
              << "  -M, --Tmax X                 Long-time scale Tmax used for theta mapping (default: 100000)\n"
              << "  -a, --alpha X                Nonlinear index blend alpha in [0,1] (default: 0)\n"
              << "  -D, --delta X                Nonlinear index softness delta >= 0 (default: 0)\n"
              << "  -d, --dir SUBDIR             Output subdirectory under Grid_data/ (default: <len>)\n"
              << "  -V, --validate               Do not write; validate generated theta/phi/int against saved files in SUBDIR\n"
              << "  -s, --spline-order n         Quadrature spline order for int.dat (default: 5)\n"
              << "  -m, --interp-method METHOD   Interp method for metadata: poly | rational | bspline (default: poly)\n"
              << "  -o, --interp-order n         Interpolation order/degree n (default: 9)\n"
              << "  -f, --fh-stencil m           FH window size m (rational only). Default: n+1; must satisfy m >= n+1\n"
              << "\n"
              << "Defaults: Tmax=100000, method=poly, order=9.\n";
}

bool maybe_handle_grid_cli(int argc, char** argv, int& exitCode) {
    if (!(argc >= 2 && std::string(argv[1]) == "grid")) return false;

    std::size_t len = 512;
    double Tmax = 1e5;
    double alpha = 0.0;
    double delta = 0.0;
    std::string subdir;
    bool validate = false;
    int spline_order = 5;
    std::string interp_method = "poly"; // default: local barycentric polynomial
    int interp_order = 9;               // default order for interpolation
    int fh_stencil = -1;                // optional FH window size (>= d+1); default: d+1

    // Capture full command line for provenance
    std::string cmdline;
    {
        std::ostringstream oss;
        for (int i = 0; i < argc; ++i) { if (i) oss << ' '; oss << argv[i]; }
        cmdline = oss.str();
    }

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
    auto read_next = [&](double &dst){ if (i+1 < argc) { dst = std::stod(argv[++i]); return true;} return false; };
        auto read_next_size = [&](std::size_t &dst){ if (i+1 < argc) { dst = static_cast<std::size_t>(std::stoll(argv[++i])); return true;} return false; };
        auto read_next_str = [&](std::string &dst){ if (i+1 < argc) { dst = argv[++i]; return true;} return false; };
        auto read_next_int = [&](int &dst){ if (i+1 < argc) { dst = std::stoi(argv[++i]); return true;} return false; };
    if ((a == "--len" || a == "-L") && !read_next_size(len)) { std::cerr << dmfe::console::ERR() << "Missing value for --len" << std::endl; exitCode = 1; return true; }
    else if ((a == "--Tmax" || a == "-M") && !read_next(Tmax)) { std::cerr << dmfe::console::ERR() << "Missing value for --Tmax" << std::endl; exitCode = 1; return true; }
    else if ((a == "--alpha" || a == "-a") && !read_next(alpha)) { std::cerr << dmfe::console::ERR() << "Missing value for --alpha" << std::endl; exitCode = 1; return true; }
    else if ((a == "--delta" || a == "-D") && !read_next(delta)) { std::cerr << dmfe::console::ERR() << "Missing value for --delta" << std::endl; exitCode = 1; return true; }
    else if ((a == "--dir" || a == "-d") && !read_next_str(subdir)) { std::cerr << dmfe::console::ERR() << "Missing value for --dir" << std::endl; exitCode = 1; return true; }
        else if (a == "--validate" || a == "-V") { validate = true; }
    else if ((a == "--spline-order" || a == "-s") && !read_next_int(spline_order)) { std::cerr << dmfe::console::ERR() << "Missing value for --spline-order" << std::endl; exitCode = 1; return true; }
    else if ((a == "--interp-method" || a == "-m") && !read_next_str(interp_method)) { std::cerr << dmfe::console::ERR() << "Missing value for --interp-method" << std::endl; exitCode = 1; return true; }
    else if ((a == "--interp-order" || a == "-o") && !read_next_int(interp_order)) { std::cerr << dmfe::console::ERR() << "Missing value for --interp-order" << std::endl; exitCode = 1; return true; }
    else if ((a == "--fh-stencil" || a == "-f") && !read_next_int(fh_stencil)) { std::cerr << dmfe::console::ERR() << "Missing value for --fh-stencil" << std::endl; exitCode = 1; return true; }
        else if (a == "-h" || a == "--help") { print_grid_usage(argv[0]); exitCode = 0; return true; }
    }

    if (subdir.empty()) subdir = std::to_string(len);

    std::vector<long double> theta;
    generate_theta_grid(len, Tmax, theta, alpha, delta);

    // Generate phi grids
    std::vector<long double> phi1, phi2;
    generate_phi_grids(theta, phi1, phi2);

    // Integration weights (keep as long double until export)
    std::vector<long double> wint;
    compute_integration_weights(theta, spline_order, wint);

    // Position grids (A1y, A2y, B2y) from theta and phi using interpolation-based inverse
    std::vector<double> posA1y, posA2y, posB2y;
    // Use long double inputs directly (internals operate in long double; outputs remain double)
    generate_pos_grids(len, Tmax, theta, phi1, phi2, posA1y, posA2y, posB2y, alpha, delta);

    // Compute interpolation weights for mapping theta -> phi1 and theta -> phi2 entries (N*N queries each)
    // For each entry (row-major), take query xq = phiX[i*N+j].
    // For local methods, we store start indices and contiguous weights of length m=n+1 per entry.
    std::vector<int> inds;              // A1 indices
    std::vector<double> weights_flat;   // A1 weights (flat)
    std::vector<int> indsA2;            // A2 indices
    std::vector<double> weightsA2_flat; // A2 weights (flat)
    const std::size_t N = len;
    const int n = std::max(1, interp_order);
    // Determine Floater–Hormann window size if needed (bounds: [n+1, N])
    const int mFH = (interp_method == "rational")
        ? (int)std::min<std::size_t>(N, (std::size_t)std::max(n + 1, (fh_stencil > 0 ? fh_stencil : (n + 1))))
        : 0;

    if (interp_method == "poly") {
    auto st = dmfe::grid::compute_barycentric_weights(theta, phi1, n);
        const int m = n + 1;
        inds.resize(N * N);
        weights_flat.resize((std::size_t)N * N * m);
        for (std::size_t k = 0; k < st.size(); ++k) {
            inds[k] = st[k].start;
            for (int j = 0; j < m; ++j) weights_flat[k * m + j] = st[k].alpha[j];
        }
        // A2
    auto st2 = dmfe::grid::compute_barycentric_weights(theta, phi2, n);
        indsA2.resize(N * N);
        const std::size_t msz = (std::size_t)N * N * (std::size_t)m;
        weightsA2_flat.resize(msz);
        for (std::size_t k = 0; k < st2.size(); ++k) {
            indsA2[k] = st2[k].start;
            for (int j = 0; j < m; ++j) weightsA2_flat[k * m + j] = st2[k].alpha[j];
        }
    } else if (interp_method == "rational") {
    auto st = dmfe::grid::compute_barycentric_rational_weights(theta, phi1, n, mFH);
        const int m = mFH;
        inds.resize(N * N);
        weights_flat.resize((std::size_t)N * N * m);
        for (std::size_t k = 0; k < st.size(); ++k) {
            inds[k] = st[k].start;
            for (int j = 0; j < m; ++j) weights_flat[k * m + j] = st[k].alpha[j];
        }
        // A2
    auto st2 = dmfe::grid::compute_barycentric_rational_weights(theta, phi2, n, mFH);
        indsA2.resize(N * N);
        weightsA2_flat.resize((std::size_t)N * N * m);
        for (std::size_t k = 0; k < st2.size(); ++k) {
            indsA2[k] = st2[k].start;
            for (int j = 0; j < m; ++j) weightsA2_flat[k * m + j] = st2[k].alpha[j];
        }
    } else if (interp_method == "bspline") {
        // Global weights per entry: store inds as -1 and write N weights per entry
    auto W = dmfe::grid::compute_bspline_weights(theta, phi1, n);
        inds.assign(N * N, -1);
        weights_flat.resize((std::size_t)N * N * N);
        for (std::size_t k = 0; k < W.size(); ++k) {
            for (std::size_t j = 0; j < N; ++j) weights_flat[k * N + j] = W[k].w[j];
        }
        // A2
    auto W2 = dmfe::grid::compute_bspline_weights(theta, phi2, n);
        indsA2.assign(N * N, -1);
        weightsA2_flat.resize((std::size_t)N * N * N);
        for (std::size_t k = 0; k < W2.size(); ++k) {
            for (std::size_t j = 0; j < N; ++j) weightsA2_flat[k * N + j] = W2[k].w[j];
        }
    } else {
    std::cerr << dmfe::console::ERR() << "Unknown --interp-method: " << interp_method << std::endl;
        exitCode = 1; return true;
    }

    if (!validate) {
        // Unified writer for all grids
    GridPaths paths = write_all_grids(theta, phi1, phi2, wint, posA1y, posA2y, posB2y, len, subdir);
        // Save generation parameters for provenance
    write_grid_generation_params(len, Tmax, spline_order, interp_method, n, mFH, subdir, cmdline, alpha, delta);
        // Write interpolation metadata for A1 and A2
        auto ip = write_A1_interp_metadata(inds, weights_flat, len, subdir);
        auto ip2 = write_A2_interp_metadata(indsA2, weightsA2_flat, len, subdir);
        // Compute and write interpolation metadata for B2: xqB2 = theta[i] / (phi2[i,j] - tiny)
        const double tiny = 1e-200;
        std::vector<long double> xqB2; xqB2.reserve(N*N);
        for (std::size_t i = 0; i < N; ++i) {
            const long double ti = theta[i];
            for (std::size_t j = 0; j < N; ++j) {
                const long double denom = phi2[i * N + j] - tiny;
                long double arg;
                if (std::fabs(denom) < 1e-300L) {
                    arg = (ti > 0) ? theta.back() : theta.front();
                } else {
                    arg = ti / denom;
                }
                xqB2.push_back(arg);
            }
        }
        std::vector<int> indsB2;
        std::vector<double> weightsB2_flat;
        if (interp_method == "poly") {
            auto stB2 = dmfe::grid::compute_barycentric_weights(theta, xqB2, n);
            const int m = n + 1;
            indsB2.resize(N * N);
            weightsB2_flat.resize((std::size_t)N * N * m);
            for (std::size_t k = 0; k < stB2.size(); ++k) {
                indsB2[k] = stB2[k].start;
                for (int j = 0; j < m; ++j) weightsB2_flat[k * m + j] = stB2[k].alpha[j];
            }
        } else if (interp_method == "rational") {
            auto stB2 = dmfe::grid::compute_barycentric_rational_weights(theta, xqB2, n, mFH);
            const int m = mFH;
            indsB2.resize(N * N);
            weightsB2_flat.resize((std::size_t)N * N * m);
            for (std::size_t k = 0; k < stB2.size(); ++k) {
                indsB2[k] = stB2[k].start;
                for (int j = 0; j < m; ++j) weightsB2_flat[k * m + j] = stB2[k].alpha[j];
            }
        } else if (interp_method == "bspline") {
            auto WB2 = dmfe::grid::compute_bspline_weights(theta, xqB2, n);
            indsB2.assign(N * N, -1);
            weightsB2_flat.resize((std::size_t)N * N * N);
            for (std::size_t k = 0; k < WB2.size(); ++k) {
                for (std::size_t j = 0; j < N; ++j) weightsB2_flat[k * N + j] = WB2[k].w[j];
            }
        }
    auto ip3 = write_B2_interp_metadata(indsB2, weightsB2_flat, len, subdir);
    std::cout << dmfe::console::DONE() << "Generated grids (len=" << len << ", Tmax=" << Tmax << ") ->\n"
                  << "  " << paths.theta
                  << "\n  " << paths.phi1
                  << "\n  " << paths.phi2
                  << "\n  " << paths.wint
                  << "\n  " << paths.posA1y
                  << "\n  " << paths.posA2y
                  << "\n  " << paths.posB2y
          << "\n  " << ip.first
          << "\n  " << ip.second
          << "\n  " << ip2.first
                  << "\n  " << ip2.second
                  << "\n  " << ip3.first
                  << "\n  " << ip3.second
                  << std::endl;
        exitCode = 0;
        return true;
    }

    // Validation flow
    double maxAbsDiff = 0.0; std::size_t mismatches = 0;
    bool ok = validate_against_saved(theta, phi1, phi2, len, subdir, 5e-16, maxAbsDiff, mismatches);
    if (!ok) {
    std::cerr << dmfe::console::ERR() << "Validation FAILED (theta/phi) or reference files missing in Grid_data/" << subdir
                  << ". Max abs diff=" << maxAbsDiff << ", mismatches=" << mismatches << std::endl;
        exitCode = 2; return true;
    }
    double maxAbsDiffW = 0.0; std::size_t mismW = 0;
    const double tolInt = 1e-7;
    // Validator expects double inputs; down-convert a view of wint
    std::vector<double> wint_d(wint.size());
    for (std::size_t i = 0; i < wint.size(); ++i) wint_d[i] = static_cast<double>(wint[i]);
    if (!validate_integration_weights(wint_d, len, subdir, tolInt, maxAbsDiffW, mismW)) {
    std::cerr << dmfe::console::ERR() << "Validation FAILED (int.dat) or reference file missing in Grid_data/" << subdir
                  << ". Max abs diff=" << maxAbsDiffW << ", mismatches=" << mismW << std::endl;
        exitCode = 2; return true;
    }
    std::cout << dmfe::console::DONE() << "Validation OK: theta/phi/int match saved outputs (max abs diffs: phi/theta="
              << maxAbsDiff << ", int=" << maxAbsDiffW << ") with int tol=" << tolInt << std::endl;

    exitCode = 0;
    return true;
}

} // namespace dmfe
