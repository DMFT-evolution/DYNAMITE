#include "search_utils.hpp"
#include "config.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

// External declaration for global config variable
extern SimulationConfig config;

vector<double> bsearchPosSorted(const vector<double>& list, const vector<double>& elem) {
    vector<double> result(elem.size());
    #pragma omp parallel for
    for (size_t l = 0; l < elem.size(); ++l) {
        double El = elem[l];
        size_t n0 = 1; // Start index (Mathematica uses 1-based indexing, C++ uses 0-based)
        size_t n1 = list.size(); // End index
        size_t m = 1; // Midpoint index
        double Lm = 0.0;

        // Binary search
        if (list[m - 1] < El) {
            while (n0 <= n1) {
                m = (n0 + n1) / 2; // Compute midpoint
                Lm = list[m - 1];
                if (Lm == El) {
                    n0 = m;
                    break;
                }
                if (Lm < El) {
                    n0 = m + 1;
                }
                else {
                    n1 = m - 1;
                }
            }
        }

        n1 = list.size(); // Reset n1
        if (Lm <= El) {
            if (m == list.size()) {
                result[l] = m;
            }
            else {
                result[l] = m + (El - Lm) / (list[m] - Lm);
            }
        }
        else {
            if (m > 1) {
                result[l] = m - (El - Lm) / (list[m - 2] - Lm);
            }
            else {
                result[l] = m;
            }
        }
    }

    return move(result);
}

vector<double> isearchPosSortedInit(const vector<double>& list, const vector<double>& elem, const vector<double>& inits) {
    vector<double> result(elem.size(), 0.0); // Initialize output vector

    #pragma omp parallel for
    for (size_t k = 0; k < inits.size()/config.len; ++k)
    {
        size_t length = list.size();
        double l0, l1, Lm, El;
        size_t n0, n1, m;
        bool even;
        double temp=length;
        // Iterate over `elem` in reverse order
        for (size_t l = config.len; l-- > 0;)
        {
            El = list.back() * elem[k*config.len+l];
            n1 = min(static_cast<size_t>(ceil(temp)), length);
            n0 = static_cast<size_t>(floor(inits[k*config.len+l]));
            m = min(n0 + 1, length);
            even = true;

            if ((Lm = list[m - 1]) > El) {
                n1 = max(static_cast<size_t>(m) - 2, (size_t)1);
            }

            // Perform the search
            while ((l0 = list[n0 - 1]) <= El && El <= (l1 = list[n1 - 1]) && n0 < n1) {
                even = !even;
                if (even) {
                    m = n0 + round((El - l0) / (l1 - l0) * (n1 - n0));
                }
                else {
                    m = (n0 + n1) / 2;
                }
                Lm = list[m - 1];
                if (Lm == El) {
                    n0 = m;
                    n1 = m - 1;
                }
                else if (Lm < El) {
                    n0 = m + 1;
                }
                else {
                    n1 = m - 1;
                }
            }

            // Compute the output value
            if (Lm <= El) {
                if (m == length) {
                    temp = m;
                }
                else {
                    temp = m + (El - Lm) / (list[m] - Lm);
                }
            }
            else {
                if (m > 1) {
                    temp = m - (El - Lm) / (list[m - 2] - Lm);
                }
                else {
                    temp = m;
                }
            }
            result[k*config.len+l] = temp;
        }
    }

    return move(result);
}
