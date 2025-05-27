// Minimal C++ program to test if Visual Studio Code is set up correctly

#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <string>
#include <numeric>
#include <iomanip>
#include <vector>
#include <chrono>
#include <algorithm>


using namespace std;
using namespace std::chrono;

template <int P>
constexpr double pow_const(double q) {
    return q * pow_const<P - 1>(q);
}

template <>
constexpr double pow_const<0>(double) {
    return 1.0;
}

constexpr int p = 3;
constexpr int p2 = 4;
constexpr double lambda = 0.5;
constexpr double TMCT = 0.805166;
constexpr double T0 = 1.001 * TMCT;
// double T0=1e50;
constexpr double Gamma = 0.0;
constexpr int maxLoop = 100;

constexpr double tmax = 1e6; //time to evolve to
constexpr double delta_t_min = 1e-5; //initial and minimal time step
constexpr double delta_max = 1e-11; //maximal error per step
constexpr double rmax = 13; // stability range of SSPRK(10,4)

double delta;
double delta_old;
int loop;
double specRad;
double delta_t;
size_t len;
int ord;

vector<double> theta, phi1, phi2, posA1y, posA2y, posB2y, weightsA1y, weightsA2y, weightsB2y, posB1xOld, posB2xOld, integ;
vector<size_t> indsA1y, indsA2y, indsB2y;

vector<double> t1grid, delta_t_ratio;

vector<double> QKv, QRv, dQKv, dQRv, rInt, drInt, rvec, drvec;

vector<double> SigmaKA1int, SigmaRA1int, SigmaKB1int, SigmaRB1int, SigmaKA2int, SigmaRA2int, SigmaKB2int, SigmaRB2int;
vector<double> QKA1int, QRA1int, QKB1int, QRB1int, QKA2int, QRA2int, QKB2int, QRB2int;

// Overload the + operator for std::vector<double>
vector<double> operator+(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for addition.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::plus<>());

    return result;
}

vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for addition.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::minus<>());

    return result;
}

// Overload the * operator for the element-wise product of two vectors
vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for element-wise multiplication.");
    }

    vector<double> result(vec1.size());

    // Use std::transform to perform element-wise multiplication
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::multiplies<>());

    return result;
}

// Overload the * operator for the product of a vector and a scalar
vector<double> operator*(const vector<double>& vec, double scalar) {
    vector<double> result(vec.size());

    // Use std::transform to multiply each element by the scalar
    std::transform(vec.begin(), vec.end(), result.begin(), [scalar](double val) { return val * scalar; });

    return result;
}

// Overload the * operator for the product of a scalar and a vector (commutative)
vector<double> operator*(double scalar, const vector<double>& vec) {
    return vec * scalar; // Reuse the previous operator
}

void Product(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
}

void Sum(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}

void Subtract(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
}

void scaleVec(const vector<double>& vec1, double real, vector<double>& result) {
#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * real;
    }
}

vector<double> getLastLenEntries(const vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw invalid_argument("len is greater than the size of the vector.");
    }
    return vector<double>(vec.end() - len, vec.end());
}

vector<double> importVectorFromFile(const string& filename) {
    vector<double> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    double value;
    while (file >> value) {
        data.push_back(value);
        // Skip tabs or newlines automatically handled by `>>`
    }

    file.close();
    return data;
}

vector<size_t> importIntVectorFromFile(const string& filename) {
    vector<size_t> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    size_t value;
    while (file >> value) {
        data.push_back(value);
        // Skip tabs or newlines automatically handled by `>>`
    }

    file.close();
    return data;
}

void import()
{
    theta = importVectorFromFile("Grid_data/theta.dat");
    phi1 = importVectorFromFile("Grid_data/phi1.dat");
    phi2 = importVectorFromFile("Grid_data/phi2.dat");
    posA1y = importVectorFromFile("Grid_data/posA1y.dat");
    posA2y = importVectorFromFile("Grid_data/posA2y.dat");
    posB2y = importVectorFromFile("Grid_data/posB2y.dat");
    indsA1y = importIntVectorFromFile("Grid_data/indsA1y.dat");
    indsA2y = importIntVectorFromFile("Grid_data/indsA2y.dat");
    indsB2y = importIntVectorFromFile("Grid_data/indsB2y.dat");
    weightsA1y = importVectorFromFile("Grid_data/weightsA1y.dat");
    weightsA2y = importVectorFromFile("Grid_data/weightsA2y.dat");
    weightsB2y = importVectorFromFile("Grid_data/weightsB2y.dat");
    integ = importVectorFromFile("Grid_data/int.dat");
    len = theta.size();
    ord = weightsB2y.size() / (len * len) - 2;
}

double flambda(const double q)
{
    return lambda * pow_const<p>(q) + (1 - lambda) * pow_const<p2>(q);
}

double Dflambda(const double q)
{
    return lambda * p * pow_const<p - 1>(q) + (1 - lambda) * p2 * pow_const<p2 - 1>(q);
}

double DDflambda(const double q)
{
    return lambda * p * (p - 1) * pow_const<p - 2>(q) + (1 - lambda) * p2 * (p2 - 1) * pow_const<p2 - 2>(q);
}

double DDDflambda(const double q)
{
    return lambda * p * (p - 1) * (p - 2) * pow_const<p - 3>(q) + (1 - lambda) * p2 * (p2 - 1) * (p2 - 2) * pow_const<p2 - 3>(q);
}

vector<double> indexVecLN3(const int num, const vector<double>& weights, const vector<size_t>& inds)
{
    size_t prod = inds.size();
    size_t length = QKv.size() - len;
    size_t depth = weights.size() / prod;
    vector<double> result(len * len, 0.0);

    switch (num)
    {
    case 1:
#pragma omp parallel for
        for (size_t j = 0; j < prod; j++)
        {
            result[j] = inner_product(weights.begin() + depth * j, weights.begin() + depth * (j + 1), QKv.begin() + length + inds[j], 0.0);
        }
        break;
    case 2:
#pragma omp parallel for
        for (size_t j = 0; j < prod; j++)
        {
            result[j] = inner_product(weights.begin() + depth * j, weights.begin() + depth * (j + 1), QRv.begin() + length + inds[j], 0.0);
        }
        break;
    case 3:
#pragma omp parallel for
        for (size_t j = 0; j < prod; j++)
        {
            result[j] = inner_product(weights.begin() + depth * j, weights.begin() + depth * (j + 1), dQKv.begin() + length + inds[j], 0.0);
        }
        break;
    case 4:
#pragma omp parallel for
        for (size_t j = 0; j < prod; j++)
        {
            result[j] = inner_product(weights.begin() + depth * j, weights.begin() + depth * (j + 1), dQRv.begin() + length + inds[j], 0.0);
        }
        break;
    }

    return result;
}

vector<double> indexVecN(const int num, const size_t length, const vector<double>& weights, const vector<size_t>& inds, const vector<double>& dtratio)
{
    size_t dims[] = { len,len };
    size_t t1len = dtratio.size();
    vector<double> result(len * len, 0.0);

    switch (num)
    {
    case 1:
#pragma omp parallel for
        for (size_t i = 0; i < dims[0]; i++)
        {
            double in3 = weights[i] * weights[i];
            double in4 = in3 * weights[i];
            if (inds[i] < t1len - 1)
            {
                for (size_t j = 0; j < dims[1]; j++)
                {
                    result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * QKv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * QKv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * dQKv[inds[i] * dims[1] + j] - (in3 + in4) * dQKv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
                }
            }
            else
            {
                for (size_t j = 0; j < dims[1]; j++)
                {
                    result[j + dims[1] * i] = (1 - in3) * QKv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * dQKv[inds[i] * dims[1] + j] + in3 * QKv[inds[i] * dims[1] + j];
                }
            }
        }
        break;

    case 2:
#pragma omp parallel for
        for (size_t i = 0; i < dims[0]; i++)
        {
            double in3 = weights[i] * weights[i];
            double in4 = in3 * weights[i];
            if (inds[i] < t1len - 1)
            {
                for (size_t j = 0; j < dims[1]; j++)
                {
                    result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * QRv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * QRv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * dQRv[inds[i] * dims[1] + j] - (in3 + in4) * dQRv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
                }
            }
            else
            {
                for (size_t j = 0; j < dims[1]; j++)
                {
                    result[j + dims[1] * i] = (1 - in3) * QRv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * dQRv[inds[i] * dims[1] + j] + in3 * QRv[inds[i] * dims[1] + j];
                }
            }
        }
        break;
    }

    return result;
}

vector<double> indexVecR2(const vector<double>& in1, const vector<double>& in2, const vector<double>& in3, const vector<size_t>& inds, const vector<double>& dtratio)
{
    size_t dims = inds.size();
    size_t t1len = dtratio.size();
    vector<double> result(len, 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < dims; i++)
    {
        if (inds[i] < t1len - 1)
        {
            result[i] = (1 - 3 * pow_const<2>(in3[i]) - 2 * pow_const<3>(in3[i])) * in1[inds[i] - 1] + (3 * pow_const<2>(in3[i]) + 2 * pow_const<3>(in3[i])) * in1[inds[i]] - (in3[i] + 2 * pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i]] - (pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i] + 1] / dtratio[inds[i] + 1];
        }
        else
        {
            result[i] = (1 - pow_const<2>(in3[i])) * in1[inds[i] - 1] + pow_const<2>(in3[i]) * in1[inds[i]] - (in3[i] + pow_const<2>(in3[i])) * in2[inds[i]];
        }
    }

    return result;
}

vector<double> indexMatAll(const int n, const vector<double>& posx, const vector<size_t>& indsy, const vector<double>& weightsy, const vector<double>& dtratio)
{
    size_t prod = indsy.size();
    size_t dims2 = weightsy.size();
    size_t depth = dims2 / prod;
    size_t t1len = dtratio.size();
    vector<double> result(len * len, 0.0);

    double inx, inx2, inx3;
    size_t inds, indsx;

#pragma omp parallel for private(inx, inx2, inx3, indsx, inds)
    for (size_t j = 0; j < prod; j++)
    {
        indsx = max(min((size_t)posx[j], (size_t)(posx[prod - 1] - 0.5)), (size_t)1);
        inx = posx[j] - indsx;
        inx2 = inx * inx;
        inx3 = inx2 * inx;
        inds = (indsx - 1) * len + indsy[j];

        auto weights_start = weightsy.begin() + 1;
        if (n == 1)
        {
            if (indsx < t1len - 1)
            {
                result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, QKv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQKv.begin() + len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, QKv.begin() + len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQKv.begin() + 2 * len + inds, 0.0) / dtratio[indsx + 1];
            }
            else
            {
                result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, QKv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, QKv.begin() + len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, dQKv.begin() + len + inds, 0.0);
            }
        }
        else
        {
            if (indsx < t1len - 1)
            {
                result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, QRv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQRv.begin() + len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, QRv.begin() + len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, dQRv.begin() + 2 * len + inds, 0.0) / dtratio[indsx + 1];
            }
            else
            {
                result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, QRv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, QRv.begin() + len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, dQRv.begin() + len + inds, 0.0);
            }
        }
    }

    return result;
}

void SigmaK(const vector<double>& qk, vector<double>& result)
{
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}

void SigmaR(const vector<double>& qk, const vector<double>& qr, vector<double>& result)
{
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]) * qr[i];
    }
}

vector<double> SigmaK10(const vector<double>& qk)
{
    vector<double> result(qk.size());
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return result;
}

vector<double> SigmaR10(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDDflambda(qk[i]) * qr[i];
    }
    return result;
}

vector<double> SigmaK01(const vector<double>& qk)
{
    vector<double> result(qk.size());
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = 0.0;
    }
    return result;
}

vector<double> SigmaR01(const vector<double>& qk, const vector<double>& qr)
{
    vector<double> result(qk.size());
#pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return result;
}

vector<double> ConvA(const vector<double>& f, const vector<double>& g, const double t)
{
    size_t length = integ.size();
    size_t depth = f.size() / length;
    vector<double> out(1, 0.0);
    if (depth == 1)
    {
        double temp = 0.0;
#pragma omp parallel for reduction(+:temp)
        for (size_t j = 0; j < length; j++)
        {
            temp += t * integ[j] * f[j] * g[j];
        }
        out[0] = temp;
    }
    else
    {
        out.resize(length, 0.0);
#pragma omp parallel for
        for (size_t j = 0; j < length; j++)
        {
            for (size_t i = 0; i < depth; i++)
            {
                out[j] += integ[i] * f[j * length + i] * g[j * length + i];
            }
            out[j] *= t * theta[j];
        }
    }
    return out;
}

vector<double> ConvR(const vector<double>& f, const vector<double>& g, const double t)
{
    size_t length = integ.size();
    size_t depth = f.size() / length;
    vector<double> out(length, 0.0);
#pragma omp parallel for
    for (size_t j = 0; j < length; j++)
    {
        for (size_t i = 0; i < depth; i++)
        {
            out[j] += integ[i] * f[j * length + i] * g[j * length + i];
        }
        out[j] *= t * (1 - theta[j]);
    }
    return out;
}

vector<double> QKstep()
{
    vector<double> temp(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
#pragma omp parallel for
    for (size_t i = 0; i < QKB1int.size(); i += len) {
        temp[i / len] = QKB1int[i];
    }
    vector<double> d1qK = (temp* (Dflambda(QKv[QKv.size() - len]) / T0)) + (qK * (-rInt.back())) + ConvR(SigmaRA2int, QKB2int, t1grid.back()) + ConvA(SigmaRA1int, QKB1int, t1grid.back()) + ConvA(SigmaKA1int, QRB1int, t1grid.back());
#pragma omp parallel for
    for (size_t i = 0; i < QKB1int.size(); i += len) {
        temp[i / len] = Dflambda(QKB1int[i]);
    }
    vector<double> d2qK = (temp * (QKv[QKv.size() - len] / T0)) + (qR*  (2 * Gamma)) + ConvR(QRA2int, SigmaKB2int, t1grid.back()) + ConvA(QRA1int, SigmaKB1int, t1grid.back()) + ConvA(QKA1int, SigmaRB1int, t1grid.back()) - (qK * rInt);
    return d1qK + (d2qK * theta);
}

vector<double> QRstep()
{
    vector<double> qR = getLastLenEntries(QRv, len);
    vector<double> d1qR = (qR * (-rInt.back())) + ConvR(SigmaRA2int, QRB2int, t1grid.back());
    vector<double> d2qR = (qR * rInt) - ConvR(QRA2int, SigmaRB2int, t1grid.back());
    return d1qR + (d2qR * theta);
}

double rstep()
{
    vector<double> sigmaK(len, 0.0), sigmaR(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
    const double t = t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    return Gamma + ConvA(sigmaR, qK, t).front() + ConvA(sigmaK, qR, t).front() + sigmaK.front() * qK.front() / T0;
}

double drstep()
{
    vector<double> sigmaK(len, 0.0), sigmaR(len, 0.0), dsigmaK(len, 0.0), dsigmaR(len, 0.0);
    vector<double> qK = getLastLenEntries(QKv, len);
    vector<double> qR = getLastLenEntries(QRv, len);
    vector<double> dqK = getLastLenEntries(dQKv, len);
    vector<double> dqR = getLastLenEntries(dQRv, len);
    const double t = t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR) * dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1).front() + ConvA(sigmaK, qR, 1).front() + ConvA(dsigmaR, qK, t).front() + ConvA(sigmaR, dqK, t).front() + ConvA(sigmaK, dqR, t).front() + (dsigmaK.front() * qK.front() + sigmaK.front() * dqK.front()) / T0;
}

double drstep2(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double t)
{
    vector<double> sigmaK(qK.size(), 0.0), sigmaR(qK.size(), 0.0), dsigmaK(qK.size(), 0.0), dsigmaR(qK.size(), 0.0);
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR)* dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1).front() + ConvA(sigmaK, qR, 1).front() + ConvA(dsigmaR, qK, t).front() + ConvA(sigmaR, dqK, t).front() + ConvA(sigmaK, dqR, t).front() + (dsigmaK.front() * qK.front() + sigmaK.front() * dqK.front()) / T0;
}

void appendAll(const vector<double>& qK,
    const vector<double>& qR,
    const vector<double>& dqK,
    const vector<double>& dqR,
    const double dr,
    const double t)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update t1grid and delta_t_ratio
    t1grid.push_back(t);
    size_t idx = t1grid.size() - 1;
    double tdiff = t1grid[idx] - t1grid[idx - 1];
    if (idx > 1) {
        double prev = t1grid[idx - 1] - t1grid[idx - 2];
        delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        delta_t_ratio.push_back(0.0);
    }

    for (size_t i = 0; i < length; i++)
    {
        QKv.push_back(qK[i]);
        QRv.push_back(qR[i]);
        dQKv.push_back(tdiff * dqK[i]);
        dQRv.push_back(tdiff * dqR[i]);
    }

    // 4) finally update drvec and rvec
    drvec.push_back(tdiff * dr);
    rvec.push_back(rstep());
}


void replaceAll(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double dr, const double t)
{
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        t1grid.back() = t;
        double tdiff = (t1grid[t1grid.size() - 1] - t1grid[t1grid.size() - 2]);

        if (t1grid.size() > 2) {
            delta_t_ratio.back() = tdiff /
                (t1grid[t1grid.size() - 2] - t1grid[t1grid.size() - 3]);
        }
        else {
            delta_t_ratio.back() = 0.0;
        }

#pragma omp parallel for
        for (size_t i = 0; i < replaceLength; i++)
        {
            QKv[length + i] = qK[i];
            QRv[length + i] = qR[i];
            dQKv[length + i] = tdiff * dqK[i];
            dQRv[length + i] = tdiff * dqR[i];
        }

        drvec.back() = tdiff * dr;
        rvec.back() = rstep();
    }
}

vector<double> bsearchPosSorted(const vector<double>& list, const vector<double>& elem) {
    vector<double> result(elem.size());
    size_t n0, n1, m;
    double Lm, El;

    for (size_t l = 0; l < elem.size(); ++l) {
        El = elem[l];
        n0 = 1; // Start index (Mathematica uses 1-based indexing, C++ uses 0-based)
        n1 = list.size(); // End index
        m = 1; // Midpoint index
        Lm = 0.0;

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

    return result;
}

vector<double> isearchPosSortedInit(const vector<double>& list, const vector<double>& elem, const vector<double>& inits) {
    size_t len = list.size();
    vector<double> result(elem.size(), len); // Initialize output vector
    double l0, l1, Lm, El;
    size_t n0, n1, m;
    bool even;
    double temp = len;

    // Iterate over `elem` in reverse order
    for (size_t l = 512; l-- > 0;) {
        El = list.back() * elem[l];
        n1 = min(static_cast<size_t>(ceil(temp)), len);
        n0 = static_cast<size_t>(floor(inits[l]));
        m = min(n0 + 1, len);
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
            if (m == len) {
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
        result[l] = temp;
    }

    return result;
}

void interpolate(const vector<double>& posB1xIn = {}, const vector<double>& posB2xIn = {},
    const bool same = false)
{
    // Compute posB1x
    vector<double> posB1x = !posB1xIn.empty() ?
        (same ? posB1xIn : isearchPosSortedInit(t1grid, theta, posB1xIn)) :
        bsearchPosSorted(t1grid, theta * t1grid.back());

    // Compute posB2x
    vector<double> posB2x = !posB2xIn.empty() ?
        (same ? posB2xIn : isearchPosSortedInit(t1grid, phi2, posB2xIn)) :
        bsearchPosSorted(t1grid, phi2 * t1grid.back());

    // Update old positions
    posB1xOld = posB1x;
    posB2xOld = posB2x;

    // Interpolate QKA1int and QRA1int
    if (t1grid.back() > 0) {
        QKA1int = indexVecLN3(1, weightsA1y, indsA1y);
        QRA1int = indexVecLN3(2, weightsA1y, indsA1y);
    }
    else {
        QKA1int.assign(len * len, QKv[0]);
        QRA1int.assign(len * len, QRv[0]);
    }
    SigmaK(QKA1int, SigmaKA1int);
    SigmaR(QKA1int, QRA1int, SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (t1grid.back() > 0) {
        QKA2int = indexVecLN3(1, weightsA2y, indsA2y);
        QRA2int = indexVecLN3(2, weightsA2y, indsA2y);
    }
    else {
        QKA2int.assign(len * len, QKv[0]);
        QRA2int.assign(len * len, QRv[0]);
    }
    SigmaR(QKA2int, QRA2int, SigmaRA2int);


    // Interpolate QKB1int and QRB1int
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
    Subtract(vector<double>(Floor.begin(), Floor.end()), posB1x, diff);
    if (t1grid.back() > 0) {
        QKB1int = indexVecN(1, len, diff, Floor, delta_t_ratio);
        QRB1int = indexVecN(2, len, diff, Floor, delta_t_ratio);
    }
    else {
        QKB1int.assign(len * len, QKv[0]);
        QRB1int.assign(len * len, QRv[0]);
    }
    SigmaK(QKB1int, SigmaKB1int);
    SigmaR(QKB1int, QRB1int, SigmaRB1int);

    // Interpolate QKB2int and QRB2int
    if (t1grid.back() > 0) {
        QKB2int = indexMatAll(1, posB2x, indsB2y, weightsB2y, delta_t_ratio);
        QRB2int = indexMatAll(2, posB2x, indsB2y, weightsB2y, delta_t_ratio);
    }
    else {
        QKB2int.assign(len * len, QKv[0]);
        QRB2int.assign(len * len, QRv[0]);
    }
    SigmaK(QKB2int, SigmaKB2int);
    SigmaR(QKB2int, QRB2int, SigmaRB2int);

    // Interpolate rInt
    if (t1grid.back() > 0) {
        rInt = indexVecR2(rvec, drvec, diff, Floor, delta_t_ratio);
    }
    else {
        rInt.assign(len, rvec[0]);
    }
}

double SSPRK104()
{
    const size_t stages = 10;
    const double amat[stages][stages] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0}
    };
    const double bvec[stages] = { 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10 };
    const double b2vec[stages] = { 0., 2.0 / 9, 0, 0, 5.0 / 18, 1.0 / 3, 0., 0., 0., 1.0 / 6 };

    // Initialize variables
    vector<vector<double>> gKvec(stages + 1, vector<double>(len, 0.0));
    gKvec[0] = getLastLenEntries(QKv, len);
    vector<vector<double>> gRvec(stages + 1, vector<double>(len, 0.0));
    gRvec[0] = getLastLenEntries(QRv, len);
    vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = t1grid.back();

    vector<double> gKe(len, 0.0);
    vector<double> gRe(len, 0.0);
    double gte = 0.0;

    vector<vector<double>> hKvec(stages, vector<double>(len, 0.0));
    vector<vector<double>> hRvec(stages, vector<double>(len, 0.0));
    vector<double> htvec(stages, 0.0);

    vector<vector<double>> posB1xvec(3, vector<double>(len, 0.0));
    vector<vector<double>> posB2xvec(3, vector<double>(len * len, 0.0));
    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (QKv.size() == len || n != 0) {
            interpolate(
                (n == 0 ? vector<double>{} : (n == 5 ? posB1xvec[0] : (n == 6 ? posB1xvec[1] : (n == 7 ? posB1xvec[2] : posB1xOld)))),
                (n == 0 ? vector<double>{} : (n == 5 ? posB2xvec[0] : (n == 6 ? posB2xvec[1] : (n == 7 ? posB2xvec[2] : posB2xOld)))),
                (n == 5 || n == 6 || n == 7)
            );
        }

        // Update position vectors
        if (n == 2) {
            posB1xvec[0] = posB1xOld;
            posB2xvec[0] = posB2xOld;
        }
        else if (n == 3) {
            posB1xvec[1] = posB1xOld;
            posB2xvec[1] = posB2xOld;
        }
        else if (n == 4) {
            posB1xvec[2] = posB1xOld;
            posB2xvec[2] = posB2xOld;
        }


        // Compute k[n]
        hKvec[n] = QKstep();
        hRvec[n] = QRstep();
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            vector<double> lastQKv = getLastLenEntries(QKv, len);
            vector<double> lastQRv = getLastLenEntries(QRv, len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], t1grid.back());
            Sum(gKvec[0], hKvec[0] * (delta_t * amat[1][0]), gKvec[n + 1]);
            Sum(gRvec[0], hRvec[0] * (delta_t * amat[1][0]), gRvec[n + 1]);
            gtvec[n + 1] = gtvec[0] + delta_t * amat[1][0] * htvec[0];
            appendAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); //Append Update

        }
        else if (n == stages - 1) {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                Sum(gKvec[n + 1], hKvec[j] * (delta_t * bvec[j]), gKvec[n + 1]);
                Sum(gRvec[n + 1], hRvec[j] * (delta_t * bvec[j]), gRvec[n + 1]);
                gtvec[n + 1] += delta_t * bvec[j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
        else {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                Sum(gKvec[n + 1], hKvec[j] * (delta_t * amat[n + 1][j]), gKvec[n + 1]);
                Sum(gRvec[n + 1], hRvec[j] * (delta_t * amat[n + 1][j]), gRvec[n + 1]);
                gtvec[n + 1] += delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]); // Replace Update
        }
    }

    // Final interpolation
    interpolate(posB1xOld, posB2xOld);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        Sum(gKe, hKvec[j] * (delta_t * b2vec[j]), gKe);
        Sum(gRe, hRvec[j] * (delta_t * b2vec[j]), gRe);
        gte += delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    double error = 0.0;
    for (size_t i = 0; i < gKvec[stages].size(); ++i) {
        error += abs(gKvec[stages][i] - gKe[i]);
    }
    for (size_t i = 0; i < gRvec[stages].size(); ++i) {
        error += abs(gRvec[stages][i] - gRe[i]);
    }
    error += abs(gtvec[stages] - gte);

    return error;
}

void init()
{
    import();

    t1grid.resize(1, 0.0);
    delta_t_ratio.resize(1, 0.0);
    specRad = 4 * sqrt(DDflambda(1));

    delta_t = delta_t_min;
    loop = 0;
    delta = 1;
    delta_old = 0;

    posB1xOld.resize(len, 1.0);
    posB2xOld.resize(len * len, 0.0);

    SigmaKA1int.resize(len * len, 0.0);
    SigmaRA1int.resize(len * len, 0.0);
    SigmaKB1int.resize(len * len, 0.0);
    SigmaRB1int.resize(len * len, 0.0);
    SigmaKA2int.resize(len * len, 0.0);
    SigmaRA2int.resize(len * len, 0.0);
    SigmaKB2int.resize(len * len, 0.0);
    SigmaRB2int.resize(len * len, 0.0);

    QKA1int.resize(len * len, 0.0);
    QRA1int.resize(len * len, 0.0);
    QKB1int.resize(len * len, 0.0);
    QRB1int.resize(len * len, 0.0);
    QKA2int.resize(len * len, 0.0);
    QRA2int.resize(len * len, 0.0);
    QKB2int.resize(len * len, 0.0);
    QRB2int.resize(len * len, 0.0);

    QKv.resize(len, 1.0);
    QRv.resize(len, 1.0);
    dQKv.resize(len, 0.0);
    dQRv.resize(len, 0.0);
    rvec.resize(1, Gamma + Dflambda(1) / T0);
    drvec.resize(1, rstep());

    rInt.resize(len, 0.0);
    drInt.resize(len, 0.0);

}

int main() {
    // 0) Initialize
    init();

    std::cout << "Starting simulation..." << std::endl;

    // 1) Open the output file for correlation
    std::ofstream corr("correlation.txt");
    if (!corr) {
        std::cerr << "Error: Unable to open correlation.txt" << std::endl;
        return 1;
    }
    corr << std::fixed << std::setprecision(9);

    auto start = high_resolution_clock::now();

    // 2) Main loop
    while (t1grid.back() < tmax && loop < maxLoop) {
        delta_old = delta;
        delta = SSPRK104();
        loop++;

        // primitive time-step adaptation
        if (delta < delta_max && loop > 5 &&
            (delta < 1.1 * delta_old || delta_old == 0) && false &&
            rmax / specRad > delta_t && delta_t_ratio.back() == 1)
        {
            delta_t *= 1.01;
        }
        else if (delta > 2 * delta_max && delta_t > delta_t_min) {
            delta_t *= 0.9;
        }

        // display a video
        std::cout << "loop: " << loop
            << " time: " << t1grid.back()
            << " delta: " << delta
            << " specRad: " << specRad
            << std::endl;

        // record QK(t,0) to file
        double t = t1grid.back();
        double qk0 = QKv[(t1grid.size() - 1) * len + 0];
        corr << t << "\t" << qk0 << "\n";
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() / 1000 << " milliseconds" << endl;

    // 3) Print the final results
    std::cout << "final delta_t: " << delta_t << std::endl;
    std::cout << "final delta:   " << delta << std::endl;
    std::cout << "final loop:    " << loop << std::endl;
    std::cout << "final t1grid:  " << t1grid.back() << std::endl;
    std::cout << "final rvec:    " << rvec.back() << std::endl;
    std::cout << "final drvec:   " << drvec.back() << std::endl;
    std::cout << "final QKv:     " << QKv[(t1grid.size() - 1) * len] - 1 << std::endl;
    std::cout << "final QRv:     " << QRv[(t1grid.size() - 1) * len] - 1 << std::endl;
    std::cout << "Simulation finished." << std::endl;

    // 4) Close the file
    corr.close();

    return 0;
}