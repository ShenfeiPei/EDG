#ifndef CPPFUNS_H_
#define CPPFUNS_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

using namespace std;

namespace cf{
    void EuDist2(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C);

    void EuDist2(vector<vector<double>> &A, vector<double> &anorm, vector<vector<double>> &B, vector<double> &bnorm, vector<vector<double>> &C);

    void EuDist2_byCol(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C);

    void EuDist2_byCol(vector<vector<double>> &A, vector<double> &anorm, vector<vector<double>> &B, vector<double> &bnorm, vector<vector<double>> &C);

    void square_sum_by_row(vector<vector<double>> &X, vector<double> &norm);
    void square_sum_by_col(vector<vector<double>> &X, vector<double> &norm);

    template <typename T, typename U>
    void argsort_TwoArr(vector<T> &v1, vector<U> &v2, vector<int> &ind);

    template <typename T>
    double median_vec2d(vector<vector<T>> &v);

    template <typename T>
    double median_v(vector<T> &v, int copy);

    template <typename T, typename U>
    void argsort_TwoArr(T *v1, U *v2, int n, int *ind);
}

#endif
