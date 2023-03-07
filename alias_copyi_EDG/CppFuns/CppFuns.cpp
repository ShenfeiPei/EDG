#include "CppFuns.h"

namespace cf{
    void EuDist2(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C){

        int n = A.size();
        int m = B.size();

        vector<double> anorm(n, 0);
        square_sum_by_row(A, anorm);

        vector<double> bnorm(m, 0);
        square_sum_by_row(B, bnorm);

        #pragma omp parallel for
        for (int k = 0; k < n * m; k++){
            int i, j;
            i = k/m;
            j = k%m;
            C[i][j] = anorm[i] + bnorm[j] - 2 * inner_product(A[i].begin(), A[i].end(), B[j].begin(), (double) 0);
        }
    }

    void EuDist2_byCol(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C){

        int n = A[0].size();
        int m = B[0].size();

        vector<double> anorm(n, 0);
        square_sum_by_col(A, anorm);

        vector<double> bnorm(m, 0);
        square_sum_by_col(B, bnorm);

        #pragma omp parallel for
        for (int k = 0; k < n * m; k++){
            int i, j;
            i = k/m;
            j = k%m;
            double tmp_d = 0;
            for (int l = 0; l < B.size(); l++){
                tmp_d += A[l][i] * B[l][j];
            }
            C[i][j] = anorm[i] + bnorm[j] - 2 * tmp_d;
        }
    }

    void EuDist2(vector<vector<double>> &A, vector<double> &anorm, vector<vector<double>> &B, vector<double> &bnorm, vector<vector<double>> &C){

        int n = A.size();
        int m = B.size();

        #pragma omp parallel for
        for (int k = 0; k < n * m; k++){
            int i, j;
            i = k/m;
            j = k%m;
            C[i][j] = anorm[i] + bnorm[j] - 2 * inner_product(A[i].begin(), A[i].end(), B[j].begin(), (double) 0);
        }
    }

    void EuDist2_byCol(vector<vector<double>> &A, vector<double> &anorm, vector<vector<double>> &B, vector<double> &bnorm, vector<vector<double>> &C){

        int n = A[0].size();
        int m = B[0].size();

        #pragma omp parallel for
        for (int k = 0; k < n * m; k++){
            int i, j;
            i = k/m;
            j = k%m;
            double tmp_d = 0;
            for (int l = 0; l < B.size(); l++){
                tmp_d += A[l][i] * B[l][j];
            }
            C[i][j] = anorm[i] + bnorm[j] - 2 * tmp_d;
        }
    }

    void square_sum_by_row(vector<vector<double>> &X, vector<double> &norm){

        #pragma omp parallel for
        for (int i = 0; i < X.size(); i++){
            norm[i] = inner_product(X[i].begin(), X[i].end(), X[i].begin(), (double) 0);
        }
    }

    void square_sum_by_col(vector<vector<double>> &X, vector<double> &norm){

        #pragma omp parallel for
        for (int j = 0; j < X[0].size(); j++){
            norm[j] = 0;
            for (int i = 0; i < X.size(); i++){
                norm[j] += X[i][j] * X[i][j];
            }
        }
    }

    template <typename T, typename U>
    void argsort_TwoArr(vector<T> &v1, vector<U> &v2, vector<int> &ind){
        iota(ind.begin(), ind.end(), 0);
        std::sort(ind.begin(), ind.end(), [&v1, &v2](int i1, int i2){
	    // v1[i1] < v1[i2]
	    T diff = abs(v1[i1] - v1[i2]);
	    if (diff > 1e-10){
	        return v1[i1] < v1[i2];
        }else{
	        return v2[i1] > v2[i2];
	    }
	});
    }
    template void argsort_TwoArr<int, int>(vector<int> &v1, vector<int> &v2, vector<int> &ind);
    template void argsort_TwoArr<double, double>(vector<double> &v1, vector<double> &v2, vector<int> &ind);
    template void argsort_TwoArr<int, double>(vector<int> &v1, vector<double> &v2, vector<int> &ind);
    template void argsort_TwoArr<double, int>(vector<double> &v1, vector<int> &v2, vector<int> &ind);

    template <typename T>
    double median_vec2d(vector<vector<T>> &v){
        double ret;

        vector<T> v2;
        for (int i = 0; i < v.size(); i++){
            for (int j = 0; j < v[i].size(); j++){
                v2.push_back(v[i][j]);
            }
        }
        std::sort(v2.begin(), v2.end());
        ret = v2[v2.size()/2];

        return ret;

    }
    template double median_vec2d<int>(vector<vector<int>> &v);
    template double median_vec2d<double>(vector<vector<double>> &v);

    template <typename T>
    double median_v(vector<T> &v, int copy){
        double ret;
        if (copy==1){
            vector<T> v_sorted = v;
            std::sort(v_sorted.begin(), v_sorted.end());
           // if (n % 2 == 1){
           //     ret = v_sorted[n/2];
           // }else{
           //     ret = (v_sorted[n/2 - 1] + v_sorted[n/2])/2;
           // }
            ret = v_sorted[v.size()/2];
        }else{
            std::sort(v.begin(), v.end());
            ret = v[v.size()/2];
    //        if (n % 2 == 1){
    //            ret = v[n/2];
    //        }else{
    //            ret = (v[n/2 - 1] + v[n/2])/2;
    //        }
        }
        return ret;

    }
    template double median_v<int>(vector<int> &v, int copy);
    template double median_v<double>(vector<double> &v, int copy);

    template <typename T, typename U>
    void argsort_TwoArr(T *v1, U *v2, int n, int *ind){
        for (int i = 0; i < n; i++) ind[i] = i;
        std::sort(ind, ind + n, [&v1, &v2](int i1, int i2){ return v1[i1] < v1[i2] || (v1[i1] < v1[i2] + 1e-8 && v2[i1] > v2[i2]); });
    }
    template void argsort_TwoArr<int, double>(int *v1, double *v2, int n, int *ind);
    template void argsort_TwoArr<double, double>(double *v1, double *v2, int n, int *ind);
    template void argsort_TwoArr<int, int>(int *v1, int *v2, int n, int *ind);
    template void argsort_TwoArr<double, int>(double *v1, int *v2, int n, int *ind);
};

