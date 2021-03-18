#ifndef EDG_H_
#define EDG_H_

#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>    // std::min_element, std::max_element
#include "CppFuns.h"
#include "Graph.h"

using namespace std;

class EDG{
public:
    int N;
    int knn;
    vector<vector<int>> NN;
    vector<vector<double>> NND;
    vector<vector<double>> NNS;

    vector<int> y;
    vector<int> nc;
    vector<int> den;
    vector<double> rho;

    double max_d;
    double t;
    double rho_g;

    EDG();
    EDG(vector<vector<int>> &NN, vector<vector<double>> &NND);
    ~EDG();

    double dist(int i, int j);

    void compute_nc();

    double compute_tr(int i, int j);

    void compute_NNS();

    void compute_rho();

    int local_den(int i);

    void compute_den();

    void clustering();
};

#endif
