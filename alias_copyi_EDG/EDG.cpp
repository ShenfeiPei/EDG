#include "EDG.h"

EDG::EDG(){}

EDG::EDG(vector<vector<int>> &NN, vector<vector<double>> &NND){
    this->NN = NN;
    this->NND = NND;
    this->N = NN.size();
    this->knn = NN[0].size();

    NNS = vector<vector<double>>(N, vector<double>(knn, 0));
    nc  = vector<int>(N, 0);
    den  = vector<int>(N, 0);
    rho  = vector<double>(N, 0);

    vector<double> tmp(N, 0);
    for (int i = 0; i < N; i++){
        tmp[i] = *std::max_element(NND[i].begin(), NND[i].end());
    }
    max_d = *std::max_element(tmp.begin(), tmp.end());
    t = cf::median_vec2d(NND);
    // cout << "sigma = " << t << endl;
}

EDG::~EDG(){}

void EDG::compute_nc(){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        int j = 0;
        nc[i] = 0;
        for (int k = 0; k < knn; k++){
            j = NN[i][k];
            if (find(NN[j].begin(), NN[j].end(), i) == NN[j].end()) break;
            else nc[i] += 1;
        }
    }
}

double EDG::dist(int i, int j){
    double ret = 0;
    if (i==j) ret = 0;
    else{
        vector<int>::iterator it;
        it = find(NN[i].begin(), NN[i].begin() + nc[i], j);
        if (it == NN[i].end()){
            ret = max_d;
        }
        else{
            int ind = std::distance(NN[i].begin(), it);
            ret = NND[i][ind];
        }
    }
    return ret;
}

double EDG::compute_tr(int i, int k){
    int j = NN[i][k];
    int wrow = 2*knn;
    int wcol = 2*knn;
    vector<vector<double>> W(wrow, vector<double>(wcol, 0));
    vector<vector<double>> L(wrow, vector<double>(wcol, 0));
    vector<double> diagA(wrow, 0);
    vector<vector<int>> ind_M(wrow, vector<int>(wcol, 0));

    //double  *W = new double [wrow*wcol];
    //double  *L = new double [wrow*wcol];
    //double *diagA = new double [wrow];
    //int *ind_M = new int [wrow*wcol];
    int tmp_i, tmp_j;
    double tmp_d;
    //int *ind4sort = new int[wrow];

    vector<int> ind4sort(wrow, 0);
    for (int a = 0; a < wrow; a++){
        ind4sort[a] = a;
    }

    double w = 1;
    for (int a = 0; a < knn; a++){
        for (int b = a + 1; b < knn; b ++){
            tmp_i = NN[i][a];
            tmp_j = NN[i][b];
            tmp_d = dist(tmp_i, tmp_j);

            W[a][b] = tmp_d;
            W[b][a] = tmp_d;

            tmp_i = NN[j][a];
            tmp_j = NN[j][b];
            tmp_d = dist(tmp_i, tmp_j);

            W[knn + a][knn + b] = tmp_d;
            W[knn + b][knn + a] = tmp_d;
        }
    }

    // Dij & Dji
    for (int a = 0; a < knn; a++){
        for (int b = 0; b < knn; b++){
            tmp_i = NN[i][a];
            tmp_j = NN[j][b];
            tmp_d = dist(tmp_i, tmp_j);
            W[a][knn + b] = tmp_d;
            W[knn + b][a] = tmp_d;
        }
    }
    for (int a = 0; a < wrow; a++) W[a][a] = -1;
    for (int a = 2; a < wrow; a++){
        cf::argsort_TwoArr(W[a], ind4sort, ind_M[a]);
    }
    for (int i1 = 0; i1 < wrow; i1++){
        tmp_i = ind_M[i1][0];
        W[i1][tmp_i] = 0;

        for (int j1 = 1; j1 < (knn + 1); j1++){
            tmp_i = ind_M[i1][j1];
            W[i1][tmp_i] = exp(-W[i1][tmp_i]/t);
        }
        for (int j1 = knn+1; j1 < wcol; j1++){
            tmp_i = ind_M[i1][j1];
            W[i1][tmp_i] = 0;
        }
    }

    for (int i1 = 0; i1 < wrow; i1++){
        diagA[i1] = 0;
        for (int j1 = 0; j1 < i1; j1++){
            diagA[i1] += -L[i1][j1];
        }
        for (int j1 = i1; j1 < wcol; j1++){
            tmp_d = (W[i1][j1] + W[j1][i1])/2;
            L[i1][j1] = -tmp_d;  // L = - A_sys
            L[j1][i1] = -tmp_d;
            diagA[i1] += tmp_d;
        }
    }

    for (int i1 = 0; i1 < wrow; i1++){
        L[i1][i1] += diagA[i1];
    }

    double La = 0, Da = 0, Lb = 0, Db = 0;
    for (int i1 = 0; i1 < knn; i1 ++){
        Da += diagA[i1];
        for (int j1 = 0; j1 < knn; j1++){
            La += L[i1][j1];
        }
    }
    for (int i1 = knn; i1 < wrow; i1 ++){
        Db += diagA[i1];
        for (int j1 = knn; j1 < wrow; j1++){
            Lb += L[i1][j1];
        }
    }

    return w*(La/Da + Lb/Db);
}

void EDG::compute_NNS(){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int k = 0; k < nc[i]; k++){
            NNS[i][k] = compute_tr(i, k);
        }
    }
}

void EDG::compute_rho(){
    double max_rho = -1;
    for (int i = 0; i < N; i++){
        
        rho[i] = std::accumulate(NNS[i].begin(), NNS[i].begin() + nc[i], 0);
        if (rho[i] > max_rho) max_rho = rho[i];
    }
    // cout << "sum_rho = " << accumulate(rho.begin(), rho.end(), 0) << endl;
    max_rho = max_rho/10;
    for (int i = 0; i < N; i++){
        rho[i] = floor(rho[i] / max_rho);
    }

    vector<double> rho_sorted(N, 0);
    for (int i = 0; i < N; i++) rho_sorted[i] = rho[i];
    sort(rho_sorted.begin(), rho_sorted.end());
    int start_id = 0;
    for (int i = 0; i < N; i++) if (rho_sorted[i] > 0){
        start_id = i;
        break;
    }
    // cout << "start_id" << start_id << endl;

    rho_g = rho_sorted[start_id + (N - start_id)/2]-1;
    // cout << "rho_g = " << rho_g << endl;
}

int EDG::local_den(int i){
    vector<double> rho_loc;
    int tmp_nb = 0;
    for (int k = 0; k < nc[i]; k++){
        tmp_nb = NN[i][k];
        if (den[tmp_nb] < 2){
            rho_loc.push_back(rho[tmp_nb]);
        }
    }

    int ret = 0;
    if (rho_loc.size() > 2){
        if (rho[i] >= cf::median_v(rho_loc, 0)) ret = 1;
        else ret = 0;
    }else ret = 1;

    return ret;

}
void EDG::compute_den(){
    for (int i = 0; i < N; i++) den[i] = 0;

    // density
    for (int i = 0; i < N; i++){
        if (rho[i] >= rho_g) den[i] = 2;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) if (den[i] < 2){
        den[i] = local_den(i);
    }
}

void EDG::clustering(){

    compute_nc();

//    show_M(nc, 1, N, 1, 5);
    compute_NNS();
    compute_rho();
    compute_den();

    // sub rho[rho>0]

    Graph g(N); // 5 vertices numbered from 0 to 4

    int j = 0;
    for (int i = 0; i < N; i++){
        if (den[i] > 0){
            for (int k = 0; k < nc[i]; k++){
                j = NN[i][k];
                if (den[j] > 0){
                    g.addEdge(i, j);
                    vector<int> tmp = {i, j};
                    edge.push_back(tmp);
                }else break;
            }
        }else{
            if (nc[i] > 0){
                j = NN[i][0];
                g.addEdge(i, j);
                vector<int> tmp = {i, j};
                edge.push_back(tmp);
            }
        }
    }

    g.connectedComponents();
    y = g.y;
}
