from libcpp.vector cimport vector

cdef extern from "CppFuns.cpp":
    pass

cdef extern from "CppFuns.h" namespace "cf":
    void argsort_TwoArr(double *v1, int *v2, int n, int *ind);
    double median_vec2d(vector[vector[double]] &v);
    double median_v(vector[double] &v, int copy);

from CppFuns.Graph_ cimport Graph

cdef extern from "EDG.cpp":
    pass

cdef extern from "EDG.h":
    cdef cppclass EDG:

        vector[vector[int]] NN
        vector[vector[double]] NND
        vector[vector[double]] NNS
        vector[int] nc
        vector[int] y 
        vector[int] den
        vector[double] rho

        double max_d
        double sigma
        double rho_g
        int knn

        EDG() except +
        EDG(vector[vector[int]] &NN, vector[vector[double]] &NND) except +
        void clustering()
        void compute_nc()
        double dist(int i, int j)
        double compute_tr(int i, int k)
        void compute_NNS()
        int local_den(int i)
        void compute_den()
        void compute_rho()

