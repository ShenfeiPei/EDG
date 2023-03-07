cimport numpy as np
import numpy as np
np.import_array()

from .EDG_ cimport EDG

cdef class PyEDG:
    cdef EDG c_EDG

    def __cinit__(self, np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2] NND):
        self.c_EDG = EDG(NN, NND)

    def opt(self):
        self.c_EDG.clustering()

    @property
    def y_pre(self):
        return np.array(self.c_EDG.y)

    @property
    def edge(self):
        return np.array(self.c_EDG.edge)

    @property
    def rho(self):
        return np.array(self.c_EDG.rho)

    @property
    def nc(self):
        return np.array(self.c_EDG.nc)
    @property
    def den(self):
        return np.array(self.c_EDG.den)
