import numpy as np
from alias_copyi_EDG import EDG
from alias_copyi_EDG import funs as Ifuns, funs_metric as Mfuns
from alias_copyi_EDG import demoapi
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

# name, knn = "Agg", 17
name, knn = "CNBC", 7

X, y_true, N, dim, c_true = demoapi.load_data(name)

D_full = Ifuns.EuDist2(X, X, squared=True)
np.fill_diagonal(D_full, -1)
NN_full = np.argsort(D_full, axis=1)
NN_full = NN_full.astype(np.int32)
np.fill_diagonal(D_full, 0)

f1_arr = list()
NN = NN_full[:, 1:(knn+1)]
NND = Ifuns.matrix_index_take(D_full, NN)

model = EDG(NN, NND)
model.opt()
y = model.y_pre

f1 = Mfuns.f1(y_true, y)
f1_arr.append(f1)
print(f1)

# paper:
# Agg,  0.996,
# CNBC, f1=0.696
