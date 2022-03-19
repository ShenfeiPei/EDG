from EDG import EDG
import funs as Ifuns
import funs_metric as Mfuns
import numpy as np

name, knn = "Agg", 10
# name, knn = "Jain", 8
# name, knn = "Gestalt", 8
# name, knn = "Three_circle", 10
# name, knn = "CNBC", 7

X, y_true, N, dim, c_true = Ifuns.load_mat(f"data/{name}.mat")
X = X.astype(np.float64)

D_full = Ifuns.EuDist2(X, X, squared=True)
np.fill_diagonal(D_full, -1)
NN_full = np.argsort(D_full, axis=1)
NN_full = NN_full.astype(np.int32)
np.fill_diagonal(D_full, 0)

NN = NN_full[:, 1:(knn+1)]
NND = Ifuns.matrix_index_take(D_full, NN)

model = EDG(NN, NND)
model.opt()
y = model.y_pre

pre = Mfuns.precision(y_true=y_true, y_pred=y)
rec = Mfuns.recall(y_true=y_true, y_pred=y)
f1 = 2 * pre * rec / (pre + rec)
print(f"pre={pre:.3f}, f1={f1:.3f}")

# paper: 
# 1. toy:
# Agg,  0.996,
# Jain, 1.000,
# Gestalt, 0.950
# Three-circle, 0.996
# 2. real:
# CNBC, pre=0.778, f1=0.696
