from EDG import EDG
import funs as Ifuns
import funs_metric as Mfuns
import numpy as np
import matplotlib.pyplot as plt

X, y_true, N, dim, c_true = Ifuns.load_mat("toydata/Agg.mat")
X = X.astype(np.float64)

D_full = Ifuns.EuDist2(X, X, squared=True)
np.fill_diagonal(D_full, -1)
NN_full = np.argsort(D_full, axis=1)
NN_full = NN_full.astype(np.int32)
np.fill_diagonal(D_full, 0)

knn = 10
NN = NN_full[:, 1:(knn+1)]
NND = Ifuns.matrix_index_take(D_full, NN)

model = EDG(NN, NND)
model.opt()
y = model.y_pre

pre = Mfuns.precision(y_true=y_true, y_pred=y)
rec = Mfuns.recall(y_true=y_true, y_pred=y)
f1 = 2 * pre * rec / (pre + rec)

print("{:.3f}".format(pre))
print("{:.3f}".format(f1))
