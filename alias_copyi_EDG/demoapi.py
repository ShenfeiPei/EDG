import numpy as np
from alias_copyi_EDG import EDG
from alias_copyi_EDG import funs as Ifuns, funs_metric as Mfuns
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt

def load_data(name):
    cur_path = os.path.dirname(__file__)
    full_name = os.path.join(cur_path, f"data/{name}.mat")
    X, y_true, N, dim, c_true = Ifuns.load_mat(full_name)
    X = X.astype(np.float64)
    return X, y_true, N, dim, c_true
