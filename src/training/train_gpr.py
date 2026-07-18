import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib

def train_surrogate(filename="aero_data.h5"):
    with h5py.File(filename, 'r') as hf:
        X = np.column_stack((hf['mach'][:], hf['alpha'][:]))
        y = hf['Cd'][:] # We target Drag for this example
    
    # Kernel: RBF is standard for smooth physical aero fields
    kernel = C(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    
    gpr.fit(X, y)
    joblib.dump(gpr, "aero_surrogate.pkl")
    print("Surrogate (GPR) model trained and saved.")

train_surrogate()