import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold
from joblib import dump, load


# Full coefficient set produced by the CFD pipeline (see src/aero/cfd_generators).
COEFF_NAMES = ["Cd", "Cy", "Cm", "Cn", "Cl_roll"]
N_COEFF = len(COEFF_NAMES)

# Input feature order: mach, alpha, beta, altitude, delta(control deflection).
FEATURE_NAMES = ["mach", "alpha", "beta", "altitude", "delta"]


def load_data(filename="aero_data.h5"):
    """Load an HDF5 aero dataset (legacy or CFD-pipeline format).

    Legacy files (generate_aero_data.py) expose mach/alpha/beta/altitude and
    Cd/Cy/Cm; the CFD pipeline (src/aero/cfd_generators) stores a single
    ``coeffs`` array of shape (N, 5) plus the same feature columns. Missing
    columns (Cn, Cl_roll, delta) are synthesised as zeros so the unified
    5-output model can be trained on either source.
    """
    with h5py.File(filename, "r") as hf:
        keys = set(hf.keys())
        need = ["mach", "alpha", "beta", "altitude"]
        for k in need:
            if k not in keys:
                raise KeyError(f"Dataset {filename} missing required field '{k}'")

        mach = np.asarray(hf["mach"][:], dtype=float)
        alpha = np.asarray(hf["alpha"][:], dtype=float)
        beta = np.asarray(hf["beta"][:], dtype=float)
        altitude = np.asarray(hf["altitude"][:], dtype=float)
        n = mach.size

        delta = np.asarray(hf["delta"][:], dtype=float) if "delta" in keys else np.zeros(n)

        if "coeffs" in keys:
            coeffs = np.asarray(hf["coeffs"][:], dtype=float)
            Cd = coeffs[:, 0]
            Cy = coeffs[:, 1]
            Cm = coeffs[:, 2]
            Cn = coeffs[:, 3] if coeffs.shape[1] > 3 else np.zeros(n)
            Cl_roll = coeffs[:, 4] if coeffs.shape[1] > 4 else np.zeros(n)
        else:
            Cd = np.asarray(hf["Cd"][:], dtype=float)
            Cy = np.asarray(hf["Cy"][:], dtype=float)
            Cm = np.asarray(hf["Cm"][:], dtype=float)
            Cn = np.asarray(hf["Cn"][:], dtype=float) if "Cn" in keys else np.zeros(n)
            Cl_roll = (
                np.asarray(hf["Cl_roll"][:], dtype=float)
                if "Cl_roll" in keys
                else np.zeros(n)
            )

        sigma = (
            np.asarray(hf["sigma"][:], dtype=float)
            if "sigma" in keys
            else np.full((n, N_COEFF), 0.02)
        )

    X = np.column_stack((mach, alpha, beta, altitude, delta))
    y = np.column_stack((Cd, Cy, Cm, Cn, Cl_roll))
    return X, y, sigma


def build_kernel():
    # Per-feature length scales: mach, alpha, beta, altitude, delta.
    return C(1.0) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0, 1.0]) + WhiteKernel(noise_level=0.01)


class MultiOutputGPR:
    """One GPR per aerodynamic coefficient (5 outputs)."""

    def __init__(self):
        self.models = [
            GaussianProcessRegressor(kernel=build_kernel(), n_restarts_optimizer=5)
            for _ in range(N_COEFF)
        ]
        self.target_names = list(COEFF_NAMES)

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    def predict(self, X, return_std=False):
        means, stds = [], []
        for model in self.models:
            if return_std:
                m, s = model.predict(X, return_std=True)
                means.append(m)
                stds.append(s)
            else:
                means.append(model.predict(X))
        if return_std:
            return np.column_stack(means), np.column_stack(stds)
        return np.column_stack(means)

    def jacobian_fd(self, X, eps=1e-6):
        """Central finite-difference Jacobian d(coeffs)/d(inputs).

        Returns (N, n_coeff, n_features). Computed via central differences on
        the real GPR ``predict`` (sklearn rejects complex input, so true
        complex-step is unavailable through it). With eps=1e-6 the derivative
        error is ~1e-9, which is machine-accurate enough for Dymos/
        OpenMDAO gradient-based optimizers.
        """
        n = X.shape[0]
        nf = X.shape[1]
        J = np.zeros((n, N_COEFF, nf), dtype=float)
        for j in range(nf):
            xp = X.copy(); xp[:, j] += eps
            xm = X.copy(); xm[:, j] -= eps
            J[:, :, j] = (self.predict(xp) - self.predict(xm)) / (2.0 * eps)
        return J

    def kfold_cv(self, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {name: [] for name in self.target_names}
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = MultiOutputGPR()
            model.fit(X_train, y_train)
            preds, _ = model.predict(X_val, return_std=True)
            for j, name in enumerate(self.target_names):
                rmse = np.sqrt(np.mean((preds[:, j] - y_val[:, j]) ** 2))
                scores[name].append(rmse)
        return {k: np.mean(v) for k, v in scores.items()}


def train_gpr(filename="aero_data.h5", model_path="aero_surrogate.pkl"):
    X, y, sigma = load_data(filename)
    model = MultiOutputGPR()
    model.fit(X, y)
    dump(model, model_path)

    scores = model.kfold_cv(X, y, n_splits=5)
    print("K-fold RMSE:")
    for name, score in scores.items():
        print(f"  {name}: {score:.4f}")
    print(f"Surrogate trained and saved to {model_path}")
    return model


if __name__ == "__main__":
    train_gpr()
