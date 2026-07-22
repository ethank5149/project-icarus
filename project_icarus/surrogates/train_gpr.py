import h5py
import os
from typing import Optional
import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Product, Sum
from sklearn.model_selection import KFold
from joblib import dump, load


def _get_rbf_params(kernel):
    """Extract RBF length_scale and ConstantKernel scale from a fitted kernel."""
    def _find_product(k):
        if isinstance(k, Product):
            return k
        if hasattr(k, "k1"):
            return _find_product(k.k1)
        if hasattr(k, "k2"):
            return _find_product(k.k2)
        return None
    prod = _find_product(kernel)
    if prod is None:
        raise ValueError("No Product(ConstantKernel, RBF) found in kernel")
    rbf = prod.k2
    constant = prod.k1.constant_value
    return rbf.length_scale, constant


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

    def predict_cupy(self, X, return_std=False):
        """GPU-accelerated batch prediction using CuPy.

        Transfers training data to GPU once, then evaluates the RBF kernel
        on the GPU for large batches.  Falls back to CPU if CuPy is unavailable
        or if the batch is too small to amortise the transfer cost.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
        return_std : bool

        Returns
        -------
        means : ndarray, shape (n_query, n_coeff)
        stds : ndarray, shape (n_query, n_coeff)  (if return_std)
        """
        try:
            import cupy as cp
        except ImportError:
            return self.predict(X, return_std=return_std)

        X = np.asarray(X, dtype=float)
        n_query = X.shape[0]
        if n_query < 64:
            return self.predict(X, return_std=return_std)

        X_train = self.models[0].X_train_
        n_train = X_train.shape[0]
        n_coeff = N_COEFF

        X_cp = cp.asarray(X)
        X_train_cp = cp.asarray(X_train)

        means_cp = cp.empty((n_query, n_coeff), dtype=float)
        stds_cp = cp.empty((n_query, n_coeff), dtype=float) if return_std else None

        for i, model in enumerate(self.models):
            ls, constant = _get_rbf_params(model.kernel_)
            ls_cp = cp.asarray(ls, dtype=float)
            y_std = getattr(model, "_y_train_std", 1.0)
            alpha_cp = cp.asarray(model.alpha_, dtype=float)

            K = constant**2 * cp.exp(
                -0.5 * cp.sum((X_cp[:, None, :] - X_train_cp[None, :, :])**2 / ls_cp**2, axis=2)
            )
            mean_cp = K @ alpha_cp
            means_cp[:, i] = mean_cp * y_std

            if return_std:
                K_diag = cp.sum(K**2, axis=1)
                L = cp.linalg.cholesky(model.kernel_(X_train, X_train) + 1e-6 * cp.eye(n_train))
                L_inv = cp.linalg.inv(L)
                K_train_inv = L_inv.T @ L_inv
                K_train_diag = cp.diag(cp.asarray(model.kernel_(X_train, X_train)))
                var_cp = K_train_diag - K_diag + model._y_train_std**2
                stds_cp[:, i] = cp.sqrt(cp.maximum(var_cp, 0.0)) * y_std

        means = cp.asnumpy(means_cp)
        if return_std:
            return means, cp.asnumpy(stds_cp)
        return means

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

    def analytical_jacobian(self, X):
        """Analytical Jacobian d(coeffs)/d(inputs) via the RBF kernel derivative.

        Returns (N, n_coeff, n_features). This is exact (up to the linear-algebra
        solve) and typically 10–100× faster than ``jacobian_fd`` because it
        requires no additional GPR evaluations.
        """
        X_train = self.models[0].X_train_
        J = np.zeros((X.shape[0], N_COEFF, X.shape[1]), dtype=float)
        for i, model in enumerate(self.models):
            ls, _ = _get_rbf_params(model.kernel_)
            y_std = getattr(model, "_y_train_std", 1.0)
            alpha = model.alpha_

            K = model.kernel_(X, X_train)
            diff = X_train[None, :, :] - X[:, None, :]
            dK = K[:, :, None] * diff / (ls ** 2)
            J[:, i, :] = y_std * np.einsum("k,qkn->qn", alpha, dK)
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


def default_model_path(vehicle_key: str, base_dir: str = "reference") -> str:
    return os.path.join(base_dir, "vehicles", vehicle_key, "surrogate.pkl")


def default_h5_path(vehicle_key: str, base_dir: str = "reference") -> str:
    return os.path.join(base_dir, "vehicles", vehicle_key, "aero.h5")


def load_vehicle_gpr(vehicle_key: str, base_dir: str = "reference"):
    """Load a per-vehicle GPR surrogate. Falls back to legacy locations."""
    import os

    # New per-vehicle location
    path = default_model_path(vehicle_key, base_dir)
    if os.path.exists(path):
        return load(path)

    # Legacy flat fallback
    legacy = os.path.join(base_dir, "surrogates", f"aero_surrogate_{vehicle_key}.pkl")
    if os.path.exists(legacy):
        return load(legacy)

    # Generic fallback
    generic = os.path.join(base_dir, "surrogates", "aero_surrogate_generic.pkl")
    if os.path.exists(generic):
        return load(generic)

    # Root fallback
    if os.path.exists("aero_surrogate.pkl"):
        return load("aero_surrogate.pkl")

    raise FileNotFoundError(
        f"No surrogate found for '{vehicle_key}'. "
        "Expected one of:\n"
        f"  {path}\n"
        f"  {legacy}\n"
        f"  {generic}\n"
        f"  aero_surrogate.pkl"
    )


def train_vehicle_gpr(
    vehicle_key: str,
    h5_path: Optional[str] = None,
    base_dir: str = "reference",
    model_path: Optional[str] = None,
):
    """Train and save a per-vehicle GPR surrogate.

    Parameters
    ----------
    vehicle_key : str
        Vehicle identifier (matches VEHICLE_PRESETS key).
    h5_path : str, optional
        Path to the per-vehicle HDF5 sweep data. Defaults to
        ``<base_dir>/vehicles/<vehicle_key>/aero.h5``.
    base_dir : str
        Root reference directory. Defaults to ``"reference"``.
    model_path : str, optional
        Explicit output path. Defaults to ``<base_dir>/vehicles/<vehicle_key>/surrogate.pkl``.

    Returns
    -------
    MultiOutputGPR
        Trained surrogate model.
    """
    import os
    if h5_path is None:
        h5_path = default_h5_path(vehicle_key, base_dir)
    if model_path is None:
        model_path = default_model_path(vehicle_key, base_dir)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    X, y, sigma = load_data(h5_path)
    model = MultiOutputGPR()
    model.fit(X, y)
    dump(model, model_path)
    scores = model.kfold_cv(X, y, n_splits=5)
    print(f"Vehicle '{vehicle_key}' K-fold RMSE:")
    for name, score in scores.items():
        print(f"  {name}: {score:.4f}")
    print(f"Surrogate trained and saved to {model_path}")
    return model


if __name__ == "__main__":
    train_gpr()
