import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold
from joblib import dump, load


def load_data(filename="aero_data.h5"):
    with h5py.File(filename, "r") as hf:
        X = np.column_stack(
            (hf["mach"][:], hf["alpha"][:], hf["beta"][:], hf["altitude"][:])
        )
        y = np.column_stack((hf["Cd"][:], hf["Cl"][:], hf["Cm"][:]))
        sigma = hf["sigma"][:]
    return X, y, sigma


def build_kernel():
    return C(1.0) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0]) + WhiteKernel(noise_level=0.01)


class MultiOutputGPR:
    def __init__(self):
        self.models = [
            GaussianProcessRegressor(kernel=build_kernel(), n_restarts_optimizer=5),
            GaussianProcessRegressor(kernel=build_kernel(), n_restarts_optimizer=5),
            GaussianProcessRegressor(kernel=build_kernel(), n_restarts_optimizer=5),
        ]
        self.target_names = ["Cd", "Cl", "Cm"]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    def predict(self, X, return_std=False):
        means = []
        stds = []
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


if __name__ == "__main__":
    train_gpr()
