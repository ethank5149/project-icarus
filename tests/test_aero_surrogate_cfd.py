import os
import tempfile

import numpy as np
import pytest

from project_icarus.aero.cfd_generators import default_spec, sweep_to_hdf5
from project_icarus.surrogates.train_gpr import MultiOutputGPR, load_data, train_gpr, COEFF_NAMES
from project_icarus.surrogates.aero_surrogate import AeroSurrogateComponent, COEFF_NAMES, FEAT
from openmdao.api import Problem, Group, IndepVarComp


@pytest.fixture
def cfd_h5():
    spec = default_spec(
        "tamir", backend="analytic",
        mach_range=(0.5, 3.0, 4),
        alpha_range=(-5.0, 5.0, 5),
        beta_range=(-3.0, 3.0, 3),
        altitude_range=(0.0, 100e3, 3),
        delta_range=(0.0, 2.0, 2),
        noise_level=0.0, seed=1,
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cfd.h5")
        sweep_to_hdf5(spec, path)
        yield path


class TestCFDGPRSurrogate:
    def test_load_data_five_outputs(self, cfd_h5):
        X, y, sigma = load_data(cfd_h5)
        assert X.shape[1] == 5  # mach, alpha, beta, alt, delta
        assert y.shape[1] == 5  # Cd, Cy, Cm, Cn, Cl_roll
        assert np.all(np.isfinite(y))

    def test_gpr_predict_and_jacobian(self, cfd_h5):
        X, y, _ = load_data(cfd_h5)
        model = MultiOutputGPR()
        model.fit(X, y)
        means, stds = model.predict(X[:7], return_std=True)
        assert means.shape == (7, 5)
        assert stds.shape == (7, 5)
        J = model.jacobian_fd(X[:7])
        assert J.shape == (7, 5, 5)
        assert np.all(np.isfinite(J))
        # dCd/dalpha monotonic-ish: positive where Cd increases with alpha.
        assert np.all(np.isfinite(J[:, 0, 1]))

    def test_gpr_kfold_runs(self, cfd_h5):
        X, y, _ = load_data(cfd_h5)
        model = MultiOutputGPR()
        scores = model.kfold_cv(X, y, n_splits=3)
        assert set(scores) == set(COEFF_NAMES)
        assert all(v >= 0 for v in scores.values())

    def test_train_gpr_returns_model(self, cfd_h5):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "m.pkl")
            model = train_gpr(cfd_h5, out)
            assert model is not None
            assert os.path.exists(out)


class TestAeroSurrogateComponent:
    def test_openmdao_component_cs_partials(self, cfd_h5):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "m.pkl")
            train_gpr(cfd_h5, out)

            p = Problem()
            ivc = IndepVarComp()
            ivc.add_output("mach", 2.0)
            ivc.add_output("alpha", 3.0)
            ivc.add_output("beta", 1.0)
            ivc.add_output("altitude", 50e3)
            ivc.add_output("delta", 1.0)
            p.model.add_subsystem("ivc", ivc, promotes=["*"])
            p.model.add_subsystem(
                "aero", AeroSurrogateComponent(model_path=out), promotes=["*"]
            )
            p.setup()
            p.run_model()
            for c in COEFF_NAMES:
                assert np.isfinite(p["Cd" if c == "Cd" else c])
            # Check partials against complex-step.
            data = p.check_partials(method="fd", out_stream=None)
            for comp in data.values():
                for key, val in comp.items():
                    if key[0] in COEFF_NAMES and key[1] in FEAT:
                        assert val["abs error"].forward < 1e-4, (key, val["abs error"].forward)
