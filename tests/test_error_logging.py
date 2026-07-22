"""Regression tests for bare-except-to-logging cleanup."""
import logging

import numpy as np
import pytest

from project_icarus.dynamics.eom_6dof import EOM6DOF
from project_icarus.c2.campaign import _serialize_target
from project_icarus.c2.persistence import save_campaign_hdf5, load_campaign_hdf5
from project_icarus.scenarios.target_factory import BallisticScenario


class TestErrorLogging:
    def test_campaign_serialize_unknown_raises(self):
        fake = object()
        with pytest.raises(ValueError, match="Cannot serialize target kind"):
            _serialize_target(fake)

    def test_persistence_meta_fallback_logs(self, caplog, tmp_path):
        class _BadBattle:
            shots = []

            @property
            def n_threats(self):
                raise RuntimeError("boom")

        bad_result = BallisticScenario(r0=np.array([6.4e6, 0.0, 0.0]),
                                       v0=np.array([0.0, 0.0, 7000.0]))
        bad_result.battle = _BadBattle()

        path = str(tmp_path / "meta_fallback.h5")
        with caplog.at_level(logging.DEBUG):
            save_campaign_hdf5(path, bad_result)
        assert any("Battle metadata extraction failed" in r.getMessage() for r in caplog.records)

    def test_persistence_load_bad_meta_logs(self, caplog, tmp_path):
        import h5py

        path = str(tmp_path / "bad_meta.h5")
        with h5py.File(path, "w") as f:
            f.attrs["meta"] = b"not-json"
        with caplog.at_level(logging.DEBUG):
            load_campaign_hdf5(path)
        assert any("Meta JSON parse failed" in r.getMessage() for r in caplog.records)
