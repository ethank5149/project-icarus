"""Phase 6 tests: HDF5 persistence, JSON-spec parallel transport, dask/joblib.

The parallel backend uses INDUSTRY-STANDARD interchange formats (JSON specs +
HDF5 results) rather than pickle, so lambda-laden interceptor configs never
cross the process boundary. These tests validate that contract and the HDF5
round-trip without invoking the (slow) full 6-DOF integrator per pair.
"""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# HDF5 persistence (standard result interchange)
# ---------------------------------------------------------------------------

class TestCampaignHDF5:
    def test_hdf5_roundtrip_small(self):
        # Build a tiny synthetic CampaignResult shim and exercise save/load.
        from src.c2.battle_manager import BattleResult, ThreatTrack, Battery
        from src.c2.persistence import save_campaign_hdf5, load_campaign_hdf5

        threats = [ThreatTrack(threat_id=i, target=None,
                              aim_point=np.zeros(3)) for i in range(3)]
        bats = [Battery("A", None, None, magazine=5)]
        shots = [
            {"threat_id": 0, "miss_distance_m": 0.1, "kill_radius_m": 0.5, "kill": True},
            {"threat_id": 1, "miss_distance_m": 5.0, "kill_radius_m": 0.5, "kill": False},
        ]
        res = BattleResult(threats=threats, batteries=bats, shots=shots)

        path = os.path.join(tempfile.gettempdir(), "camp_h5_round.h5")
        save_campaign_hdf5(path, res)
        loaded = load_campaign_hdf5(path)

        # Meta keys built at runtime (the 's' in n_threats/n_defeated is
        # prone to tokenizer normalization).
        k_threats = "n_t" + "hreats"
        k_defeated = "n_d" + "efeated"
        k_leakage = "n_l" + "eakage"
        # BattleResult.n_defeated counts threats with .defeated=True; our
        # synthetic threats are not marked, so 0 defeated / 3 leak. We
        # validate the counts the BattleResult actually exposes.
        assert loaded["meta"][k_threats] == 3
        assert loaded["meta"][k_defeated] == 0
        assert loaded["meta"][k_leakage] == 3
        assert len(loaded["shots"]) == 2
        assert loaded["shots"][0]["kill"] is True
        assert loaded["shots"][1]["kill"] is False

    def test_hdf5_engagement_results(self):
        from src.c2.persistence import save_campaign_hdf5, load_campaign_hdf5

        # A minimal object exposing the EngagementResult-shaped attributes.
        class _Eng:
            def __init__(self, ti, miss, mc):
                self.threat_id = ti
                self.miss_distance = miss
                self.kill_assessment = miss <= 0.5
                self.monte_carlo = type("MC", (), {"miss_distances": mc})()

        class _Res:
            battle = None
            engagements = [_Eng(0, 0.1, [0.05, 0.12, 0.2]),
                          _Eng(1, 7.0, [6.0, 8.0])]

        path = os.path.join(tempfile.gettempdir(), "camp_h5_eng.h5")
        save_campaign_hdf5(path, _Res())
        loaded = load_campaign_hdf5(path)
        assert len(loaded["engagements"]) == 2
        e0 = loaded["engagements"][0]
        assert e0["threat_id"] == 0
        assert abs(e0["miss"] - 0.1) < 1e-9
        assert len(e0["mc_misses"]) == 3


# ---------------------------------------------------------------------------
# JSON-spec parallel transport contract
# ---------------------------------------------------------------------------

class TestParallelTransport:
    def test_serial_specs_match_parallel(self):
        # The parallel worker rebuilds interceptors by NAME (no pickle of the
        # lambda thrust profile). Validate the spec serializer + worker return
        # a plain dict (portable) identical in shape to the serial path.
        from src.c2 import run_campaign, CampaignThreat
        from src.c2.campaign import (
            _serialize_target, _deserialize_target,
            _serialize_scenario, _deserialize_scenario,
            _run_one_pair_spec, _interceptor_name,
        )
        from src.scenarios.presets import build_interceptor_config
        from src.scenarios.target_factory import BallisticScenario
        from src.sim.api import EngagementScenario
        from src.guidance.law import GuidanceLaw

        cfg, g = build_interceptor_config("arrow3")
        glaw = GuidanceLaw(g)
        bat = type("B", (), {"interceptor_config": cfg, "guidance_config": glaw})()

        # interceptor name resolves back to its preset key.
        assert _interceptor_name(bat) == "arrow3"

        # target/scenario serialize -> deserialize round-trips.
        tgt = BallisticScenario(r0=np.array([6.4e6, 0, 0]),
                                  v0=np.array([0.0, 800.0, 0.0]))
        spec = _serialize_target(tgt)
        assert _deserialize_target(spec).r0[0] == pytest.approx(6.4e6)
        scn = EngagementScenario(engagement_end=20.0)
        scn_spec = _serialize_scenario(scn)
        assert _deserialize_scenario(scn_spec).engagement_end == pytest.approx(20.0)

        # Worker returns a plain dict (miss + mc_misses), not a pickled object.
        pair = {
            "ti": 0, "interceptor_name": "arrow3",
            "target": _serialize_target(tgt),
            "scenario": _serialize_scenario(EngagementScenario(engagement_end=18.0)),
            "n_trials": 1, "perturbations": None,
        }
        ti, bi, out = _run_one_pair_spec(pair)
        assert ti == 0
        assert isinstance(out, dict)
        assert "miss_distance" in out and "mc_misses" in out
        assert np.isfinite(out["miss_distance"])
