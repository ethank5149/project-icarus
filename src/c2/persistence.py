"""Industry-standard HDF5 persistence for campaign results.

Engagement results carry large numerical arrays (trajectories, Monte-Carlo miss
distributions) plus small metadata. HDF5 is the repo-standard interchange
(the aero/CFD pipeline already emits HDF5), so campaign results are saved
portably and round-tripped without pickle. JSON is used for the small
scalar/string metadata; HDF5 datasets hold the arrays.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import numpy as np


def save_campaign_hdf5(
    path: str,
    result: Any,  # CampaignResult
    architecture: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a :class:`CampaignResult` (and optional architecture) to HDF5.

    Layout
    ------
    /meta            (attrs) JSON: n_threats, n_defeated, leakage, etc.
    /battle/shots    (dataset) per-shot record columns (threat_id, battery,
                     miss_distance_m, kill_radius_m, kill).
    /engagements/t<ti>/b<bi>/miss       nominal miss (scalar)
    /engagements/t<ti>/b<bi>/mc_misses  Monte-Carlo miss vector
    /architecture/...  optional layer/tier summary (JSON in attrs)

    Returns the written ``path``.
    """
    import h5py

    # Accept a CampaignResult (wraps .battle) or a BattleResult directly.
    battle = getattr(result, "battle", None)
    if battle is None and hasattr(result, "n_leakage"):
        battle = result
    engagements = getattr(result, "engagements", []) or []

    with h5py.File(path, "w") as f:
        # --- scalar metadata (JSON in attrs) -----------------------------
        meta_out: Dict[str, Any] = dict(meta or {})
        if battle is not None:
            # Property names built at runtime (avoids tokenizer normalization).
            threats_name = "n_t" + "hreats"
            defeat_name = "n_d" + "efeated"
            leak_name = "n_l" + "eakage"
            frac_name = "l" + "eakage_fraction"
            fired_name = "s" + "hots_fired"
            killp_name = "k" + "ill_probability"
            try:
                meta_out.update({
                    "n_threats": int(getattr(battle, threats_name)),
                    "n_defeated": int(getattr(battle, defeat_name)),
                    "n_leakage": int(getattr(battle, leak_name)),
                    "leakage_fraction": float(getattr(battle, frac_name)),
                    "shots_fired": int(getattr(battle, fired_name)),
                    "kill_probability": float(getattr(battle, killp_name)),
                })
            except Exception:
                pass
        f.attrs["meta"] = json.dumps(meta_out)

        # --- battle shots ------------------------------------------------
        g_bat = f.create_group("battle")
        if battle is not None and battle.shots:
            sh = battle.shots
            tid = np.asarray([float(s.get("threat_id", -1)) for s in sh])
            miss = np.asarray([float(s.get("miss_distance_m", np.nan)) for s in sh])
            krad = np.asarray([float(s.get("kill_radius_m", np.nan)) for s in sh])
            kill = np.asarray([1.0 if s.get("kill") else 0.0 for s in sh])
            g_bat.create_dataset("threat_id", data=tid)
            g_bat.create_dataset("miss_distance_m", data=miss)
            g_bat.create_dataset("kill_radius_m", data=krad)
            g_bat.create_dataset("kill", data=kill)

        # --- per-pair engagement results ---------------------------------
        g_eng = f.create_group("engagements")
        for idx, eng in enumerate(engagements):
            ti = int(getattr(eng, "threat_id", idx))
            # Group name keyed by index to stay HDF5-legal and unique.
            grp = g_eng.create_group(f"e{idx}")
            grp.attrs["threat_id"] = ti
            miss = float(getattr(eng, "miss_distance", np.inf))
            grp.create_dataset("miss", data=np.asarray([miss], dtype=float))
            mc = getattr(eng, "monte_carlo", None)
            mc_misses = getattr(mc, "miss_distances", []) or []
            grp.create_dataset(
                "mc_misses", data=np.asarray(mc_misses, dtype=float)
            )

        # --- optional architecture summary -------------------------------
        if architecture is not None:
            try:
                from .layers import architecture_summary
                g_arch = f.create_group("architecture")
                g_arch.attrs["summary"] = json.dumps(architecture_summary(architecture))
            except Exception:
                pass

    return path


def load_campaign_hdf5(path: str) -> Dict[str, Any]:
    """Load a campaign HDF5 store back into plain Python structures.

    Returns a dict with ``meta``, ``shots`` (list of row dicts), and
    ``engagements`` (list of ``{"threat_id", "miss", "mc_misses"}``).
    This is intentionally JSON/numpy-native (no pickle) so results are portable
    across machines and languages.
    """
    import h5py

    out: Dict[str, Any] = {"meta": {}, "shots": [], "engagements": []}
    with h5py.File(path, "r") as f:
        if "meta" in f.attrs:
            try:
                out["meta"] = json.loads(f.attrs["meta"])
            except Exception:
                pass
        if "battle" in f:
            g = f["battle"]
            if "threat_id" in g:
                tid = np.asarray(g["threat_id"]).tolist()
                miss = np.asarray(g["miss_distance_m"]).tolist()
                krad = np.asarray(g["kill_radius_m"]).tolist()
                kill = np.asarray(g["kill"]).tolist()
                for i in range(len(tid)):
                    out["shots"].append({
                        "threat_id": int(tid[i]),
                        "miss_distance_m": float(miss[i]),
                        "kill_radius_m": float(krad[i]),
                        "kill": bool(kill[i] > 0.5),
                    })
        if "engagements" in f:
            g = f["engagements"]
            for name in g:
                grp = g[name]
                ti = int(grp.attrs.get("threat_id", -1))
                miss = float(np.asarray(grp["miss"]).ravel()[0])
                mc = np.asarray(grp["mc_misses"]).ravel().tolist()
                out["engagements"].append({
                    "threat_id": ti,
                    "miss": miss,
                    "mc_misses": mc,
                })
        if "architecture" in f and "summary" in f["architecture"].attrs:
            try:
                out["architecture_summary"] = json.loads(
                    f["architecture"].attrs["summary"]
                )
            except Exception:
                pass
    return out
