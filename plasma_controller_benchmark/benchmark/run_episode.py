"""Single-episode benchmark runner.

A policy is anything implementing:

    class Policy:
        def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float] | None:
            '''Return new setpoints for this tick, or None to hold.'''
        def setpoints(self) -> dict[str, float]:
            '''Return current setpoints (used at tick zero for logging).'''

Setpoint keys must be one of: 'Ip', 'NBI', 'ECRH', 'gas', 'impurity'.
Units are SI (Amps, Watts, particles/s; impurity is dimensionless).

The runner threads each tick through:
    1. read plasma state from TORAX 1.4
    2. ask policy for new setpoints
    3. let hell_disturbance perturb them
    4. push perturbed setpoints back into TORAX
    5. advance TORAX one control_dt
    6. log a row to the per-tick CSV
"""
from __future__ import annotations
import csv
import math
import sys
import time
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pathlib import Path

import h5py  # noqa: F401  (Windows DLL ordering: must precede TORAX)
import numpy as np

from torax import ToraxConfig, set_jax_precision
from torax.experimental import (
    make_step_fn,
    get_initial_state_and_post_processed_outputs,
    TimeVaryingScalarUpdate,
)

# Repo-local imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from scenario import iterhybrid_xhard
from benchmark.plasma_features import PlasmaFeatures
from benchmark.hell_disturbance import HellDisturbance, HellConfig


# TORAX 1.4 config-path → human-readable actuator name
ACTUATOR_PATHS = {
    "Ip":       "profile_conditions.Ip",
    "NBI":      "sources.generic_heat.P_total",
    "ECRH":     "sources.ecrh.P_total",
    "gas":      "sources.gas_puff.S_total",
    "impurity": "sources.impurity_radiation.radiation_multiplier",
}
SCALAR_ACTUATORS = {"impurity"}


PER_TICK_COLUMNS = [
    "step", "t",
    "Te_core", "Ti_core", "ne_core",
    "beta_N", "q95", "q_min", "Q_fusion", "H98", "f_bootstrap", "fgw",
    "Ip_set", "NBI_set", "ECRH_set", "gas_set", "impurity_set",
    "z_eff_dist",
    "disrupted", "reason",
]


def _features(state, ppo, t: float) -> PlasmaFeatures:
    return PlasmaFeatures(
        t       = float(t),
        Te_core = float(state.core_profiles.T_e.value[0]),
        Ti_core = float(state.core_profiles.T_i.value[0]),
        ne_core = float(state.core_profiles.n_e.value[0]) / 1e20,
        beta_N  = float(ppo.beta_N),
        q95     = float(ppo.q95),
        q_min   = float(ppo.q_min),
        Q_fus   = float(ppo.Q_fusion),
        H98     = float(ppo.H98),
        f_BS    = float(ppo.f_bootstrap),
        fGW     = float(ppo.fgw_n_e_line_avg),
    )


def _disrupted(f: PlasmaFeatures) -> tuple[bool, str]:
    if not math.isfinite(f.q_min) or not math.isfinite(f.beta_N):
        return True, "non-finite-state"
    if f.q_min < 0.5:
        return True, "q_min<0.5"
    if f.beta_N > 3.0:
        return True, "beta_N>3.0 (Troyon kink)"
    return False, ""


def _build_overrides(setpoints, t):
    out = {}
    t_arr = np.array([float(t)], dtype=np.float64)
    for name, val in setpoints.items():
        if name.startswith("__"):
            continue
        if name not in ACTUATOR_PATHS:
            continue
        if name in SCALAR_ACTUATORS:
            out[ACTUATOR_PATHS[name]] = float(val)
        else:
            v_arr = np.array([float(val)], dtype=np.float64)
            out[ACTUATOR_PATHS[name]] = TimeVaryingScalarUpdate(
                value=v_arr, time=t_arr,
            )
    return out


def run_episode(
    policy,
    seed:        int,
    t_final:     float = 1200.0,
    control_dt:  float = 1.0,
    out_csv:     Path | None = None,
    verbose:     bool = False,
    hell_cfg:    HellConfig | None = None,
) -> dict:
    """Run one episode of the IterHybrid-XHARD benchmark.

    Args:
        policy:     anything with .step(features, dt) and .setpoints()
        seed:       seed for the hell disturbance RNG (also propagated to
                    any policy that uses random numbers)
        t_final:    simulation duration in seconds (1200 = standard)
        control_dt: control loop period in seconds (1.0 = standard)
        out_csv:    optional per-tick log file
        verbose:    print progress lines every 10 ticks
        hell_cfg:   adversarial cascade configuration; pass the result of
                    benchmark.hell_disturbance.scaled_hell_config(severity)

    Returns:
        Summary dict with t_final, n_ticks, disrupted, Q_mean, etc.
    """
    set_jax_precision()
    cfg_dict = iterhybrid_xhard.get_config() if hasattr(iterhybrid_xhard, 'get_config') \
                                              else iterhybrid_xhard.CONFIG.copy()
    cfg_dict["numerics"]["t_final"] = t_final

    if verbose:
        print(f"Building TORAX 1.4 config + step_fn ...", flush=True)
    t0 = time.time()
    cfg = ToraxConfig.from_dict(cfg_dict)
    step_fn = make_step_fn(cfg)
    state, ppo = get_initial_state_and_post_processed_outputs(step_fn)
    provider = step_fn.runtime_params_provider
    if verbose:
        print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    hell = HellDisturbance(cfg=hell_cfg, seed=seed)

    rows: list[list] = []
    disrupted = False
    reason = ""
    tick = 0
    t_start = time.time()

    while not bool(step_fn.is_done(state.t)):
        t_now = float(state.t)
        f = _features(state, ppo, t_now)
        disr, why = _disrupted(f)

        commanded = policy.setpoints() if policy.setpoints() else {}
        perturbed = hell.step(t_now, commanded)
        z_eff_now = perturbed.get("__z_eff_value", float("nan"))

        rows.append([
            tick, t_now,
            f.Te_core, f.Ti_core, f.ne_core,
            f.beta_N, f.q95, f.q_min, f.Q_fus, f.H98, f.f_BS, f.fGW,
            perturbed.get("Ip", float("nan")),
            perturbed.get("NBI", float("nan")),
            perturbed.get("ECRH", float("nan")),
            perturbed.get("gas", float("nan")),
            perturbed.get("impurity", float("nan")),
            z_eff_now,
            int(disr), why,
        ])

        if verbose and tick % 10 == 0:
            print(f"  t={t_now:>6.1f}  Te={f.Te_core:>5.2f}  "
                  f"bN={f.beta_N:>5.2f}  Q={f.Q_fus:>6.2f}  "
                  f"qmin={f.q_min:>5.2f}  H98={f.H98:>4.2f}  "
                  f"fGW={f.fGW:>4.2f}", flush=True)

        if disr:
            disrupted = True
            reason = why
            break

        sps_new = policy.step(f, control_dt)
        if sps_new is not None:
            final = hell.step(t_now, sps_new)
            provider = provider.update_provider_from_mapping(
                _build_overrides(final, t_now),
            )

        try:
            state, ppo = step_fn.jitted_fixed_time_step(
                dt=control_dt,
                input_state=state,
                previous_post_processed_outputs=ppo,
                runtime_params_overrides=provider,
            )
        except Exception as e:
            disrupted = True
            reason = f"torax-step-failed:{type(e).__name__}"
            break

        tick += 1

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(PER_TICK_COLUMNS)
            w.writerows(rows)

    Q_vals    = [r[8]  for r in rows if math.isfinite(r[8])]
    bN_vals   = [r[5]  for r in rows if math.isfinite(r[5])]
    qmin_vals = [r[7]  for r in rows if math.isfinite(r[7])]
    H98_vals  = [r[9]  for r in rows if math.isfinite(r[9])]

    return dict(
        seed         = seed,
        t_final      = float(rows[-1][1]) if rows else 0.0,
        n_ticks      = len(rows),
        disrupted    = disrupted,
        reason       = reason,
        Q_final      = Q_vals[-1] if Q_vals else float("nan"),
        Q_max        = max(Q_vals) if Q_vals else float("nan"),
        Q_mean       = sum(Q_vals) / len(Q_vals) if Q_vals else float("nan"),
        beta_N_max   = max(bN_vals) if bN_vals else float("nan"),
        q_min_min    = min(qmin_vals) if qmin_vals else float("nan"),
        H98_mean     = sum(H98_vals) / len(H98_vals) if H98_vals else float("nan"),
        events_total   = hell.events_total,
        events_by_kind = dict(hell.events_by_kind),
        wall_s         = time.time() - t_start,
        csv            = str(out_csv) if out_csv else "",
    )
