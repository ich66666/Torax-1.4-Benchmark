"""Adversarial disturbance injector for the IterHybrid-XHARD benchmark.

Per-tick stochastic process that simulates a tokamak in fault-cascade mode:
ELM crashes, NBI/ECRH dropouts, gas-puff flow noise, Z_eff random walk.
Disturbance probabilities and amplitudes are tuned far above any
documented ITER worst-case; this is an adversarial control-theory stress
test, not a fidelity simulation.

Each `step(t, current_setpoints, rng)` returns a dict of override patches
that get merged into whatever the policy commanded.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class HellConfig:
    elm_period_s:       float = 5.0
    elm_wp_drop_frac:   float = 0.15
    elm_radiation_spike: float = 3.0

    nbi_dropout_prob:   float = 0.20
    nbi_dropout_min_s:  float = 1.0
    nbi_dropout_max_s:  float = 3.0

    ecrh_dropout_prob:  float = 0.15
    ecrh_dropout_min_s: float = 1.0
    ecrh_dropout_max_s: float = 2.0

    zeff_drift_sigma:   float = 0.15
    zeff_min:           float = 1.5
    zeff_max:           float = 4.0

    gas_noise_factor:   float = 5.0

    enable_elm:         bool = True
    enable_nbi_dropout: bool = True
    enable_ecrh_dropout: bool = True
    enable_zeff_drift:  bool = True
    enable_gas_noise:   bool = True


def scaled_hell_config(severity: float = 1.0) -> HellConfig:
    """Scale every stochastic disturbance parameter by `severity`.

    severity = 1.0 returns the standard XHARD cascade (already ~100x ITER
    worst-case for NBI, ~30x for ECRH). severity = 1.5 -> nominally "150x"
    cascade, 2.0 -> "200x", etc. Probabilities are clamped to <= 0.95 so
    there's always some chance the actuator is online.
    """
    return HellConfig(
        elm_period_s        = 5.0 / max(severity, 1e-3),
        elm_wp_drop_frac    = min(0.95, 0.15 * severity),
        elm_radiation_spike = 3.0 * severity,
        nbi_dropout_prob    = min(0.95, 0.20 * severity),
        ecrh_dropout_prob   = min(0.95, 0.15 * severity),
        zeff_drift_sigma    = 0.15 * severity,
        gas_noise_factor    = 5.0 * severity,
    )


class HellDisturbance:
    """Stateful adversarial disturbance generator. One instance per episode."""

    def __init__(self, cfg: HellConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or HellConfig()
        self.rng = np.random.default_rng(seed)

        self._next_elm_t: float = 0.0
        self._nbi_blackout_until: float = -1.0
        self._ecrh_blackout_until: float = -1.0
        self._z_eff: float = 2.0
        self.event_log: list[tuple[float, str]] = []

    def step(self, t: float, setpoints: Dict[str, float]) -> Dict[str, float]:
        """Return modified setpoints + extra non-actuator overrides
        (impurity for ELM, Z_eff for drift)."""
        out = dict(setpoints)

        if self.cfg.enable_nbi_dropout:
            if t >= self._nbi_blackout_until and \
               self.rng.random() < self.cfg.nbi_dropout_prob:
                dur = self.rng.uniform(self.cfg.nbi_dropout_min_s,
                                        self.cfg.nbi_dropout_max_s)
                self._nbi_blackout_until = t + dur
                self.event_log.append((t, f"NBI_dropout({dur:.1f}s)"))
            if t < self._nbi_blackout_until:
                out["NBI"] = 0.0

        if self.cfg.enable_ecrh_dropout:
            if t >= self._ecrh_blackout_until and \
               self.rng.random() < self.cfg.ecrh_dropout_prob:
                dur = self.rng.uniform(self.cfg.ecrh_dropout_min_s,
                                        self.cfg.ecrh_dropout_max_s)
                self._ecrh_blackout_until = t + dur
                self.event_log.append((t, f"ECRH_dropout({dur:.1f}s)"))
            if t < self._ecrh_blackout_until:
                out["ECRH"] = 0.0

        if self.cfg.enable_gas_noise:
            gas_base = out.get("gas", 1.0e20)
            noise = self.rng.normal(0.0, self.cfg.gas_noise_factor) * gas_base
            out["gas"] = max(1.0e19, gas_base + noise)

        if self.cfg.enable_elm and t >= self._next_elm_t:
            out["impurity"] = self.cfg.elm_radiation_spike
            self._next_elm_t = t + self.cfg.elm_period_s
            self.event_log.append((t, f"ELM(rad x{self.cfg.elm_radiation_spike})"))

        if self.cfg.enable_zeff_drift:
            self._z_eff += self.rng.normal(0.0, self.cfg.zeff_drift_sigma)
            self._z_eff = float(np.clip(self._z_eff,
                                         self.cfg.zeff_min,
                                         self.cfg.zeff_max))
            out["__z_eff_value"] = self._z_eff

        return out

    def summary(self) -> dict:
        kinds = {}
        for _, ev in self.event_log:
            tag = ev.split("(")[0]
            kinds[tag] = kinds.get(tag, 0) + 1
        return {"events_total": len(self.event_log), "events_by_kind": kinds,
                "final_z_eff": self._z_eff}
