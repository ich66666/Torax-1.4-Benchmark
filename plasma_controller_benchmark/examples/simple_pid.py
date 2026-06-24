"""Example controller: simple per-actuator PID against fixed targets.

This is NOT a strong controller — it is a TEMPLATE that shows how to
plug your own algorithm into the benchmark.  Use this as a starting
point and replace `compute_correction()` with whatever method you
want to test.

To run:
    python -m benchmark.multi_seed --policy examples.simple_pid:SimplePID \
        --seeds 10 --severity 5.0 --t-final 1200
"""
from __future__ import annotations
import numpy as np

from benchmark.plasma_features import (
    PlasmaFeatures, clamp,
    BETA_N_TARGET, Q_TARGET, NGW_TARGET, Q_MIN_TARGET,
)
from benchmark.actuator import Actuator, default_actuators


class SimplePID:
    """Minimum-viable example: feedback per actuator on Q_fus and beta_N."""

    def __init__(self, seed: int = 0,
                 actuators: list[Actuator] | None = None):
        self.actuators = actuators or default_actuators(seed)
        self.rng = np.random.default_rng(seed)
        # Integral state for each actuator
        self.integ = {a.name: 0.0 for a in self.actuators}
        self.prev_Q  = Q_TARGET
        self.prev_bN = BETA_N_TARGET

    def compute_correction(self, f: PlasmaFeatures) -> dict[str, float]:
        """Return multiplicative delta per actuator (range ~ [-0.1, +0.1])."""
        # Q error: too low → push heating up.  Too high → cut heating.
        eQ  = (f.Q_fus - Q_TARGET) / max(1.0, Q_TARGET)
        # beta_N error: positive = exceeding target.
        ebN = f.beta_N - BETA_N_TARGET
        # Density error: positive = above Greenwald target.
        efGW = f.fGW - NGW_TARGET
        # q_min margin: negative = approaching disruption.
        emqmin = Q_MIN_TARGET - f.q_min

        delta = {
            # NBI: cut on too-high beta_N or Q_fus, boost on low Te
            "NBI":  clamp(-0.6 * eQ - 0.4 * ebN, -0.05, +0.05),
            # ECRH: same envelope as NBI but slightly more conservative
            "ECRH": clamp(-0.4 * eQ - 0.3 * ebN - 0.2 * emqmin, -0.05, +0.05),
            # Gas puff: push up on low fGW, cut on density excursion
            "gas":  clamp(-0.5 * efGW + 0.2 * (1.0 - f.H98), -0.05, +0.05),
            # Impurity: ramp up to absorb Q overshoot
            "impurity": clamp(+0.5 * eQ + 0.3 * ebN, -0.05, +0.05),
            # Ip: leave alone — Ip changes beta_N through Troyon normalization
            "Ip":  0.0,
        }
        return delta

    def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float]:
        delta = self.compute_correction(f)
        out = {}
        for a in self.actuators:
            d = delta.get(a.name, 0.0)
            target = a.u_current * (1.0 + d)
            # apply through the actuator's slew + bounds + noise
            out[a.name] = a.drive_to(target, dt)
        return out

    def setpoints(self) -> dict[str, float]:
        return {a.name: a.u_current for a in self.actuators}
