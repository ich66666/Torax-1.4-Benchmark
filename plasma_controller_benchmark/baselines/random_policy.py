"""Random baseline policy.

Per-tick uniform-random step within the slew bounds of each actuator.
This is the lower bound of "any sensible control" — if your controller
can't beat this, it's not doing anything.

Documented to collapse the plasma in ≤ 5 seconds on IterHybrid-XHARD at
severity ≥ 1.0 across all 100 seeds tested.
"""
from __future__ import annotations
import numpy as np

from ..benchmark.plasma_features import PlasmaFeatures, clamp
from ..benchmark.actuator import Actuator, default_actuators


class RandomPolicy:
    def __init__(self, seed: int = 0,
                 actuators: list[Actuator] | None = None):
        self.actuators = actuators or default_actuators(seed)
        self.rng = np.random.default_rng(seed)
        self.n_decisions = 0

    def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float]:
        out = {}
        for a in self.actuators:
            du = self.rng.uniform(-a.slew_per_s * dt, +a.slew_per_s * dt)
            a.u_current = clamp(a.u_current + du, a.low, a.high)
            out[a.name] = a.u_current
            self.n_decisions += 1
        return out

    def setpoints(self) -> dict[str, float]:
        return {a.name: a.u_current for a in self.actuators}
