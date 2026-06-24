"""Hold baseline policy.

Returns None at every tick → the runner sends no override and the
plasma evolves with whatever setpoints the scenario started with
(an open-loop reference run).

Useful as a control for "did my controller actually help, or did the
plasma just survive on its own?".
"""
from __future__ import annotations
from ..benchmark.plasma_features import PlasmaFeatures
from ..benchmark.actuator import Actuator, default_actuators


class HoldPolicy:
    def __init__(self, seed: int = 0,
                 actuators: list[Actuator] | None = None):
        self.actuators = actuators or default_actuators(seed)
        self.n_decisions = 0

    def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float] | None:
        self.n_decisions += 1
        return None

    def setpoints(self) -> dict[str, float]:
        return {a.name: a.u_current for a in self.actuators}
