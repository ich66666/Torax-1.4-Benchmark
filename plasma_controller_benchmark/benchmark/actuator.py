"""Generic actuator model with slew-rate limit + bound clamp + noise.

A policy returns a dict of {actuator_name: target_setpoint} per tick.
The runner pushes each target through the matching Actuator, which
applies physical-realism constraints:
  - slew rate limit  (du/dt cap from the source paper)
  - hard bounds      (low / high)
  - actuator noise   (Gaussian process noise around the slew)

The benchmark ships a 5-actuator default set matching the standard
ITER Hybrid scenario.  Users can also instantiate Actuator directly
for custom configurations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .plasma_features import clamp


@dataclass
class Actuator:
    name:           str
    low:            float
    high:           float
    u_nominal:      float
    u_current:      float
    slew_per_s:     float
    actuator_noise: float = 0.005
    rng:            np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0))

    def drive_to(self, u_target: float, dt: float) -> float:
        """Move toward u_target subject to slew + bounds + noise.
        Returns the new u_current."""
        u_target = clamp(u_target, self.low, self.high)
        u_target += self.rng.normal(
            0.0, self.actuator_noise * self.slew_per_s * dt)
        du_max = self.slew_per_s * dt
        du = clamp(u_target - self.u_current, -du_max, du_max)
        self.u_current = clamp(self.u_current + du, self.low, self.high)
        return self.u_current


def default_actuators(seed: int = 0,
                       emergency_mode: bool = True) -> list[Actuator]:
    """Standard 5-actuator set for the ITER Hybrid scenario.

    emergency_mode = True uses the slew rates from the ITER Disruption
    Mitigation System spec (Lehnen 2015, Hollmann 2015).  emergency_mode
    = False uses the routine-operation slew limits (Snipes 2017).

    Ip stays at 0.2 MA/s either way — ramping Ip changes beta_N through
    the Troyon normalization, so it's the wrong knob for fast burning-Q
    correction.

    Actuator units:
      Ip       in A   (typical operating point 12.5 MA)
      NBI      in W   (full power 33 MW)
      ECRH     in W   (full power 20 MW)
      gas      in particles/s
      impurity is a unitless radiation multiplier (Z_eff fraction)
    """
    rng = np.random.default_rng(seed)
    def _r():
        return np.random.default_rng(rng.integers(0, 2**31 - 1))

    if emergency_mode:
        Ip_slew, NBI_slew, ECRH_slew = 2.0e5, 1.65e7, 2.0e7
        gas_slew, imp_slew = 5.0e22, 2.0
    else:
        Ip_slew, NBI_slew, ECRH_slew = 2.0e5, 3.3e6, 2.0e6
        gas_slew, imp_slew = 3.0e21, 0.5

    return [
        Actuator(name="Ip",
                 low=2.0e6, high=15.0e6,
                 u_nominal=12.5e6, u_current=12.5e6,
                 slew_per_s=Ip_slew, rng=_r()),
        Actuator(name="NBI",
                 low=0.0, high=33.0e6,
                 u_nominal=33.0e6, u_current=33.0e6,
                 slew_per_s=NBI_slew, rng=_r()),
        Actuator(name="ECRH",
                 low=0.0, high=20.0e6,
                 u_nominal=20.0e6, u_current=20.0e6,
                 slew_per_s=ECRH_slew, rng=_r()),
        Actuator(name="gas",
                 low=1.0e20, high=5.0e22,
                 u_nominal=1.0e20, u_current=1.0e20,
                 slew_per_s=gas_slew, rng=_r()),
        Actuator(name="impurity",
                 low=0.0, high=2.0,
                 u_nominal=0.0, u_current=0.0,
                 slew_per_s=imp_slew, rng=_r()),
    ]
