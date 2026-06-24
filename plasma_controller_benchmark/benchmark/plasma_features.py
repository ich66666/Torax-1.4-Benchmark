"""Plasma feature container + standard physics thresholds.

The 11 features in PlasmaFeatures are computed at each control tick by
the runner and passed to the policy.  All units are SI / TORAX-native
(temperatures in keV, fGW dimensionless, etc.).
"""
from __future__ import annotations
import math
from dataclasses import dataclass


# ---- Standard ITER / DEMO thresholds --------------------------------------
# Sources:
#   - Shimada 2007 Nucl. Fusion 47 S1     (q-profile limits, beta_N targets)
#   - Howard 2024 Nucl. Fusion 64 086013  (Q_fus target window)
#   - La Haye 2006 Phys. Plasmas 13 055501 (beta-N hard limit / NTM onset)
#   - de Vries 2011 Nucl. Fusion 51 053018 (Greenwald-fraction limits)

BETA_N_TARGET, BETA_N_PUSH, BETA_N_HARD = 1.5, 1.7, 2.4
NGW_TARGET,    NGW_PUSH,    NGW_HARD    = 0.75, 0.80, 0.95
Q_MIN_TARGET,  Q_MIN_PUSH,  Q_MIN_HARD  = 1.5,  1.35, 1.05
Q95_LOWER,     Q95_TARGET               = 3.0,  4.0
Q_TARGET,      Q_PUSH,      Q_HARD      = 5.0,  8.0, 15.0

TE_TARGET_KEV = 22.0
TI_TARGET_KEV = 18.0


# ---- Convenience helpers --------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(slots=True)
class PlasmaFeatures:
    """One per-tick snapshot of the plasma state, as seen by a controller."""
    t:        float   # simulation time (s)
    Te_core:  float   # core electron temperature (keV)
    Ti_core:  float   # core ion temperature (keV)
    ne_core:  float   # core electron density (10^19 m^-3)
    beta_N:   float   # normalized beta (Troyon units)
    q95:      float   # safety factor at psi=0.95
    q_min:    float   # minimum safety factor (whole profile)
    Q_fus:    float   # fusion gain (P_fus / P_aux)
    H98:      float   # confinement enhancement factor over ITER89-P
    f_BS:     float   # bootstrap current fraction
    fGW:      float   # Greenwald density fraction

    @property
    def risk(self) -> float:
        """Composite [0, 1] danger score from q_min and beta_N margins.
        0 = healthy, 1 = at or beyond hard limit."""
        r_q = max(0.0, (Q_MIN_PUSH - self.q_min) /
                  max(1e-9, Q_MIN_PUSH - Q_MIN_HARD))
        r_b = max(0.0, (self.beta_N - BETA_N_PUSH) /
                  max(1e-9, BETA_N_HARD - BETA_N_PUSH))
        return clamp(max(r_q, r_b), 0.0, 1.0)
