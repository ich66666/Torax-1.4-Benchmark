# Policy interface

To plug a controller into the benchmark, write a class that exposes
two methods.  That's the whole API surface.

```python
class MyPolicy:
    def __init__(self, seed: int = 0):
        """Seed for any internal RNG.  Use it for reproducibility."""

    def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float] | None:
        """Called once per control tick.

        Args:
            f:   the current plasma state (see plasma_features.py)
            dt:  the control loop period in seconds (= --control-dt)
        Returns:
            A dict mapping actuator names to target setpoints, or
            None to hold the current setpoints unchanged.
        """

    def setpoints(self) -> dict[str, float]:
        """Return the current setpoints (after the most recent step()
        was applied).  The runner reads this each tick before calling
        step() so the disturbance injector knows what was commanded.
        """
```

## Actuator names and units

The 5 actuators the scenario understands:

| Name       | Unit             | Operating range                       |
|------------|------------------|---------------------------------------|
| `Ip`       | Amperes          | 2.0e6 .. 15.0e6  (nominal 12.5e6)     |
| `NBI`      | Watts            | 0.0  .. 33.0e6   (full power 33 MW)   |
| `ECRH`     | Watts            | 0.0  .. 20.0e6   (full power 20 MW)   |
| `gas`      | particles/s      | 1.0e20 .. 5.0e22                      |
| `impurity` | dimensionless    | 0.0  .. 2.0  (Z_eff radiation factor) |

You don't have to drive all five — any key you omit from the returned
dict will hold its previous value.

## PlasmaFeatures fields

See [`benchmark/plasma_features.py`](../benchmark/plasma_features.py)
for the dataclass.  Eleven fields per tick:

| Field    | Unit          | Notes                                       |
|----------|---------------|---------------------------------------------|
| `t`      | seconds       | Simulation time since episode start         |
| `Te_core`| keV           | Core electron temperature                   |
| `Ti_core`| keV           | Core ion temperature                        |
| `ne_core`| 10^20 m^-3    | Core electron density                       |
| `beta_N` | Troyon units  | Normalized plasma pressure                  |
| `q95`    | dimensionless | Safety factor at psi=0.95                   |
| `q_min`  | dimensionless | Minimum q over the profile (< 0.5 disrupts) |
| `Q_fus`  | dimensionless | Fusion gain P_fus / P_aux                   |
| `H98`    | dimensionless | Confinement enhancement vs ITER89-P         |
| `f_BS`   | dimensionless | Bootstrap current fraction                  |
| `fGW`    | dimensionless | Greenwald density fraction (> 0.95 hard)    |

Plus a derived property:

| Property | Range  | Notes                                       |
|----------|--------|---------------------------------------------|
| `f.risk` | [0, 1] | 0 = healthy, 1 = at q_min or beta_N hard limit |

## How the runner uses your policy

For each tick of the simulation:

1. Reads plasma state from TORAX → `PlasmaFeatures f`.
2. Calls `f.disrupted?` — if yes, exits the episode early.
3. Reads your current setpoints via `policy.setpoints()`.
4. Perturbs them via the hell_disturbance injector.
5. Logs a row (commanded + perturbed values).
6. Calls `policy.step(f, dt)` to get new setpoints.
7. Pushes the perturbed-then-policy setpoints back into TORAX.
8. Advances TORAX one `control_dt`.
9. Repeats until `t >= t_final` or disruption.

If `policy.step()` returns None, step 6 is skipped and the previous
setpoints remain.  This is how the `HoldPolicy` baseline implements
open-loop control.

## Common patterns

### Using the actuator helper

Most policies want slew-rate-limited, bound-clamped output.  Use the
shipped `Actuator` class:

```python
from benchmark.actuator import Actuator, default_actuators

class MyPolicy:
    def __init__(self, seed=0):
        self.actuators = default_actuators(seed)

    def step(self, f, dt):
        # compute desired setpoint for each actuator
        targets = self.compute_targets(f)  # your logic
        out = {}
        for a in self.actuators:
            out[a.name] = a.drive_to(targets[a.name], dt)
        return out

    def setpoints(self):
        return {a.name: a.u_current for a in self.actuators}
```

`Actuator.drive_to(u_target, dt)` enforces slew, bounds, and adds a
small noise term — same machinery used internally by the baselines.

### Pure setpoint output

If you want to bypass the slew-limit machinery and just write raw
setpoints to TORAX:

```python
class MyDirectPolicy:
    def __init__(self, seed=0):
        self._sps = {"Ip": 12.5e6, "NBI": 33.0e6, "ECRH": 20.0e6,
                     "gas": 1.0e20, "impurity": 0.0}
    def step(self, f, dt):
        # Just slam new values in.  TORAX accepts whatever you send.
        return self._sps
    def setpoints(self):
        return self._sps
```

The hell_disturbance injector will perturb your output before it
reaches TORAX, regardless of how you compute it.
