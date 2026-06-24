# Plasma Controller Benchmark — IterHybrid-XHARD on TORAX 1.4

An adversarial control-theory benchmark for tokamak plasma stabilization.
Plug in your own controller (PID, RL agent, MPC, anything that maps
plasma features → actuator setpoints) and see how many of N seeds it
keeps alive for 1200 simulated seconds under a hostile disturbance
cascade — ELM crashes, NBI/ECRH dropouts, gas-puff flow noise, Z_eff
drift, all tuned far above any documented ITER worst case.

The benchmark is intentionally **harder than the physics** of any real
fusion reactor.  It is a stress test for controllers, not a fidelity
simulation.

Created by Bernd Westrick (2026).

## What's in the box

```
plasma_controller_benchmark/
├── scenario/
│   └── iterhybrid_xhard.py     flat-top initial state, post-ramp,
│                               high-beta high-temperature
├── benchmark/
│   ├── plasma_features.py      11-feature plasma state container
│   ├── actuator.py             5-actuator slew/bounds/noise model
│   ├── hell_disturbance.py     adversarial disturbance injector
│   ├── run_episode.py          single-episode runner
│   └── multi_seed.py           multi-seed sweep + CLI
├── baselines/
│   ├── hold.py                 open-loop (no control) reference
│   └── random_policy.py        uniform-random within slew bounds
├── examples/
│   └── simple_pid.py           template for writing your own controller
└── docs/
    ├── POLICY_INTERFACE.md     what your controller must implement
    └── METRICS.md              what each summary number means
```

## Requirements

- Python ≥ 3.10
- TORAX 1.4.0  (Google DeepMind, see [Citations](#citations))
- JAX 0.10.x   (TORAX dependency)
- NumPy, SciPy, h5py

Install:

```bash
pip install -r requirements.txt
```

`h5py` must be installed before TORAX on Windows (DLL load-order
issue).  The runner enforces this with an explicit early import.

## Quick start

Run the two baseline policies on 10 seeds at severity 5.0:

```bash
python -m benchmark.multi_seed --policy baselines.hold:HoldPolicy \
    --seeds 10 --severity 5.0 --t-final 1200 --out-dir results/hold

python -m benchmark.multi_seed --policy baselines.random_policy:RandomPolicy \
    --seeds 10 --severity 5.0 --t-final 1200 --out-dir results/random
```

You will see one row per seed as it finishes:

```
=== HoldPolicy XHARD sweep  severity=5.0  t_final=1200.0  seeds=10 ===
seed  surv   t_end  Q_mean  q_min   H98  events  wall_s
   0   YES  1199.0    1.23  0.612  1.10    2261    25.3
   1    no    47.0    0.94  0.498  0.95     112     1.8
   ...
```

A summary CSV is written to `<out-dir>/<policy>_summary.csv`.

## Plug in your own controller

Implement a class with two methods:

```python
from benchmark.plasma_features import PlasmaFeatures

class MyController:
    def __init__(self, seed: int = 0):
        ...
    def step(self, f: PlasmaFeatures, dt: float) -> dict[str, float] | None:
        """Return new setpoints, or None to hold the current ones."""
        return {"Ip": ..., "NBI": ..., "ECRH": ..., "gas": ..., "impurity": ...}
    def setpoints(self) -> dict[str, float]:
        """Return current setpoints (used by the runner for logging)."""
```

Then run it:

```bash
python -m benchmark.multi_seed \
    --policy my_module:MyController \
    --seeds 100 --severity 5.0 --t-final 1200
```

See [examples/simple_pid.py](examples/simple_pid.py) for a starting
template.  See [docs/POLICY_INTERFACE.md](docs/POLICY_INTERFACE.md)
for the full contract.

## Benchmark parameters

| Flag           | Default | Meaning                                              |
|----------------|---------|------------------------------------------------------|
| `--seeds`      | 10      | Number of independent episodes                       |
| `--severity`   | 5.0     | Disturbance amplitude multiplier (1.0 = baseline)    |
| `--t-final`    | 1200.0  | Episode duration in seconds (~20 min plasma)         |
| `--control-dt` | 1.0     | Control loop period in seconds                       |

The published benchmark point is **severity 5.0, t_final 1200 s, 100
seeds**.  Sub-runs (10–30 seeds, t_final 60–150 s) are useful for
debugging but the survival rate doesn't stabilize until ~50 seeds.

## Metrics

Each episode contributes one row to the summary CSV:

| Column       | Meaning                                              |
|--------------|------------------------------------------------------|
| `disrupted`  | True if the plasma hit a disruption criterion        |
| `t_final`    | Simulation seconds completed before exit             |
| `Q_mean`     | Average fusion gain Q_fus across the episode         |
| `q_min_min`  | Minimum value of q_min reached (disruption < 0.5)    |
| `H98_mean`   | Average H_98 confinement enhancement factor          |
| `events_total` | Number of injected disturbance events              |

See [docs/METRICS.md](docs/METRICS.md) for the full definitions and the
plasma physics behind each.

## Citations

If you publish results obtained with this benchmark, please cite:

- **This benchmark:** Westrick, B. (2026). *IterHybrid-XHARD: An
  adversarial control benchmark for tokamak plasma stabilization on
  TORAX 1.4.*  Software, MIT license.
- **TORAX simulator:** Citrin, J. and Google DeepMind (2024).
  *TORAX — Tokamak transport simulator in JAX.*
  <https://github.com/google-deepmind/torax>.  The original benchmark
  predecessor was developed against TORAX 1.0; the current version
  uses TORAX 1.4.x and the step-level API
  (`torax.experimental.make_step_fn`).
- **Hybrid scenario:** Polevoi, A. R. et al. (2005). *ITER Scenario 3
  Hybrid, Q ≈ 5–8.*  Nucl. Fusion 45, 1451.
- **Hybrid mode in DEMO conditions:** van Mulders, S. et al. (2021).
  *Hybrid mode in DEMO conditions.*  Nucl. Fusion 61, 086019.
- **Plasma stability thresholds** (Troyon limit, q_min margin, Greenwald
  fraction) follow Shimada 2007 (Nucl. Fusion 47, S1), La Haye 2006
  (Phys. Plasmas 13, 055501), and de Vries 2011 (Nucl. Fusion 51,
  053018).
- **ITER Disruption Mitigation System slew rates** follow Snipes 2017
  (Fusion Eng. Des.), Lehnen 2015 (J. Nucl. Mater.), and Hollmann 2015
  (Phys. Plasmas).

## License

MIT — see [LICENSE](LICENSE).  Copyright (c) 2025-2026 Bernd Westrick.

The scenario file (`scenario/iterhybrid_xhard.py`) and parts of the
disturbance model are derived from the author's earlier work on the
gymtorax companion (TORAX 1.0) and were forward-ported to TORAX 1.4.x
for this release.
