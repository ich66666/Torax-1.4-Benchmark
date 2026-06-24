# Metrics

Each episode contributes one row to the summary CSV.  Here is what
each column means and why it matters.

## Survival

| Column      | Type | Meaning                                          |
|-------------|------|--------------------------------------------------|
| `disrupted` | bool | Did the plasma hit a disruption criterion?       |
| `reason`    | str  | If disrupted: which criterion fired              |
| `t_final`   | float| Last simulation time reached (seconds)           |
| `n_ticks`   | int  | Number of control ticks completed                |

A run is considered **survived** if it reached the configured `t_final`
without disruption (`disrupted == False`).  Disruption criteria:

- `q_min < 0.5` — fast-growth current-drive instability.
- `beta_N > 3.0` — Troyon kink boundary.
- `non-finite-state` — TORAX produced a NaN/Inf in q_min or beta_N
  (typically a Q-overshoot blowing up the equilibrium solver).
- `torax-step-failed:*` — TORAX itself threw an exception
  (rare but possible under extreme inputs).

Survival rate over N seeds is the headline metric.  Comparable to
published runs only when N, severity, and t_final all match.

## Plasma operation quality

| Column       | Unit          | Meaning                                          |
|--------------|---------------|--------------------------------------------------|
| `Q_final`    | dimensionless | Fusion gain at episode end                       |
| `Q_max`      | dimensionless | Peak Q reached during episode                    |
| `Q_mean`     | dimensionless | Average Q across all ticks                       |
| `beta_N_max` | Troyon units  | Peak normalized pressure                         |
| `q_min_min`  | dimensionless | Minimum q_min reached (closer to 0.5 = closer to disruption) |
| `H98_mean`   | dimensionless | Average confinement enhancement                  |

Read together these characterize the *operating point* the controller
held the plasma at.  A controller that survived 100/100 by parking
the plasma at Q_mean = 1.5 (L-mode quiescence) is technically a
survivor but not a useful controller.  Healthy values:

- `Q_mean` around 5 (the ITER Hybrid design point).
- `H98_mean` ≥ 1.5 (clean H-mode confinement).
- `q_min_min` ≥ 0.5 with margin (closer to 1 = more comfortable).
- `beta_N_max` ≤ 2.4 (below the Troyon hard limit).

A common pathology is **Q-overshoot during ramp-up**: Q runs up to
20–40 in the first few seconds, blowing past the equilibrium solver's
stability margin and producing a non-finite-state disruption.
Controllers that smooth the ramp-up phase typically score much higher
than those that don't.

## Disturbance accounting

| Column           | Type  | Meaning                                          |
|------------------|-------|--------------------------------------------------|
| `events_total`   | int   | Total adversarial events injected this episode   |
| `events_by_kind` | dict  | Per-event-type breakdown                         |

Event types:

- `ELM` — edge-localized mode crash (energy spike + radiation increase)
- `NBI_dropout` — neutral-beam outage for 1–3 s
- `ECRH_dropout` — electron-cyclotron outage for 1–2 s
- `Z_eff_drift` — random walk in impurity radiation (continuous, not
  counted as discrete event but visible in `z_eff_dist` column)

`events_total` at severity 5.0 typically lands around 250 / 1200 s
(one event every ~5 s on average).  Surviving episodes ride this
out via active feedback; failing episodes are usually cut short by
the very first or second event.

## Wall-time and meta

| Column      | Type  | Meaning                                          |
|-------------|-------|--------------------------------------------------|
| `wall_s`    | float | Wall-clock seconds spent on this episode         |
| `seed`      | int   | Disturbance and policy RNG seed                  |
| `csv`       | str   | Path to the per-tick log file                    |

`wall_s` will vary widely depending on your hardware and whether the
TORAX JIT compile has been amortized across earlier episodes (first
episode of a fresh process pays the compile cost; subsequent ones
share it).

## Per-tick CSV

In addition to the summary, each episode produces a per-tick file
with all 20 columns at every control_dt:

```
step, t, Te_core, Ti_core, ne_core, beta_N, q95, q_min, Q_fusion,
H98, f_bootstrap, fgw, Ip_set, NBI_set, ECRH_set, gas_set,
impurity_set, z_eff_dist, disrupted, reason
```

These are the raw inputs to whatever post-processing or plots you
want to make — control-trajectory inspection, statistics across
seeds, etc.
