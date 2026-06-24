"""Multi-seed sweep runner.

Runs N seeds against a chosen policy and writes:
  - one per-tick CSV per episode      (out_dir/<policy>_seed<i>.csv)
  - one summary CSV across all seeds  (out_dir/<policy>_summary.csv)
"""
from __future__ import annotations
import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

from .run_episode import run_episode
from .hell_disturbance import scaled_hell_config


def _load_policy(spec: str):
    """spec = 'module:ClassName'  e.g.  'baselines.hold:HoldPolicy'
       Returns a callable that produces a fresh policy per seed."""
    mod_name, cls_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return lambda seed: cls(seed=seed)


def main():
    ap = argparse.ArgumentParser(
        description="Run a multi-seed sweep of the IterHybrid-XHARD benchmark.")
    ap.add_argument("--policy",     required=True,
                    help="Policy spec 'module:ClassName' "
                         "(e.g. 'baselines.hold:HoldPolicy')")
    ap.add_argument("--policy-name", default=None,
                    help="Tag used in output filenames "
                         "(default: derived from --policy)")
    ap.add_argument("--seeds",      type=int,   default=10)
    ap.add_argument("--severity",   type=float, default=5.0,
                    help="XHARD severity multiplier (1.0 = standard, "
                         "5.0 = the published benchmark point)")
    ap.add_argument("--t-final",    type=float, default=1200.0,
                    help="Simulation duration in seconds")
    ap.add_argument("--control-dt", type=float, default=1.0)
    ap.add_argument("--out-dir",    type=Path,
                    default=Path("results"))
    args = ap.parse_args()

    policy_factory = _load_policy(args.policy)
    pol_tag = args.policy_name or args.policy.split(":")[-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hell_cfg = scaled_hell_config(severity=args.severity)

    print(f"=== {pol_tag} XHARD sweep  severity={args.severity}  "
          f"t_final={args.t_final}  seeds={args.seeds} ===")
    print(f"{'seed':>4} {'surv':>5} {'t_end':>7} {'Q_mean':>7} "
          f"{'q_min':>6} {'H98':>5} {'events':>7} {'wall_s':>7}")
    print("-" * 60)

    results = []
    t0 = time.time()
    for seed in range(args.seeds):
        csv_path = args.out_dir / f"{pol_tag}_seed{seed}.csv"
        try:
            pol = policy_factory(seed)
            res = run_episode(
                pol,
                seed=seed,
                t_final=args.t_final,
                control_dt=args.control_dt,
                out_csv=csv_path,
                verbose=False,
                hell_cfg=hell_cfg,
            )
        except Exception as e:
            print(f"{seed:>4d}  ERROR: {e}", flush=True)
            continue
        surv = "YES" if not res["disrupted"] else "no"
        print(f"{seed:>4d} {surv:>5} {res['t_final']:>7.1f} "
              f"{res['Q_mean']:>7.2f} {res['q_min_min']:>6.3f} "
              f"{res['H98_mean']:>5.2f} {res['events_total']:>7d} "
              f"{res['wall_s']:>7.1f}", flush=True)
        results.append(res)

    elapsed = time.time() - t0
    survivors = sum(1 for r in results if not r["disrupted"])
    print("-" * 60)
    print(f"Survival: {survivors}/{len(results)} = "
          f"{100*survivors/max(1, len(results)):.1f}%   "
          f"(total wall: {elapsed:.0f}s)")

    if results:
        agg = args.out_dir / f"{pol_tag}_summary.csv"
        with open(agg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Summary: {agg}")


if __name__ == "__main__":
    main()
