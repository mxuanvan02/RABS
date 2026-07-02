#!/usr/bin/env python3
"""Verify the dual-update bound for RABS-PD empirically (derive-then-verify).

Re-runs the RABS-PD bandwidth dual eta_t = [eta_t + alpha*(B_t - B_target)]_+
and logs: (i) sup_t eta_t (boundedness), (ii) running-average budget tracking,
(iii) time-average constraint violation (1/T) sum (B_t - B_target).

Theory predicts (projected dual ascent / drift argument):
  - eta_t stays bounded because B_t in {1,2,3} gives a bounded per-step
    increment alpha*(B_t - B_target).
  - (1/T) sum_t (B_t - B_target) <= (eta_1 - eta_{T+1})/(alpha T)
    => time-average budget converges to <= B_target + O(1/T).
Stdlib only; imports the simulation primitives from the main script.
"""
from __future__ import annotations
import statistics, random
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rabs_main", ROOT / "code" / "run_rabs_adaptive_bandwidth.py")
assert SPEC and SPEC.loader, "could not load main simulation module"
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

ALPHA = 0.010          # bandwidth dual step (matches main script)
B_TARGET = 1.55        # bandwidth target (matches main script)
B_MIN, B_MAX = 1, 3


def run_trajectory(steps, network, seed, start, end):
    """Replay RABS-PD logging the bandwidth dual trajectory only."""
    rng = random.Random(seed); bad = False
    pbad = M.NETWORKS[network][1] / (M.NETWORKS[network][1] + M.NETWORKS[network][3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    eta_hist = []; B_hist = []; running_avg = []; viol_cumsum = 0.0
    for k in range(start + 1, end):
        true = steps[k]
        B, R = M.choose_B_rabs_family('rabs_pd', true, hat, age, pbad, dual)
        idx = M.choose_sensors(true, hat, age, B, 'rabs_pd')
        age = [a + 1 for a in age]
        for i in idx:
            ok, bad = M.channel(network, rng, bad)
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok:
                hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = M.eval_step(true, hat)
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        # log BEFORE the dual update so eta_hist[k] is the eta used at slot k
        eta_hist.append(dual['bw']); B_hist.append(B)
        viol_cumsum += (B - B_TARGET)
        running_avg.append(viol_cumsum / len(B_hist))
        dual['bw'] = max(0.0, dual['bw'] + ALPHA * (B - B_TARGET))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
    return eta_hist, B_hist, running_avg


def main():
    steps = M.read_steps()
    per = 1000
    starts = [i * per for i in range(0, 16) if (i + 1) * per <= len(steps)]
    seeds = M.SEEDS
    incr_max = ALPHA * (B_MAX - B_TARGET)
    sup_eta = 0.0; final_avgs = []; all_avgB = []
    for net in M.NETWORKS:
        for start in starts:
            for seed in seeds:
                eta, Bs, ravg = run_trajectory(steps, net, seed, start, start + per)
                sup_eta = max(sup_eta, max(eta))
                final_avgs.append(ravg[-1])           # time-avg (B_t - B_target)
                all_avgB.append(sum(Bs) / len(Bs))
    T = per - 1
    print("=== RABS-PD dual-update empirical verification ===")
    print(f"runs                         : {len(M.NETWORKS)*len(starts)*len(seeds)}")
    print(f"horizon per run (T)          : {T}")
    print(f"alpha (dual step)            : {ALPHA}")
    print(f"B_target                     : {B_TARGET}")
    print(f"per-step increment bound     : alpha*(B_max-B_target) = {incr_max:.4f}")
    print(f"measured sup_t eta_t         : {sup_eta:.4f}")
    print(f"mean time-avg (B_t-B_target) : {statistics.mean(final_avgs):+.5f}")
    print(f"max  time-avg (B_t-B_target) : {max(final_avgs):+.5f}")
    print(f"mean realized avg B          : {statistics.mean(all_avgB):.4f}")
    print(f"theory drift bound eta_sup/(alpha*T) : {sup_eta/(ALPHA*T):+.5f}")
    print("interpretation: time-avg budget violation <= eta_sup/(alpha*T); "
          "measured violation should sit at/below this and near 0.")


if __name__ == '__main__':
    main()
