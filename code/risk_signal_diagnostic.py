#!/usr/bin/env python3
"""Diagnostic for the explicit risk channels on the ERA5 replay.

The ablation shows that removing all three risk channels leaves the composite
objective within the CI, so on these data they do not improve the measured
objective. This script measures a candidate mechanism: how sharply the violation
probability p separates zones during real heat episodes, compared with the
deviation proxy delta.

Key measurement: p is computed from a Gaussian tail beyond the safety bound, so
once a zone is well past 34 C the tail probability narrows toward its ceiling.
p still orders the zones (its top two tie in only ~3% of hours), but its spread
contracts exactly when the value is high, whereas delta (bounded drift since the
last successful update) keeps varying. This is a diagnostic, not a proof that
the risk terms are unnecessary; it characterises one regime, not all.

Reports, over every zone-hour that is a true violation:
  - mean / median / stdev / max of p and of delta
  - share of samples at the ceiling (>= 0.95) and near it (>= 0.80)
  - Kendall/Spearman-free discrimination proxy: number of distinct ranking
    orders induced by p alone vs delta alone vs the deployed score
  - the same statistics restricted to the subset where p >= 0.80 (the saturated
    regime that dominates heat episodes)

Deterministic; reads the same cached ERA5 CSVs as the main pipeline.
Writes outputs/rabs/rabs_risk_signal_diagnostic.csv.
"""
from __future__ import annotations
import csv, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/era5_vn'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ['can_tho', 'soc_trang', 'ca_mau']
SAFE_MIN, SAFE_MAX = 22.0, 34.0
SIGMA = 1.2
HALF_WIDTH = (SAFE_MAX - SAFE_MIN) / 2.0     # r = 6, as in the deployed score
STALE_HOURS = 3                               # gateway estimate lag used for delta
CEILING = 0.95
NEAR = 0.80
W_RISK, W_DEV, W_AOI = 0.55, 0.25, 0.20
A_REF = 8.0


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def read_stations():
    cols = []
    for s in STATIONS:
        xs = []
        with (DATA / f'{s}_2024.csv').open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['temp_c'] not in ('', 'None'):
                    xs.append(float(r['temp_c']))
        cols.append(xs)
    T = min(len(c) for c in cols)
    return cols, T


def main():
    cols, T = read_stations()
    ps, ds = [], []
    # per-hour records so we can compare induced rankings within the same hour
    hours = []
    for t in range(STALE_HOURS, T):
        rec = []
        any_v = False
        for z in range(3):
            x = cols[z][t]
            xlast = cols[z][t - STALE_HOURS]
            p = phi((SAFE_MIN - x) / SIGMA) + (1.0 - phi((SAFE_MAX - x) / SIGMA))
            p = min(1.0, max(0.0, p))
            v = 1 if (x < SAFE_MIN or x > SAFE_MAX) else 0
            d = min(abs(x - xlast) / HALF_WIDTH, 1.0)
            a = min(STALE_HOURS / A_REF, 1.0)
            rec.append({'p': p, 'd': d, 'a': a, 'v': v, 'x': x})
            if v:
                ps.append(p); ds.append(d); any_v = True
        if any_v:
            hours.append(rec)

    def stats(xs, name):
        return {
            'signal': name, 'n': len(xs),
            'mean': statistics.mean(xs), 'median': statistics.median(xs),
            'stdev': statistics.pstdev(xs), 'min': min(xs), 'max': max(xs),
            'share_ge_0.95': sum(1 for v in xs if v >= CEILING) / len(xs),
            'share_ge_0.80': sum(1 for v in xs if v >= NEAR) / len(xs),
        }

    rows = [stats(ps, 'p_vio (violation hours)'), stats(ds, 'delta (violation hours)')]

    # Restricted to the saturated regime: among zone-hours with p >= 0.80,
    # how much does p still vary, and does delta still discriminate?
    sat = [(rec, z) for rec in hours for z in range(3) if rec[z]['p'] >= NEAR]
    if sat:
        sat_p = [rec[z]['p'] for rec, z in sat]
        sat_d = [rec[z]['d'] for rec, z in sat]
        rows.append(stats(sat_p, 'p_vio | p>=0.80'))
        rows.append(stats(sat_d, 'delta | p>=0.80'))

    # Discrimination test: within each violation hour, does p alone produce a
    # unique ordering of the 3 zones, or does it tie? Ties mean the signal
    # cannot say which zone to poll first.
    def tie_rate(keyfn, ceil=None):
        ties = 0; total = 0
        for rec in hours:
            vals = [keyfn(rec[z]) for z in range(3)]
            if ceil is not None and max(vals) < ceil:
                continue
            total += 1
            # count hours where the top-2 are indistinguishable at 4 decimals
            srt = sorted(vals, reverse=True)
            if abs(srt[0] - srt[1]) < 1e-4:
                ties += 1
        return ties / total if total else float('nan'), total

    t_p, n_p = tie_rate(lambda r: r['p'])
    t_d, n_d = tie_rate(lambda r: r['d'])
    t_s, n_s = tie_rate(lambda r: W_RISK * r['p'] + W_DEV * r['d'] + W_AOI * r['a'])
    # and restricted to the saturated hours (p_max >= 0.80)
    t_ps, n_ps = tie_rate(lambda r: r['p'], ceil=NEAR)
    t_ds, n_ds = tie_rate(lambda r: r['d'], ceil=NEAR)

    print(f"violation zone-hours sampled: {len(ps)}   violation hours: {len(hours)}")
    print(f"{'signal':28}{'n':>6}{'mean':>8}{'stdev':>8}{'>=0.95':>9}{'>=0.80':>9}")
    for r in rows:
        print(f"{r['signal']:28}{r['n']:>6}{r['mean']:>8.4f}{r['stdev']:>8.4f}"
              f"{r['share_ge_0.95']*100:>8.1f}%{r['share_ge_0.80']*100:>8.1f}%")

    print("\n--- top-2 tie rate within violation hours (higher = less discriminating) ---")
    print(f"  p alone          : {t_p*100:.1f}%  (n={n_p})")
    print(f"  delta alone      : {t_d*100:.1f}%  (n={n_d})")
    print(f"  deployed score S : {t_s*100:.1f}%  (n={n_s})")
    print(f"  p alone, saturated hours (p_max>=0.80): {t_ps*100:.1f}%  (n={n_ps})")
    print(f"  delta alone, same hours              : {t_ds*100:.1f}%  (n={n_ds})")

    fields = list(rows[0].keys()) + ['tie_rate_p', 'tie_rate_delta', 'tie_rate_score',
                                     'tie_rate_p_saturated', 'tie_rate_delta_saturated',
                                     'n_violation_hours']
    with (OUT / 'rabs_risk_signal_diagnostic.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            r.update({'tie_rate_p': t_p, 'tie_rate_delta': t_d, 'tie_rate_score': t_s,
                      'tie_rate_p_saturated': t_ps, 'tie_rate_delta_saturated': t_ds,
                      'n_violation_hours': len(hours)})
            w.writerow(r)
    print('WROTE', OUT / 'rabs_risk_signal_diagnostic.csv')


if __name__ == '__main__':
    main()
