#!/usr/bin/env python3
"""Non-stationary channel extension for RABS (derive-then-verify, real run).

Adds a time-varying ("drift") Gilbert-Elliott channel whose loss parameters
oscillate slowly over the horizon, emulating diurnal/humidity-driven changes
in greenhouse link quality. Reuses the canonical simulator primitives so the
policy logic is byte-identical to the main results; only the channel is
replaced. Reports RABS-PD against fixed and content-aware baselines.

Outputs:
  outputs/rabs/rabs_nonstationary_summary.csv
  outputs/tables/table_rabs_nonstationary.tex
Stdlib only.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rabs_main", ROOT / "code" / "run_rabs_adaptive_bandwidth.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

OUT = ROOT / "outputs" / "rabs"; OUT.mkdir(parents=True, exist_ok=True)
OUTT = ROOT / "outputs" / "tables"; OUTT.mkdir(parents=True, exist_ok=True)

# Base anchor (mean of burst and severe_burst) plus a slow oscillation.
# params: (gl, pgb, bl, pbg) = good-loss, P(G->B), bad-loss, P(B->G)
BASE = (0.07, 0.045, 0.74, 0.185)
PERIOD = 250          # slots per drift cycle
AMP = 0.5             # +-50% relative oscillation on loss rates
DISPLAY = ['fixed_b2', 'fixed_b3', 'max_aoi', 'voi_b2',
           'rabs_h', 'rabs_l', 'rabs_pd', 'oracle_b']


def drift_params(t):
    gl, pgb, bl, pbg = BASE
    phase = math.sin(2 * math.pi * t / PERIOD)
    f = 1.0 + AMP * phase
    return (min(0.99, gl * f), pgb, min(0.99, bl * f), pbg)


def drift_channel(t, rng, bad):
    gl, pgb, bl, pbg = drift_params(t)
    if bad:
        ok = rng.random() > bl
        if rng.random() < pbg: bad = False
    else:
        ok = rng.random() > gl
        if rng.random() < pgb: bad = True
    return ok, bad


def run_nonstationary(steps, policy, seed, start, end):
    rng = random.Random(seed); bad = False
    pbad = BASE[1] / (BASE[1] + BASE[3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]; counts = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []; okn = att = 0
    for k in range(start + 1, end):
        true = steps[k]
        if policy == 'oracle_b':
            best = (1e9, 1)
            for B in [1, 2, 3]:
                _, l, ao, miss_risk = M.oracle_candidate(true, hat, age, B)
                obj = l + 0.04 * B + 0.010 * ao + 0.030 * miss_risk
                if obj < best[0]: best = (obj, B)
            B = best[1]
        elif policy in ['rabs_l', 'rabs_pd']:
            B, _ = M.choose_B_rabs_family(policy, true, hat, age, pbad, dual)
        else:
            B, _ = M.choose_B_threshold(policy, true, hat, age, pbad)
        idx = M.choose_sensors(true, hat, age, B, policy)
        bus.append(B); age = [a + 1 for a in age]
        for i in idx:
            ok, bad = drift_channel(k, rng, bad); att += 1; okn += 1 if ok else 0; counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = M.eval_step(true, hat); loss.append(l); tp += a; fp += b; tn += c; fn += d
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        if policy == 'rabs_pd':
            dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
            dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
            dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    objective = sum(loss) / len(loss) + 0.04 * avgB + 0.015 * (sum(aoi) / len(aoi))
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    return {'policy': policy, 'seed': seed, 'window_start': start,
            'objective': objective, 'avg_bandwidth': avgB,
            'bandwidth_saving_vs_b3_pct': 100 * (3 - avgB) / 3,
            'missed_pct': missed, 'avg_aoi': sum(aoi) / len(aoi),
            'packet_success_pct': 100 * okn / att if att else 100}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    steps = M.read_steps(); per = 1000
    starts = [i * per for i in range(0, 16) if (i + 1) * per <= len(steps)]
    raw = []
    for start in starts:
        for seed in M.SEEDS:
            for pol in DISPLAY:
                raw.append(run_nonstationary(steps, pol, seed, start, start + per))
    with (OUT / 'rabs_nonstationary_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    # aggregate
    agg = {}
    for pol in DISPLAY:
        sub = [r for r in raw if r['policy'] == pol]
        agg[pol] = {m: mean([r[m] for r in sub]) for m in
                    ['objective', 'avg_bandwidth', 'bandwidth_saving_vs_b3_pct', 'missed_pct', 'avg_aoi', 'packet_success_pct']}
        agg[pol]['obj_ci'] = ci([r['objective'] for r in sub])
    # tex table
    lines = ['\\begin{tabular}{lrrrrr}', '\\hline',
             'Policy & Obj. & $\\bar{B}$ & Save (\\%) & Missed (\\%) & AoI \\\\', '\\hline']
    for pol in DISPLAY:
        a = agg[pol]
        pol_tex = pol.replace('_', '\\_')
        lines.append(f"{pol_tex} & {a['objective']:.4f} & {a['avg_bandwidth']:.2f} & "
                     f"{a['bandwidth_saving_vs_b3_pct']:.1f} & {a['missed_pct']:.2f} & {a['avg_aoi']:.2f} \\\\")
    lines += ['\\hline', '\\end{tabular}']
    (OUTT / 'table_rabs_nonstationary.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT / 'rabs_nonstationary_summary.csv')
    print(f"{'policy':12} {'obj':>7} {'B':>5} {'save%':>6} {'miss%':>6} {'pkt%':>6}")
    for pol in DISPLAY:
        a = agg[pol]
        print(f"{pol:12} {a['objective']:7.4f} {a['avg_bandwidth']:5.2f} "
              f"{a['bandwidth_saving_vs_b3_pct']:6.1f} {a['missed_pct']:6.2f} {a['packet_success_pct']:6.1f}")


if __name__ == '__main__':
    main()
