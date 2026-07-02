#!/usr/bin/env python3
"""RABS urgency-score ablation on the REAL 3-zone greenhouse replay.

The deployed urgency score is
    S_i = w_risk * p_vio + w_dev * delta + w_aoi * aoi_norm ,  (0.55,0.25,0.20)

To show every component contributes, we re-run RABS-PD with each term
zeroed out (and the remaining weights renormalized), on the same real data,
seeds, and windows as the main experiment. Lower composite objective = better.

Stdlib-only, reproducible. Selector/budget logic is identical to
run_rabs_adaptive_bandwidth.py; only the urgency weights change.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/source/safety_probability_calibration_raw.csv'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)
OUTT = ROOT / 'outputs/tables'; OUTT.mkdir(parents=True, exist_ok=True)

NETWORKS = {'burst': (0.06, 0.035, 0.65, 0.22),
            'severe_burst': (0.08, 0.055, 0.82, 0.15)}
SAFE_MIN, SAFE_MAX = 22.0, 30.0
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 1000

# ablation variants: (w_risk, w_dev, w_aoi) BEFORE renormalization
VARIANTS = {
    'full':        (0.55, 0.25, 0.20),   # deployed
    'no_risk':     (0.00, 0.25, 0.20),   # drop violation-probability term
    'no_dev':      (0.55, 0.00, 0.20),   # drop deviation proxy
    'no_aoi':      (0.55, 0.25, 0.00),   # drop AoI freshness
}


def read_steps():
    d = {}
    with SRC.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            w = int(float(r['window'])); t = int(float(r['t'])); loop = int(float(r['loop']))
            rec = {'x': float(r['xt']), 'mu': float(r['mu']),
                   'p': float(r['p_gaussian']), 'v': int(float(r['true_violation']))}
            d.setdefault((w, t), {})[loop] = rec
    steps = []
    for key in sorted(d):
        if len(d[key]) == 3:
            steps.append([d[key][i] for i in range(3)])
    return steps


def channel(kind, rng, bad):
    gl, pgb, bl, pbg = NETWORKS[kind]
    if bad:
        ok = rng.random() > bl
        if rng.random() < pbg: bad = False
    else:
        ok = rng.random() > gl
        if rng.random() < pgb: bad = True
    return ok, bad


def proxy_error(records, hat, i):
    return min(abs(records[i]['mu'] - hat[i]) / 8.0, 1.0)


def risk_score(records, hat, age, pbad):
    ps = [r['p'] for r in records]
    err = [proxy_error(records, hat, i) for i in range(3)]
    maxp = max(ps); meanp = sum(ps) / 3; meana = sum(age) / 3
    return 0.45 * maxp + 0.20 * meanp + 0.18 * min(meana / 8.0, 1.0) + 0.10 * pbad + 0.07 * max(err)


def choose_sensors(true, hat, age, B, w):
    wr, wd, wa = w
    s = wr + wd + wa
    wr, wd, wa = wr / s, wd / s, wa / s   # renormalize so weights sum to 1
    scores = []
    for i in range(3):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        score = wr * true[i]['p'] + wd * err + wa * age_score
        scores.append(score)
    return sorted(range(3), key=lambda i: scores[i], reverse=True)[:B]


def eval_step(true, hat):
    loss = 0; tp = fp = tn = fn = 0
    for i in range(3):
        pred = 1 if (hat[i] < SAFE_MIN or hat[i] > SAFE_MAX or true[i]['p'] >= 0.55) else 0
        y = true[i]['v']
        if pred and y: tp += 1
        elif pred and not y: fp += 1
        elif not pred and y: fn += 1
        else: tn += 1
        loss += (abs(hat[i] - true[i]['x']) / 8.0) ** 2 + 5 * (1 if y and not pred else 0) + 1 * (1 if pred and not y else 0)
    return loss / 3, tp, fp, tn, fn


def predict_candidate(true, hat, age, B, w):
    idx = choose_sensors(true, hat, age, B, w)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['mu']
    l, tp, fp, tn, fn = eval_step(true, h)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(3)) / 3
    miss_risk = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
    return idx, l, pred_aoi, miss_risk


def run(steps, w, network, seed, start, end):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]; counts = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    for k in range(start + 1, end):
        true = steps[k]
        R = risk_score(true, hat, age, pbad)
        best = (1e18, 1)
        for B in [1, 2, 3]:
            _, l, ao, miss_risk = predict_candidate(true, hat, age, B, w)
            score = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                     + (0.035 + dual['miss']) * miss_risk - 0.030 * R * B)
            if score < best[0]: best = (score, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B, w)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad); counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat); loss.append(l); tp += a; fp += b; tn += c; fn += d
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    avg_aoi = sum(aoi) / len(aoi)
    objective = sum(loss) / len(loss) + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    # tail metrics: CVaR_0.95 of the per-slot loss (mean of worst 5% slots)
    sl = sorted(loss, reverse=True)
    k = max(1, int(0.05 * len(sl)))
    cvar95 = sum(sl[:k]) / k
    var95 = sl[k - 1]
    return {'objective': objective, 'avg_bandwidth': avgB,
            'missed_pct': missed, 'avg_aoi': avg_aoi,
            'cvar95_loss': cvar95, 'var95_loss': var95}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    steps = read_steps()
    starts = [i * PER for i in range(16) if (i + 1) * PER <= len(steps)]
    metrics = ['objective', 'avg_bandwidth', 'missed_pct', 'avg_aoi', 'cvar95_loss', 'var95_loss']
    rows = []
    for net in NETWORKS:
        for name, w in VARIANTS.items():
            acc = {m: [] for m in metrics}
            for start in starts:
                for seed in SEEDS:
                    r = run(steps, w, net, seed, start, start + PER)
                    for m in metrics: acc[m].append(r[m])
            rec = {'network': net, 'variant': name, 'n': len(acc['objective'])}
            for m in metrics:
                rec[m + '_mean'] = mean(acc[m]); rec[m + '_ci95'] = ci(acc[m])
            rows.append(rec)
    with (OUT / 'rabs_ablation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('net           variant    Obj      B_avg  Missed%  AoI    CVaR95  VaR95')
    for r in rows:
        print(f"{r['network']:<13} {r['variant']:<9} {r['objective_mean']:.4f}  "
              f"{r['avg_bandwidth_mean']:.2f}   {r['missed_pct_mean']:.2f}    {r['avg_aoi_mean']:.2f}   "
              f"{r['cvar95_loss_mean']:.4f}  {r['var95_loss_mean']:.4f}")
    print('WROTE', OUT / 'rabs_ablation_summary.csv')


if __name__ == '__main__':
    main()
