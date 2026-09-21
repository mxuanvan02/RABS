#!/usr/bin/env python3
"""RABS urgency-score ablation on the HARD ERA5 Vietnam replay.

The three Mekong-delta stations (Can Tho, Soc Trang, Ca Mau) are the N=3 zones.
The deployed urgency score is
    S_i = w_risk * g(p) + w_dev * delta + w_aoi * aoi_norm ,  (0.55,0.25,0.20)
where the risk channel g(p) is the value-of-uncertainty form 4p(1-p).

We compare, on the same real data/seeds/windows:
  - full            : VoU risk channel + dev + aoi
  - raw_p           : replace VoU 4p(1-p) by raw probability p (naive heuristic)
  - no_risk         : drop the risk channel entirely
  - no_dev          : drop the deviation proxy
  - no_aoi          : drop the AoI freshness term
Lower composite objective = better. Stdlib-only, reproducible.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/era5_vn'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ['can_tho', 'soc_trang', 'ca_mau']
SAFE_MIN, SAFE_MAX = 22.0, 34.0
SIGMA = 1.2
NETWORKS = {'burst': (0.06, 0.035, 0.65, 0.22),
            'severe_burst': (0.08, 0.055, 0.82, 0.15)}
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 720

# variant -> (risk_form, w_risk, w_dev, w_aoi) before renormalization
VARIANTS = {
    'full':    ('vou', 0.55, 0.25, 0.20),
    'raw_p':   ('raw', 0.55, 0.25, 0.20),
    'no_risk': ('vou', 0.00, 0.25, 0.20),
    'no_dev':  ('vou', 0.55, 0.00, 0.20),
    'no_aoi':  ('vou', 0.55, 0.25, 0.00),
}


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
    steps = []
    for t in range(T):
        row = []
        for zi in range(3):
            x = cols[zi][t]
            p = phi((SAFE_MIN - x) / SIGMA) + (1.0 - phi((SAFE_MAX - x) / SIGMA))
            p = min(1.0, max(0.0, p))
            v = 1 if (x < SAFE_MIN or x > SAFE_MAX) else 0
            row.append({'x': x, 'mu': x, 'p': p, 'v': v})
        steps.append(row)
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


def choose_sensors(true, hat, age, B, form, w):
    wr, wd, wa = w
    s = wr + wd + wa
    wr, wd, wa = wr / s, wd / s, wa / s
    scores = []
    for i in range(3):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        p = true[i]['p']
        rc = 4.0 * p * (1.0 - p) if form == 'vou' else p
        scores.append(wr * rc + wd * err + wa * age_score)
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


def predict_candidate(true, hat, age, B, form, w):
    idx = choose_sensors(true, hat, age, B, form, w)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['mu']
    l, tp, fp, tn, fn = eval_step(true, h)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(3)) / 3
    miss_risk = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
    return idx, l, pred_aoi, miss_risk


def run(steps, form, w, network, seed, start, end):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    for k in range(start + 1, end):
        true = steps[k]
        R = risk_score(true, hat, age, pbad)
        best = (1e18, 1)
        for B in [1, 2, 3]:
            _, l, ao, mr = predict_candidate(true, hat, age, B, form, w)
            score = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                     + (0.035 + dual['miss']) * mr - 0.030 * R * B)
            if score < best[0]: best = (score, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B, form, w)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad)
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
    sl = sorted(loss, reverse=True); kk = max(1, int(0.05 * len(sl)))
    cvar95 = sum(sl[:kk]) / kk
    return {'objective': objective, 'avg_bandwidth': avgB, 'missed_pct': missed,
            'avg_aoi': avg_aoi, 'cvar95_loss': cvar95}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    steps = read_stations()
    starts = [i * PER for i in range(20) if (i + 1) * PER <= len(steps)]
    metrics = ['objective', 'avg_bandwidth', 'missed_pct', 'avg_aoi', 'cvar95_loss']
    rows = []
    for net in NETWORKS:
        for name, (form, wr, wd, wa) in VARIANTS.items():
            acc = {m: [] for m in metrics}
            for start in starts:
                for seed in SEEDS:
                    r = run(steps, form, (wr, wd, wa), net, seed, start, start + PER)
                    for m in metrics: acc[m].append(r[m])
            rec = {'network': net, 'variant': name, 'n': len(acc['objective'])}
            for m in metrics:
                rec[m + '_mean'] = mean(acc[m]); rec[m + '_ci95'] = ci(acc[m])
            rows.append(rec)
    with (OUT / 'rabs_era5_ablation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('net           variant    Obj      B_avg  Missed%  CVaR95')
    for r in rows:
        print(f"{r['network']:<13} {r['variant']:<9} {r['objective_mean']:.4f}  "
              f"{r['avg_bandwidth_mean']:.2f}   {r['missed_pct_mean']:.2f}    {r['cvar95_loss_mean']:.4f}")
    print('WROTE', OUT / 'rabs_era5_ablation_summary.csv')


if __name__ == '__main__':
    main()
