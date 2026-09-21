#!/usr/bin/env python3
"""Empirical calibration between the missed-risk SURROGATE Mhat_t
(unpolled exit-risk, what the controller regulates) and the REALIZED per-slot
miss indicator m_t (what the operator cares about), on the deployed RABS-PD
under severe_burst. Reports mean surrogate, mean realized, mean gap, and
Pearson/Spearman correlation, per seed x window (n=240).

Deterministic stdlib-only; reuses the released primitives.
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
NETWORK = 'severe_burst'
GE = (0.08, 0.055, 0.82, 0.15)
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 720

def phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

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

def channel(rng, bad):
    gl, pgb, bl, pbg = GE
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

def choose_sensors(true, hat, age, B):
    scores = []
    for i in range(3):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        scores.append(0.55 * true[i]['p'] + 0.25 * err + 0.20 * age_score)
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

def predict_candidate(true, hat, age, B):
    idx = choose_sensors(true, hat, age, B)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['mu']
    l, tp, fp, tn, fn = eval_step(true, h)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(3)) / 3
    miss_risk = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
    return idx, l, pred_aoi, miss_risk

def run_calib(steps, seed, start, end):
    """Deployed RABS-PD; also record per-slot (Mhat_t, realized m_t) pairs."""
    rng = random.Random(seed); bad = False
    pbad = GE[1] / (GE[1] + GE[3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    pairs = []   # (window-mean Mhat, window-mean realized) aggregated at end
    Mh_sum = 0.0; m_sum = 0.0; nslots = 0
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    for k in range(start + 1, end):
        true = steps[k]
        R = risk_score(true, hat, age, pbad)
        best = (1e18, 1)
        for B in [1, 2, 3]:
            _, l, ao, mr = predict_candidate(true, hat, age, B)
            score = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                     + (0.035 + dual['miss']) * mr - 0.030 * R * B)
            if score < best[0]: best = (score, B)
        B = best[1]
        # surrogate BEFORE the slot's transmissions is computed on the chosen set
        _, _, _, mr_chosen = predict_candidate(true, hat, age, B)
        idx = choose_sensors(true, hat, age, B)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(rng, bad)
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat); loss.append(l)
        tp += a; fp += b; tn += c; fn += d
        m_t = d / (a + d) if (a + d) else 0.0     # realized per-slot miss rate
        Mh_sum += mr_chosen; m_sum += m_t; nslots += 1
        mean_aoi = sum(age) / 3
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (m_t - 0.008))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    avg_aoi = sum(aoi) / len(aoi)
    objective = sum(loss) / len(loss) + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    return {'objective': objective, 'missed_pct': missed,
            'Mhat_win': Mh_sum / nslots, 'm_win': m_sum / nslots}

def mean(xs): return sum(xs) / len(xs)
def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx * dy > 0 else float('nan')

def main():
    steps = read_stations()
    starts = [i * PER for i in range(20) if (i + 1) * PER <= len(steps)]
    rows = []
    for start in starts:
        for seed in SEEDS:
            rows.append(run_calib(steps, seed, start, start + PER))
    Mh = [r['Mhat_win'] for r in rows]
    mm = [r['m_win'] for r in rows]
    eps = [abs(a - b) for a, b in zip(Mh, mm)]
    print(f"n windows = {len(rows)} (severe_burst, deployed RABS-PD)")
    print(f"mean surrogate  E[Mhat]  = {mean(Mh):.4f}")
    print(f"mean realized   E[m]     = {mean(mm):.4f}")
    print(f"mean |Mhat - m| (epsilon)= {mean(eps):.4f}   max = {max(eps):.4f}")
    print(f"Pearson(Mhat, m)         = {pearson(Mh, mm):.3f}")
    # violation-conditional calibration: slots where realized miss happened
    hit = [(a, b) for a, b in zip(Mh, mm) if b > 0]
    if hit:
        print(f"windows with realized misses: {len(hit)};  E[Mhat | m>0] = {mean([a for a,_ in hit]):.4f}")
    with (OUT / 'rabs_surrogate_calibration.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('WROTE', OUT / 'rabs_surrogate_calibration.csv')

if __name__ == '__main__':
    main()
