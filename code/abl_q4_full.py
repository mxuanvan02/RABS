#!/usr/bin/env python3
"""Q4 final: ablate ALL THREE risk channels of RABS-PD, on two channel regimes.

Risk channels in the deployed selector
  score = Lhat(b) + (c_B+lB)b + (c_A+lA)Ahat(b) + (c_M+lM)Mhat(b) - c_R*R*B
and ranking  S_i = w_risk*p_i + w_dev*delta_i + w_aoi*aoi_i :

  (1) rank risk    : w_risk=0.55 on p_i in choose_sensors
  (2) budget risk  : -c_R * R_{t-} * B   (the namesake "risk-adaptive scaling")
  (3) miss surrogate: (c_M + lambda^M) * Mhat(b), lambda^M the PD dual feedback

Variants (share data/seeds/windows/duals otherwise):
  deployed         : all three ON
  no_rank_risk     : (1) off
  no_budget_risk   : (2) off  (c_R=0)
  no_miss_surrogate: (3) off  (c_M=0, lambda^M frozen 0)
  no_risk_all      : (1)+(2)+(3) all off  -- strip every risk mechanism

Regimes: severe_burst (submitted) + extreme_burst (harsher, from branch-B).
Paired Wilcoxon deployed-vs-ablated; effect size r = |Z|/sqrt(n_nonzero),
Z from scipy's tie-corrected two-sided p (robust to zero-inflated miss rates).
Deterministic. Writes outputs/rabs/rabs_q4_full_ablation.csv.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/era5_vn'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ['can_tho', 'soc_trang', 'ca_mau']
SAFE_MIN, SAFE_MAX = 22.0, 34.0
SIGMA = 1.2
NETWORKS = {'severe_burst':  (0.08, 0.055, 0.82, 0.15),
            'extreme_burst': (0.12, 0.070, 0.90, 0.08)}
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 720
# name -> (rank_risk, c_R, miss_surrogate)
VARIANTS = {
    'deployed':          (True,  0.030, True),
    'no_rank_risk':      (False, 0.030, True),
    'no_budget_risk':    (True,  0.000, True),
    'no_miss_surrogate': (True,  0.030, False),
    'no_risk_all':       (False, 0.000, False),
}

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

def choose_sensors(true, hat, age, B, rank_risk):
    wr, wd, wa = (0.55, 0.25, 0.20) if rank_risk else (0.0, 0.25, 0.20)
    s = wr + wd + wa
    wr, wd, wa = wr / s, wd / s, wa / s
    scores = []
    for i in range(3):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        scores.append(wr * true[i]['p'] + wd * err + wa * age_score)
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

def predict_candidate(true, hat, age, B, rank_risk):
    idx = choose_sensors(true, hat, age, B, rank_risk)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['mu']
    l, tp, fp, tn, fn = eval_step(true, h)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(3)) / 3
    miss_risk = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
    return idx, l, pred_aoi, miss_risk

def run(steps, rank_risk, c_R, miss_sur, network, seed, start, end):
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
            _, l, ao, mr = predict_candidate(true, hat, age, B, rank_risk)
            cm = 0.035 if miss_sur else 0.0
            lm = dual['miss'] if miss_sur else 0.0
            score = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                     + (cm + lm) * mr - c_R * R * B)
            if score < best[0]: best = (score, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B, rank_risk)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad)
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat); loss.append(l); tp += a; fp += b; tn += c; fn += d
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        if miss_sur:
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

def wilcox(x, y):
    """(delta_mean, p, r, n_nonzero). r=|Z|/sqrt(n_nz) from tie-corrected p."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, float('nan'), float('nan'), 0
    res = stats.wilcoxon(x, y)
    p = float(res.pvalue)
    if p <= 0 or p >= 1:
        r = 1.0 if p <= 0 else 0.0
    else:
        z = abs(stats.norm.ppf(1 - p / 2.0))
        r = min(1.0, z / math.sqrt(n))
    return float(np.mean(d)), p, r, n

def main():
    steps = read_stations()
    starts = [i * PER for i in range(20) if (i + 1) * PER <= len(steps)]
    metrics = ['objective', 'avg_bandwidth', 'missed_pct', 'avg_aoi', 'cvar95_loss']
    per_runs = {}
    rows = []
    for net in NETWORKS:
        for name, (rr, cR, ms) in VARIANTS.items():
            acc = {m: [] for m in metrics}
            runs = []
            for start in starts:
                for seed in SEEDS:
                    r = run(steps, rr, cR, ms, net, seed, start, start + PER)
                    runs.append(r)
                    for m in metrics: acc[m].append(r[m])
            per_runs[(net, name)] = runs
            rec = {'network': net, 'variant': name, 'rank_risk': rr, 'c_R': cR,
                   'miss_surrogate': ms, 'n': len(acc['objective'])}
            for m in metrics:
                rec[m + '_mean'] = mean(acc[m]); rec[m + '_ci95'] = ci(acc[m])
            rows.append(rec)
            print(f"{net:<14} {name:<17} obj={rec['objective_mean']:.4f}±{rec['objective_ci95']:.4f}  "
                  f"B={rec['avg_bandwidth_mean']:.3f}  miss={rec['missed_pct_mean']:.2f}±{rec['missed_pct_ci95']:.2f}  "
                  f"cvar95={rec['cvar95_loss_mean']:.4f}", flush=True)
    print("\n--- paired Wilcoxon: deployed vs ablated (r=|Z|/sqrt(n_nonzero)) ---")
    stat_rows = []
    for net in NETWORKS:
        base = per_runs[(net, 'deployed')]
        for name in VARIANTS:
            if name == 'deployed': continue
            alt = per_runs[(net, name)]
            do, po, ro, no = wilcox([r['objective'] for r in base], [r['objective'] for r in alt])
            dm, pm, rm, nm = wilcox([r['missed_pct'] for r in base], [r['missed_pct'] for r in alt])
            print(f"[{net}] {name:<17} obj d={do:+.4f} p={po:.3g} r={ro:.2f}(n={no}) | "
                  f"miss d={dm:+.3f}pp p={pm:.3g} r={rm:.2f}(n={nm})")
            stat_rows.append({'network': net, 'variant': name,
                              'obj_delta': do, 'obj_p': po, 'obj_r': ro, 'obj_nnz': no,
                              'miss_delta_pp': dm, 'miss_p': pm, 'miss_r': rm, 'miss_nnz': nm})
    with (OUT / 'rabs_q4_full_ablation.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT / 'rabs_q4_full_ablation_stats.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys())); w.writeheader(); w.writerows(stat_rows)
    print('WROTE', OUT / 'rabs_q4_full_ablation.csv')

if __name__ == '__main__':
    main()
