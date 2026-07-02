#!/usr/bin/env python3
"""RABS scalability stress test: does the primal-dual budget rule keep its
safety--bandwidth advantage as the number of monitored zones N grows?

The public greenhouse replay contains only N=3 measured zones. To probe
scaling we build SYNTHETIC multi-zone traces by replaying the three real
streams with independent circular time offsets and small per-zone thermal
biases, then recomputing violations against the safety band. This is a
stress test of the scheduler, NOT a claim of additional real data.

Stdlib-only, reproducible. Reuses the exact RABS-PD selector logic from
run_rabs_adaptive_bandwidth.py, generalized from hardcoded 3 to arbitrary N.
"""
from __future__ import annotations
import csv, random, statistics, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/source/safety_probability_calibration_raw.csv'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)
OUTT = ROOT / 'outputs/tables'; OUTT.mkdir(parents=True, exist_ok=True)

NETWORKS = {'burst': (0.06, 0.035, 0.65, 0.22),
            'severe_burst': (0.08, 0.055, 0.82, 0.15)}
SAFE_MIN, SAFE_MAX = 22.0, 30.0
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43]      # 8 seeds (scaling sweep)
N_VALUES = [3, 8, 12, 20]
PER = 1000
N_STARTS = 6


def read_base_steps():
    """Return the 3 real zone streams as steps[t] = [rec0, rec1, rec2]."""
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


def make_zone_steps(base, N, gen):
    """Synthesize N zone streams from the 3 real ones.

    Zone j uses real stream (j % 3), circularly shifted in time by a random
    offset, with a small constant thermal bias b_j. Violations are recomputed
    from the biased temperature against [SAFE_MIN, SAFE_MAX]; p is clamped.
    """
    T = len(base)
    specs = []
    for j in range(N):
        src = j % 3
        offset = gen.randrange(T)
        bias = 0.0 if j < 3 else gen.uniform(-1.5, 1.5)  # first 3 zones = real
        specs.append((src, offset, bias))
    zsteps = []
    for t in range(T):
        row = []
        for (src, offset, bias) in specs:
            r = base[(t + offset) % T][src]
            x = r['x'] + bias
            mu = r['mu'] + bias
            v = 1 if (x < SAFE_MIN or x > SAFE_MAX) else 0
            # shift violation probability with the bias-driven proximity to band
            p = r['p']
            if bias != 0.0:
                margin = min(x - SAFE_MIN, SAFE_MAX - x)
                p = max(0.0, min(1.0, p + max(0.0, (1.0 - margin / 4.0)) * 0.15))
            row.append({'x': x, 'mu': mu, 'p': p, 'v': v})
        zsteps.append(row)
    return zsteps


def channel(kind, rng, bad):
    gl, pgb, bl, pbg = NETWORKS[kind]
    if bad:
        ok = rng.random() > bl
        if rng.random() < pbg: bad = False
    else:
        ok = rng.random() > gl
        if rng.random() < pgb: bad = True
    return ok, bad


def jain(x):
    s = sum(x); ss = sum(v * v for v in x)
    return s * s / (len(x) * ss + 1e-12) if s else 1.0


def proxy_error(records, hat, i):
    return min(abs(records[i]['mu'] - hat[i]) / 8.0, 1.0)


def risk_score(records, hat, age, pbad, N):
    ps = [r['p'] for r in records]
    err = [proxy_error(records, hat, i) for i in range(N)]
    maxp = max(ps); meanp = sum(ps) / N; meana = sum(age) / N
    return 0.45 * maxp + 0.20 * meanp + 0.18 * min(meana / 8.0, 1.0) + 0.10 * pbad + 0.07 * max(err)


def choose_sensors(true, hat, age, B, N, policy='rabs'):
    scores = []
    for i in range(N):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        if policy == 'max_aoi':
            score = age[i]
        else:  # rabs urgency
            score = 0.55 * true[i]['p'] + 0.25 * err + 0.20 * age_score
        scores.append(score)
    return sorted(range(N), key=lambda i: scores[i], reverse=True)[:B]


def eval_step(true, hat, N):
    loss = 0
    tp = fp = tn = fn = 0
    for i in range(N):
        pred = 1 if (hat[i] < SAFE_MIN or hat[i] > SAFE_MAX or true[i]['p'] >= 0.55) else 0
        y = true[i]['v']
        if pred and y: tp += 1
        elif pred and not y: fp += 1
        elif not pred and y: fn += 1
        else: tn += 1
        loss += (abs(hat[i] - true[i]['x']) / 8.0) ** 2 + 5 * (1 if y and not pred else 0) + 1 * (1 if pred and not y else 0)
    return loss / N, tp, fp, tn, fn


def predict_candidate(true, hat, age, B, N):
    idx = choose_sensors(true, hat, age, B, N)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['mu']
    l, tp, fp, tn, fn = eval_step(true, h, N)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(N)) / N
    miss_risk = sum(true[i]['p'] for i in range(N) if i not in idx) / N
    return idx, l, pred_aoi, miss_risk


def oracle_candidate(true, hat, age, B, N):
    idx = choose_sensors(true, hat, age, B, N)
    h = hat[:]
    for i in idx:
        h[i] = true[i]['x']
    l, tp, fp, tn, fn = eval_step(true, h, N)
    pred_aoi = sum((0 if i in idx else age[i] + 1) for i in range(N)) / N
    miss_risk = sum(true[i]['p'] for i in range(N) if i not in idx) / N
    return idx, l, pred_aoi, miss_risk


def budget_candidates(N):
    """Modest candidate set spanning sparse to full polling."""
    cand = {1, max(1, round(N / 8)), max(1, round(N / 4)),
            max(1, round(N / 2)), max(1, round(3 * N / 4)), N}
    return sorted(cand)


def choose_B_rabs_pd(true, hat, age, pbad, dual, N, B_target):
    R = risk_score(true, hat, age, pbad, N)
    best = (1e18, 1)
    for B in budget_candidates(N):
        _, loss, aoi, miss_risk = predict_candidate(true, hat, age, B, N)
        score = (loss
                 + (0.030 + dual['bw']) * B
                 + (0.010 + dual['aoi']) * aoi
                 + (0.035 + dual['miss']) * miss_risk
                 - 0.030 * R * B)
        if score < best[0]:
            best = (score, B)
    return best[1], R


def run(zsteps, policy, network, seed, start, end, N):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in zsteps[start]]; age = [0] * N; counts = [0] * N
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    B_target = 0.52 * N       # same budget fraction as the N=3 paper (1.55/3)
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []; okn = att = 0
    for k in range(start + 1, end):
        true = zsteps[k]
        if policy == 'oracle_b':
            best = (1e9, 1)
            for B in budget_candidates(N):
                _, l, ao, miss_risk = oracle_candidate(true, hat, age, B, N)
                obj = l + 0.04 * B + 0.010 * ao + 0.030 * miss_risk
                if obj < best[0]: best = (obj, B)
            B = best[1]
        elif policy == 'rabs_pd':
            B, _ = choose_B_rabs_pd(true, hat, age, pbad, dual, N, B_target)
        elif policy == 'fixed_full':
            B = N
        elif policy == 'fixed_half':
            B = max(1, round(N / 2))
        elif policy == 'max_aoi':
            B = max(1, round(N / 2))
        else:
            raise ValueError(policy)
        idx = choose_sensors(true, hat, age, B, N, policy)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad); att += 1; okn += 1 if ok else 0; counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat, N); loss.append(l); tp += a; fp += b; tn += c; fn += d
        mean_aoi = sum(age) / N; miss_rate_step = d / (a + d) if (a + d) else 0.0
        if policy == 'rabs_pd':
            dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - B_target))
            dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
            dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    saving = 100 * (N - avgB) / N
    avg_aoi = sum(aoi) / len(aoi)
    objective = sum(loss) / len(loss) + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    return {'policy': policy, 'network': network, 'N': N, 'seed': seed,
            'objective': objective, 'loss_mean': sum(loss) / len(loss),
            'avg_bandwidth': avgB, 'bandwidth_saving_vs_full_pct': saving,
            'missed_pct': missed, 'avg_aoi': avg_aoi,
            'packet_success_pct': 100 * okn / att if att else 100,
            'fairness': jain(counts)}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    base = read_base_steps()
    starts = [i * PER for i in range(N_STARTS) if (i + 1) * PER <= len(base)]
    policies = ['fixed_full', 'fixed_half', 'max_aoi', 'rabs_pd', 'oracle_b']
    raw = []
    zone_gen = random.Random(2026)  # fixed seed for synthetic zone layout
    for N in N_VALUES:
        zsteps = make_zone_steps(base, N, zone_gen)
        for net in NETWORKS:
            for start in starts:
                for seed in SEEDS:
                    for pol in policies:
                        raw.append(run(zsteps, pol, net, seed, start, start + PER, N))
    # aggregate
    metrics = ['objective', 'avg_bandwidth', 'bandwidth_saving_vs_full_pct',
               'missed_pct', 'avg_aoi', 'packet_success_pct', 'fairness']
    summ = []
    for N in N_VALUES:
        for net in NETWORKS:
            for pol in policies:
                sub = [r for r in raw if r['N'] == N and r['network'] == net and r['policy'] == pol]
                rec = {'N': N, 'network': net, 'policy': pol, 'n': len(sub)}
                for m in metrics:
                    vals = [r[m] for r in sub]
                    rec[m + '_mean'] = mean(vals); rec[m + '_ci95'] = ci(vals)
                summ.append(rec)
    with (OUT / 'rabs_scaling_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_scaling_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    # console view: severe_burst, RABS-PD vs full polling
    print('N   pol          Obj      B_avg   Save%   Missed%  AoI')
    for N in N_VALUES:
        for pol in ['fixed_full', 'rabs_pd', 'oracle_b']:
            r = next(x for x in summ if x['N'] == N and x['network'] == 'severe_burst' and x['policy'] == pol)
            print(f"{N:<3} {pol:<12} {r['objective_mean']:.4f}  {r['avg_bandwidth_mean']:.2f}   "
                  f"{r['bandwidth_saving_vs_full_pct_mean']:.1f}   {r['missed_pct_mean']:.2f}    {r['avg_aoi_mean']:.2f}")
    print('WROTE', OUT / 'rabs_scaling_summary.csv')


if __name__ == '__main__':
    main()
