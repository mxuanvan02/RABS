#!/usr/bin/env python3
"""RABS scalability evaluation on the HARD ERA5 Vietnam replay.

Scaling is measured on real ERA5 hourly 2-m temperature (2024) for up to 20
Mekong-delta stations spanning the whole delta (province capitals and towns).
Each monitored zone is a distinct real location -- no synthetic or replicated
traces -- so larger N is a genuinely larger real-data deployment, not a
stress-test artefact.

Same violation-probability urgency channel and primal-dual budget rule as the
main ERA5 experiment. Stdlib-only, reproducible. Lower objective is better.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/era5_vn'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

# 20 real Mekong-delta stations, ordered so the first three match the main
# experiment. Each is a distinct real ERA5 location (see data/fetch_era5_vn.py).
STATIONS = ['can_tho', 'soc_trang', 'ca_mau', 'long_xuyen', 'rach_gia',
            'my_tho', 'ben_tre', 'vinh_long', 'tra_vinh', 'cao_lanh',
            'tan_an', 'bac_lieu', 'vi_thanh', 'chau_doc', 'ha_tien',
            'sa_dec', 'go_cong', 'nga_bay', 'duyen_hai', 'phu_quoc']
SAFE_MIN, SAFE_MAX = 22.0, 34.0
SIGMA = 1.2
NETWORKS = {'burst': (0.06, 0.035, 0.65, 0.22),
            'severe_burst': (0.08, 0.055, 0.82, 0.15)}
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59]
N_VALUES = [3, 8, 12, 20]
PER = 720
N_STARTS = 6


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def read_stations(n):
    """Read the first n real station streams (one distinct location each)."""
    cols = []
    for s in STATIONS[:n]:
        xs = []
        with (DATA / f'{s}_2024.csv').open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['temp_c'] not in ('', 'None'):
                    xs.append(float(r['temp_c']))
        cols.append(xs)
    T = min(len(c) for c in cols)
    return [[cols[zi][t] for zi in range(n)] for t in range(T)]


def zone_steps_real(n):
    """Build N-zone step records directly from N distinct real stations."""
    base = read_stations(n)
    zsteps = []
    for t in range(len(base)):
        row = []
        for zi in range(n):
            x = base[t][zi]
            p = phi((SAFE_MIN - x) / SIGMA) + (1.0 - phi((SAFE_MAX - x) / SIGMA))
            p = min(1.0, max(0.0, p))
            v = 1 if (x < SAFE_MIN or x > SAFE_MAX) else 0
            row.append({'x': x, 'mu': x, 'p': p, 'v': v})
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
        else:
            # Urgency channel = violation probability (VoI-style).
            score = 0.55 * true[i]['p'] + 0.25 * err + 0.20 * age_score
        scores.append(score)
    return sorted(range(N), key=lambda i: scores[i], reverse=True)[:B]


def eval_step(true, hat, N):
    loss = 0; tp = fp = tn = fn = 0
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


def budget_candidates(N):
    cand = {1, max(1, round(N / 8)), max(1, round(N / 4)),
            max(1, round(N / 2)), max(1, round(3 * N / 4)), N}
    return sorted(cand)


def choose_B_rabs_pd(true, hat, age, pbad, dual, N):
    R = risk_score(true, hat, age, pbad, N)
    best = (1e18, 1)
    for B in budget_candidates(N):
        _, loss, aoi, miss_risk = predict_candidate(true, hat, age, B, N)
        score = (loss + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * aoi
                 + (0.035 + dual['miss']) * miss_risk - 0.030 * R * B)
        if score < best[0]:
            best = (score, B)
    return best[1]


def run(zsteps, policy, network, seed, start, end, N):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in zsteps[start]]; age = [0] * N; counts = [0] * N
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    B_target = 0.52 * N
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []; okn = att = 0
    for k in range(start + 1, end):
        true = zsteps[k]
        if policy == 'rabs_pd':
            B = choose_B_rabs_pd(true, hat, age, pbad, dual, N)
        elif policy == 'fixed_full':
            B = N
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
            'objective': objective, 'avg_bandwidth': avgB,
            'bandwidth_saving_vs_full_pct': saving, 'missed_pct': missed, 'avg_aoi': avg_aoi}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    T0 = len(read_stations(3))
    starts = [i * PER for i in range(N_STARTS) if (i + 1) * PER <= T0]
    policies = ['fixed_full', 'max_aoi', 'rabs_pd']
    raw = []
    for N in N_VALUES:
        zsteps = zone_steps_real(N)
        for net in NETWORKS:
            for start in starts:
                for seed in SEEDS:
                    for pol in policies:
                        raw.append(run(zsteps, pol, net, seed, start, start + PER, N))
    metrics = ['objective', 'avg_bandwidth', 'bandwidth_saving_vs_full_pct', 'missed_pct', 'avg_aoi']
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
    with (OUT / 'rabs_era5_scaling_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_era5_scaling_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    print('N   pol         Obj      B_avg   Save%   Missed%')
    for N in N_VALUES:
        for pol in ['fixed_full', 'rabs_pd']:
            r = next(x for x in summ if x['N'] == N and x['network'] == 'severe_burst' and x['policy'] == pol)
            print(f"{N:<3} {pol:<11} {r['objective_mean']:.4f}  {r['avg_bandwidth_mean']:.2f}   "
                  f"{r['bandwidth_saving_vs_full_pct_mean']:.1f}   {r['missed_pct_mean']:.2f}")
    print('WROTE', OUT / 'rabs_era5_scaling_summary.csv')


if __name__ == '__main__':
    main()
