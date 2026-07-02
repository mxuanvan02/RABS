#!/usr/bin/env python3
"""RABS-CVaR cross-sectional tail-risk study across network size N.

KEY IDEA (derive-then-verify)
-----------------------------
Earlier attempts priced the CVaR of the *temporal* loss tail. On this
greenhouse replay that tail is light, so a temporal-CVaR term adds nothing
over VoU + primal-dual duals. The tail that actually matters here is
CROSS-SECTIONAL: at each slot the scheduler leaves (N - B) zones unpolled,
and the worst of those unpolled exit-risks is the real hazard. With N=3 that
"tail" is 1-2 points (noisy, little to gain); as N grows the unpolled set
grows and its upper tail becomes a well-populated, meaningful distribution.

RABS-CVaR therefore prices the Rockafellar-Uryasev CVaR_alpha of the
per-slot UNPOLLED risk set:
    CVaR_alpha(unpolled risks) = min_xi [ xi + (1-alpha)^-1 E (r - xi)_+ ]
evaluated per candidate budget b. This spends a marginal slot precisely when
a candidate would leave an unusually high-risk zone unattended -- cutting the
missed-violation tail. Prediction: the CVaR benefit (lower missed% and lower
cross-sectional CVaR vs plain RABS-PD) GROWS with N.

Real N=3 greenhouse streams anchor the study; N>3 use the same synthetic
scalability construction as run_rabs_scaling.py. Stdlib-only, reproducible,
nothing hardcoded to printed numbers. Lower is better on every column
except Avg. Bw.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/source/safety_probability_calibration_raw.csv'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

NETWORKS = {
    'burst':         (0.06, 0.035, 0.65, 0.22),
    'severe_burst':  (0.08, 0.055, 0.82, 0.15),
    'extreme_burst': (0.10, 0.075, 0.92, 0.10),
}
SAFE_MIN, SAFE_MAX = 22.0, 30.0
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59]
N_VALUES = [3, 8, 12, 20]
PER = 1000
N_STARTS = 6
ALPHA = 0.9
POLICIES = ['rabs_pd', 'rabs_cvar']


def read_base_steps():
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
    T = len(base)
    specs = []
    for j in range(N):
        src = j % 3
        offset = gen.randrange(T)
        bias = 0.0 if j < 3 else gen.uniform(-1.5, 1.5)
        specs.append((src, offset, bias))
    zsteps = []
    for t in range(T):
        row = []
        for (src, offset, bias) in specs:
            r = base[(t + offset) % T][src]
            x = r['x'] + bias
            mu = r['mu'] + bias
            v = 1 if (x < SAFE_MIN or x > SAFE_MAX) else 0
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


def proxy_error(records, hat, i):
    return min(abs(records[i]['mu'] - hat[i]) / 8.0, 1.0)


def risk_score(records, hat, age, pbad, N):
    ps = [r['p'] for r in records]
    err = [proxy_error(records, hat, i) for i in range(N)]
    maxp = max(ps); meanp = sum(ps) / N; meana = sum(age) / N
    return 0.45 * maxp + 0.20 * meanp + 0.18 * min(meana / 8.0, 1.0) + 0.10 * pbad + 0.07 * max(err)


def choose_sensors(true, hat, age, B, N):
    scores = []
    for i in range(N):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        vou = 4.0 * true[i]['p'] * (1.0 - true[i]['p'])
        scores.append(0.55 * vou + 0.25 * err + 0.20 * age_score)
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


def cvar_unpolled(true, idx, N, alpha):
    """Rockafellar-Uryasev CVaR_alpha of the exit-risk over UNPOLLED zones."""
    risks = sorted((true[i]['p'] for i in range(N) if i not in idx), reverse=True)
    if not risks:
        return 0.0
    k = max(1, int(round((1.0 - alpha) * len(risks))))
    return sum(risks[:k]) / k


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


def run(zsteps, policy, network, seed, start, end, N):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in zsteps[start]]; age = [0] * N; counts = [0] * N
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    B_target = 0.52 * N
    lam = 0.035
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    cvar_hist = []
    for k in range(start + 1, end):
        true = zsteps[k]
        R = risk_score(true, hat, age, pbad, N)
        best = (1e18, 1)
        for B in budget_candidates(N):
            idx_b, l, ao, miss_risk = predict_candidate(true, hat, age, B, N)
            base = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                    + (0.035 + dual['miss']) * miss_risk - 0.030 * R * B)
            if policy == 'rabs_cvar':
                # price the cross-sectional CVaR of the unpolled exit-risk
                base += lam * cvar_unpolled(true, idx_b, N, ALPHA)
            if base < best[0]:
                best = (base, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B, N)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad); counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat, N); loss.append(l)
        tp += a; fp += b; tn += c; fn += d
        # realized cross-sectional CVaR of unpolled risk this slot
        cs_cvar = cvar_unpolled(true, idx, N, ALPHA)
        cvar_hist.append(cs_cvar)
        mean_aoi = sum(age) / N; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - B_target))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        if policy == 'rabs_cvar':
            # trailing alpha-quantile of realized cross-sectional CVaR = VaR level
            win = cvar_hist[-100:]
            if len(win) >= 10:
                sw = sorted(win); xi = sw[min(len(sw) - 1, int(ALPHA * len(sw)))]
            else:
                xi = 0.0
            breach = 1.0 if cs_cvar > xi else 0.0
            lam = min(0.20, max(0.0, lam + 0.010 * (breach - (1.0 - ALPHA))))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    mean_loss = sum(loss) / len(loss)
    avg_aoi = sum(aoi) / len(aoi)
    objective = mean_loss + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    cs_cvar_mean = sum(cvar_hist) / len(cvar_hist)
    return {'policy': policy, 'network': network, 'N': N, 'seed': seed,
            'avg_bandwidth': avgB, 'mean_loss': mean_loss,
            'cs_cvar': cs_cvar_mean, 'objective': objective, 'missed_pct': missed}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    base = read_base_steps()
    starts = [i * PER for i in range(N_STARTS) if (i + 1) * PER <= len(base)]
    zone_gen = random.Random(2026)
    raw = []
    for N in N_VALUES:
        zsteps = make_zone_steps(base, N, zone_gen)
        for net in NETWORKS:
            for start in starts:
                for seed in SEEDS:
                    for pol in POLICIES:
                        raw.append(run(zsteps, pol, net, seed, start, start + PER, N))
    metrics = ['avg_bandwidth', 'mean_loss', 'cs_cvar', 'objective', 'missed_pct']
    summ = []
    for N in N_VALUES:
        for net in NETWORKS:
            for pol in POLICIES:
                sub = [r for r in raw if r['N'] == N and r['network'] == net and r['policy'] == pol]
                rec = {'N': N, 'network': net, 'policy': pol, 'n': len(sub)}
                for m in metrics:
                    vals = [r[m] for r in sub]
                    rec[m + '_mean'] = mean(vals); rec[m + '_ci95'] = ci(vals)
                summ.append(rec)
    with (OUT / 'rabs_cvar_scaling_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_cvar_scaling_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    print('=== CVaR benefit vs N (severe_burst): PD vs CVaR ===')
    print('N    pol        Bw      Missed%   CS-CVaR   Obj')
    for N in N_VALUES:
        for pol in POLICIES:
            r = next(x for x in summ if x['N'] == N and x['network'] == 'severe_burst' and x['policy'] == pol)
            print(f"{N:<4} {pol:<10} {r['avg_bandwidth_mean']:.3f}   {r['missed_pct_mean']:.3f}    "
                  f"{r['cs_cvar_mean']:.4f}   {r['objective_mean']:.4f}")
    print('WROTE', OUT / 'rabs_cvar_scaling_summary.csv')


if __name__ == '__main__':
    main()
