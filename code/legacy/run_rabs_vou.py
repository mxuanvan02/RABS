#!/usr/bin/env python3
"""RABS urgency-form study: derive-then-verify the value-of-uncertainty (VoU)
term against the deployed raw-p_vio heuristic.

MOTIVATION (theory, see manuscript):
The deployed urgency score uses w_risk * p_vio. But polling a sensor only has
value if the fresh reading can CHANGE the safety decision. When p_vio -> 1 the
detector already flags a violation, so an extra poll carries ~zero decision
value, yet raw p_vio ranks it highest -- and double-counts the p>=0.55 detector
rule inside eval_step. The Bayes-optimal leading-order value-of-information for
a binary safety decision is the DECISION UNCERTAINTY g(p) = p(1-p) (or the
binary entropy H(p)), which is maximal at the p~0.5 boundary and vanishes once
the outcome is certain.

This script compares 4 urgency forms x 2 budget modes (risk-adaptive c_R on/off)
across N in {3,8,12,20}. N=3 uses the REAL greenhouse streams; N>3 are the same
synthetic scalability stress test as run_rabs_scaling.py. Real data anchors the
result; nothing is hardcoded. Lower composite objective = better.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/source/safety_probability_calibration_raw.csv'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

NETWORKS = {'burst': (0.06, 0.035, 0.65, 0.22),
            'severe_burst': (0.08, 0.055, 0.82, 0.15)}
SAFE_MIN, SAFE_MAX = 22.0, 30.0
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59]
N_VALUES = [3, 8, 12, 20]
PER = 1000
N_STARTS = 6
DETECT_P = 0.55   # detector threshold used in eval_step

# urgency forms: how the "risk-ish" channel enters the per-sensor score.
# weights are (w_risk_channel, w_dev, w_aoi), renormalized to sum 1.
URGENCY_FORMS = ['pvio', 'norisk', 'vou', 'entropy']
BUDGET_MODES = ['risk_adaptive', 'no_risk_budget']   # c_R on / off


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


def jain(x):
    s = sum(x); ss = sum(v * v for v in x)
    return s * s / (len(x) * ss + 1e-12) if s else 1.0


def proxy_error(records, hat, i):
    return min(abs(records[i]['mu'] - hat[i]) / 8.0, 1.0)


def binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    import math
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def risk_channel(p, form):
    """Map violation probability -> per-sensor risk-channel value."""
    if form == 'pvio':
        return p                      # deployed heuristic
    if form == 'norisk':
        return 0.0                    # ablation: drop it
    if form == 'vou':
        return 4.0 * p * (1.0 - p)    # decision uncertainty, scaled to [0,1]
    if form == 'entropy':
        return binary_entropy(p)      # binary entropy, in [0,1]
    raise ValueError(form)


def risk_score(records, hat, age, pbad, N):
    ps = [r['p'] for r in records]
    err = [proxy_error(records, hat, i) for i in range(N)]
    maxp = max(ps); meanp = sum(ps) / N; meana = sum(age) / N
    return 0.45 * maxp + 0.20 * meanp + 0.18 * min(meana / 8.0, 1.0) + 0.10 * pbad + 0.07 * max(err)


def urgency_weights(form):
    # keep the deployed split; renormalize when risk channel is dropped
    wr, wd, wa = 0.55, 0.25, 0.20
    if form == 'norisk':
        wr = 0.0
    s = wr + wd + wa
    return wr / s, wd / s, wa / s


def choose_sensors(true, hat, age, B, N, form):
    wr, wd, wa = urgency_weights(form)
    scores = []
    for i in range(N):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        rc = risk_channel(true[i]['p'], form)
        scores.append(wr * rc + wd * err + wa * age_score)
    return sorted(range(N), key=lambda i: scores[i], reverse=True)[:B]


def eval_step(true, hat, N):
    loss = 0; tp = fp = tn = fn = 0
    for i in range(N):
        pred = 1 if (hat[i] < SAFE_MIN or hat[i] > SAFE_MAX or true[i]['p'] >= DETECT_P) else 0
        y = true[i]['v']
        if pred and y: tp += 1
        elif pred and not y: fp += 1
        elif not pred and y: fn += 1
        else: tn += 1
        loss += (abs(hat[i] - true[i]['x']) / 8.0) ** 2 + 5 * (1 if y and not pred else 0) + 1 * (1 if pred and not y else 0)
    return loss / N, tp, fp, tn, fn


def predict_candidate(true, hat, age, B, N, form):
    idx = choose_sensors(true, hat, age, B, N, form)
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


def run(zsteps, form, budget_mode, network, seed, start, end, N):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in zsteps[start]]; age = [0] * N; counts = [0] * N
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    B_target = 0.52 * N
    cR = 0.030 if budget_mode == 'risk_adaptive' else 0.0
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []; okn = att = 0
    for k in range(start + 1, end):
        true = zsteps[k]
        R = risk_score(true, hat, age, pbad, N)
        best = (1e18, 1)
        for B in budget_candidates(N):
            _, l, ao, miss_risk = predict_candidate(true, hat, age, B, N, form)
            score = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao
                     + (0.035 + dual['miss']) * miss_risk - cR * R * B)
            if score < best[0]: best = (score, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B, N, form)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad); att += 1; okn += 1 if ok else 0; counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat, N); loss.append(l); tp += a; fp += b; tn += c; fn += d
        mean_aoi = sum(age) / N; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - B_target))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    saving = 100 * (N - avgB) / N
    avg_aoi = sum(aoi) / len(aoi)
    objective = sum(loss) / len(loss) + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    sl = sorted(loss, reverse=True); kk = max(1, int(0.05 * len(sl)))
    cvar95 = sum(sl[:kk]) / kk
    return {'form': form, 'budget_mode': budget_mode, 'network': network, 'N': N,
            'seed': seed, 'objective': objective, 'avg_bandwidth': avgB,
            'bandwidth_saving_vs_full_pct': saving, 'missed_pct': missed,
            'avg_aoi': avg_aoi, 'cvar95_loss': cvar95,
            'packet_success_pct': 100 * okn / att if att else 100, 'fairness': jain(counts)}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    base = read_base_steps()
    starts = [i * PER for i in range(N_STARTS) if (i + 1) * PER <= len(base)]
    raw = []
    zone_gen = random.Random(2026)
    zone_cache = {N: make_zone_steps(base, N, zone_gen) for N in N_VALUES}
    for N in N_VALUES:
        zsteps = zone_cache[N]
        for net in NETWORKS:
            for start in starts:
                for seed in SEEDS:
                    for form in URGENCY_FORMS:
                        for bm in BUDGET_MODES:
                            raw.append(run(zsteps, form, bm, net, seed, start, start + PER, N))
    metrics = ['objective', 'avg_bandwidth', 'bandwidth_saving_vs_full_pct',
               'missed_pct', 'avg_aoi', 'cvar95_loss', 'packet_success_pct', 'fairness']
    summ = []
    for N in N_VALUES:
        for net in NETWORKS:
            for form in URGENCY_FORMS:
                for bm in BUDGET_MODES:
                    sub = [r for r in raw if r['N'] == N and r['network'] == net
                           and r['form'] == form and r['budget_mode'] == bm]
                    rec = {'N': N, 'network': net, 'form': form, 'budget_mode': bm, 'n': len(sub)}
                    for m in metrics:
                        vals = [r[m] for r in sub]
                        rec[m + '_mean'] = mean(vals); rec[m + '_ci95'] = ci(vals)
                    summ.append(rec)
    with (OUT / 'rabs_vou_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_vou_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    # Console view 1: urgency forms at risk_adaptive budgeting, severe_burst
    print('=== urgency form comparison (severe_burst, risk_adaptive budget) ===')
    print('N   form      Obj      Missed%  CVaR95   B_avg')
    for N in N_VALUES:
        for form in URGENCY_FORMS:
            r = next(x for x in summ if x['N'] == N and x['network'] == 'severe_burst'
                     and x['form'] == form and x['budget_mode'] == 'risk_adaptive')
            print(f"{N:<3} {form:<8}  {r['objective_mean']:.4f}   {r['missed_pct_mean']:.2f}"
                  f"     {r['cvar95_loss_mean']:.4f}   {r['avg_bandwidth_mean']:.2f}")
    # Console view 2: does risk-adaptive BUDGET (#1) matter? vou form, severe_burst
    print()
    print('=== risk-adaptive budgeting on/off (vou urgency, severe_burst) ===')
    print('N   budget_mode       Obj      Missed%  CVaR95')
    for N in N_VALUES:
        for bm in BUDGET_MODES:
            r = next(x for x in summ if x['N'] == N and x['network'] == 'severe_burst'
                     and x['form'] == 'vou' and x['budget_mode'] == bm)
            print(f"{N:<3} {bm:<16}  {r['objective_mean']:.4f}   {r['missed_pct_mean']:.2f}     {r['cvar95_loss_mean']:.4f}")
    print('WROTE', OUT / 'rabs_vou_summary.csv')


if __name__ == '__main__':
    main()
