#!/usr/bin/env python3
"""RABS tail-risk study on REAL ERA5 Vietnam climate data (hard regime).

Three Mekong-delta stations -- Can Tho, Soc Trang, Ca Mau -- act as the three
monitored zones (same N=3 layout described in the manuscript). Hourly 2-m
temperature for 2024 is fetched from the Open-Meteo ERA5 archive by
data/fetch_era5_vn.py and stored under data/era5_vn/. Unlike the mild-tail
greenhouse replay, this trace has genuine heat episodes (2-4% of hours above
34C, peaks ~38C), so the loss distribution is heavy-tailed and a tail-aware
allocator has room to help.

All three selectors share the SAME value-of-uncertainty (VoU) urgency ranking
and the SAME primal-dual bandwidth/AoI/miss duals as the main experiment; they
differ ONLY in how they price the missed-event risk:

    RABS-PD    : expected miss-risk penalty            (c_M + lambda^M) * Mhat(b)
    RABS-MV    : mean + dispersion of unpolled risk     + w_v * Var[unpolled p]
    RABS-CVaR  : Rockafellar-Uryasev CVaR_alpha of the unpolled exit-risk tail

Stdlib-only, reproducible. Lower is better on every column except Avg. Bw.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/era5_vn'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ['can_tho', 'soc_trang', 'ca_mau']
# Safety band for greenhouse-style monitoring: lower comfort bound 22C, upper
# heat-stress bound 34C. The upper bound is what the heat episodes breach.
SAFE_MIN, SAFE_MAX = 22.0, 34.0
SIGMA = 1.2            # predictor stddev used to turn temperature into p_vio
ALPHA = 0.90          # CVaR confidence level (CVaR_{0.9}, matches manuscript)

# Gilbert-Elliott channels of increasing tail severity.
NETWORKS = {
    'burst':         (0.06, 0.035, 0.65, 0.22),
    'severe_burst':  (0.08, 0.055, 0.82, 0.15),
    'extreme_burst': (0.10, 0.075, 0.92, 0.10),
}
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 720             # 30-day windows (hourly)
POLICIES = ['rabs_pd', 'rabs_mv', 'rabs_cvar']


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def read_stations():
    """Return steps[t] = [zone0, zone1, zone2] aligned across the 3 stations."""
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
            # violation probability that the true temperature is outside the band,
            # given a Gaussian predictor centred at x with stddev SIGMA.
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


def choose_sensors(true, hat, age, B):
    """VoU urgency: value-of-uncertainty risk channel g(p)=4p(1-p), same as main."""
    scores = []
    for i in range(3):
        err = proxy_error(true, hat, i)
        age_score = min(age[i] / 8.0, 1.0)
        vou = 4.0 * true[i]['p'] * (1.0 - true[i]['p'])
        scores.append(0.55 * vou + 0.25 * err + 0.20 * age_score)
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
    unpolled = [true[i]['p'] for i in range(3) if i not in idx]
    miss_risk = sum(unpolled) / 3
    return idx, l, pred_aoi, miss_risk, unpolled


def run(steps, policy, network, seed, start, end):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]; counts = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    lam = 0.035
    mr_hist = []
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    for k in range(start + 1, end):
        true = steps[k]
        R = risk_score(true, hat, age, pbad)
        # VaR level = empirical alpha-quantile of recent realized unpolled miss-risk.
        win = mr_hist[-100:]
        if len(win) >= 10:
            sw = sorted(win)
            xi = sw[min(len(sw) - 1, int(ALPHA * len(sw)))]
        else:
            xi = 0.0
        best = (1e18, 1)
        for B in [1, 2, 3]:
            _, l, ao, miss_risk, unpolled = predict_candidate(true, hat, age, B)
            base = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao - 0.030 * R * B)
            if policy == 'rabs_pd':
                score = base + (0.035 + dual['miss']) * miss_risk
            elif policy == 'rabs_mv':
                mv_var = statistics.pvariance(unpolled) if len(unpolled) > 1 else 0.0
                score = base + (0.035 + dual['miss']) * miss_risk + 0.80 * mv_var
            elif policy == 'rabs_cvar':
                # Rockafellar-Uryasev CVaR surrogate on the PREDICTED unpolled
                # exit-risk: keep the expected-miss penalty, then price the tail
                # exceedance beyond the VaR level xi. Spends budget precisely
                # when a candidate leaves an unusually high exit-risk unpolled.
                cvar_excess = max(0.0, miss_risk - xi) / (1.0 - ALPHA)
                score = base + (0.035 + dual['miss']) * miss_risk + lam * cvar_excess
            else:
                raise ValueError(policy)
            if score < best[0]:
                best = (score, B)
        B = best[1]
        idx = choose_sensors(true, hat, age, B)
        bus.append(B); age = [aa + 1 for aa in age]
        for i in idx:
            ok, bad = channel(network, rng, bad); counts[i] += 1
            pbad = min(0.99, max(0.01, 0.85 * pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i] = true[i]['x']; age[i] = 0
        l, a, b, c, d = eval_step(true, hat); loss.append(l)
        tp += a; fp += b; tn += c; fn += d
        realized_mr = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
        mr_hist.append(realized_mr)
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        if policy == 'rabs_cvar':
            breach = 1.0 if realized_mr > xi else 0.0
            lam = min(0.20, max(0.0, lam + 0.010 * (breach - (1.0 - ALPHA))))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    mean_loss = sum(loss) / len(loss)
    avg_aoi = sum(aoi) / len(aoi)
    objective = mean_loss + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    miss_risk_mean = fn / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    # tail metric: CVaR_alpha of the per-slot loss (mean of worst (1-alpha) frac)
    sl = sorted(loss, reverse=True)
    kk = max(1, int((1.0 - ALPHA) * len(sl)))
    cvar = sum(sl[:kk]) / kk
    return {'policy': policy, 'network': network, 'seed': seed,
            'avg_bandwidth': avgB, 'mean_loss': mean_loss, 'cvar': cvar,
            'miss_risk': miss_risk_mean, 'objective': objective, 'missed_pct': missed}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    steps = read_stations()
    starts = [i * PER for i in range(20) if (i + 1) * PER <= len(steps)]
    raw = []
    for net in NETWORKS:
        for start in starts:
            for seed in SEEDS:
                for pol in POLICIES:
                    raw.append(run(steps, pol, net, seed, start, start + PER))
    metrics = ['avg_bandwidth', 'mean_loss', 'cvar', 'miss_risk', 'objective', 'missed_pct']
    summ = []
    for net in NETWORKS:
        for pol in POLICIES:
            sub = [r for r in raw if r['network'] == net and r['policy'] == pol]
            rec = {'network': net, 'policy': pol, 'n': len(sub)}
            for m in metrics:
                vals = [r[m] for r in sub]
                rec[m + '_mean'] = mean(vals); rec[m + '_ci95'] = ci(vals)
            summ.append(rec)
    with (OUT / 'rabs_cvar_era5_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_cvar_era5_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    print(f'windows={len(starts)} seeds={len(SEEDS)} n/cell={len(starts)*len(SEEDS)}')
    print('net            pol         Bw     Loss    CVaR    MissRisk  Obj     Missed%')
    for net in NETWORKS:
        for pol in POLICIES:
            r = next(x for x in summ if x['network'] == net and x['policy'] == pol)
            print(f"{net:<14} {pol:<10} {r['avg_bandwidth_mean']:.3f}  {r['mean_loss_mean']:.4f}  "
                  f"{r['cvar_mean']:.4f}  {r['miss_risk_mean']:.4f}    {r['objective_mean']:.4f}  {r['missed_pct_mean']:.3f}")
    print('WROTE', OUT / 'rabs_cvar_era5_summary.csv')


if __name__ == '__main__':
    main()
