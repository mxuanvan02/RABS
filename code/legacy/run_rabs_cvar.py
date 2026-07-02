#!/usr/bin/env python3
"""RABS tail-risk variants: RABS-CVaR vs RABS-PD and RABS-MV.

Reproducible, stdlib-only. Uses the SAME real greenhouse replay
(safety_probability_calibration_raw.csv) and the SAME value-of-uncertainty
(VoU) urgency channel as run_rabs_adaptive_bandwidth.py, so the tail-risk
study is consistent with the main results.

Three Gilbert-Elliott channels of increasing tail severity act as a
hard-regime stress test:
    burst < severe_burst < extreme_burst
For each, we compare three budget selectors that share the identical VoU
urgency ranking and primal-dual bandwidth/AoI duals, differing ONLY in how
they price the missed-event risk term:

    RABS-PD    : expected miss-risk penalty      (c_M + lambda^M) * Mhat(b)
    RABS-MV    : mean + variance penalty         c_M*Mhat + c_V*Var[loss tail]
    RABS-CVaR  : Rockafellar-Uryasev CVaR_alpha  lambda*[xi + (1-alpha)^-1 (L-xi)_+]

CVaR / VaR follow the Rockafellar-Uryasev (2000) identity, with xi (the VaR
level) and the dual lambda updated online by projected subgradient steps.
Lower is better on every reported column except Avg. Bw.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/source/safety_probability_calibration_raw.csv'
OUT = ROOT / 'outputs/rabs'; OUT.mkdir(parents=True, exist_ok=True)
OUTT = ROOT / 'outputs/tables'; OUTT.mkdir(parents=True, exist_ok=True)

# (good->bad loss, P(good->bad), bad-state loss, P(bad->good))
# increasing tail severity: longer/heavier bad runs.
NETWORKS = {
    'burst':         (0.06, 0.035, 0.65, 0.22),
    'severe_burst':  (0.08, 0.055, 0.82, 0.15),
    'extreme_burst': (0.10, 0.075, 0.92, 0.10),
}
SAFE_MIN, SAFE_MAX = 22.0, 30.0
SEEDS = [7, 11, 17, 23, 29, 31, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
PER = 1000
ALPHA = 0.9          # CVaR confidence level (matches manuscript CVaR_{0.9})
POLICIES = ['rabs_pd', 'rabs_mv', 'rabs_cvar']


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
    miss_risk = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
    return idx, l, pred_aoi, miss_risk


def run(steps, policy, network, seed, start, end):
    rng = random.Random(seed); bad = False
    pbad = NETWORKS[network][1] / (NETWORKS[network][1] + NETWORKS[network][3])
    hat = [r['x'] for r in steps[start]]; age = [0, 0, 0]; counts = [0, 0, 0]
    dual = {'bw': 0.0, 'aoi': 0.0, 'miss': 0.0}
    # CVaR state. The tail-heavy quantity in this system is the unpolled
    # miss-risk during burst episodes (channel bad -> repeated losses -> AoI
    # spikes -> exit-risk piles up), NOT the average tracking loss. RABS-CVaR
    # therefore prices the tail of the MISS-RISK: xi is the empirical
    # alpha-quantile of recent realized miss-risk, and lam is the tail dual.
    lam = 0.035
    mr_hist = []
    loss = []; tp = fp = tn = fn = 0; bus = []; aoi = []
    for k in range(start + 1, end):
        true = steps[k]
        R = risk_score(true, hat, age, pbad)
        # VaR level = empirical alpha-quantile of recent realized miss-risk.
        win = mr_hist[-100:]
        if len(win) >= 10:
            sw = sorted(win)
            xi = sw[min(len(sw) - 1, int(ALPHA * len(sw)))]
        else:
            xi = 0.0
        best = (1e18, 1)
        for B in [1, 2, 3]:
            _, l, ao, miss_risk = predict_candidate(true, hat, age, B)
            base = (l + (0.030 + dual['bw']) * B + (0.010 + dual['aoi']) * ao - 0.030 * R * B)
            if policy == 'rabs_pd':
                score = base + (0.035 + dual['miss']) * miss_risk
            elif policy == 'rabs_mv':
                # mean-variance on the PREDICTED miss-risk across candidate budgets:
                # penalize both the expected miss-risk and its dispersion vs the
                # per-slot maximum-risk sensor left unpolled (candidate-dependent).
                unpolled = [true[i]['p'] for i in range(3)
                            if i not in choose_sensors(true, hat, age, B)]
                mv_var = statistics.pvariance(unpolled) if len(unpolled) > 1 else 0.0
                tail_gap = (max(unpolled) - miss_risk) if unpolled else 0.0
                score = (base + (0.035 + dual['miss']) * miss_risk
                         + 0.80 * mv_var + 0.30 * tail_gap)
            elif policy == 'rabs_cvar':
                # Rockafellar-Uryasev CVaR surrogate on the PREDICTED miss-risk:
                # keep the expected-miss penalty, then add the tail exceedance
                # of miss-risk beyond the VaR level xi. This spends budget
                # precisely when a candidate leaves an unusually high exit-risk
                # unpolled (burst onset), cutting missed violations in the tail.
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
        # realized unpolled miss-risk this slot = exit-risk left on unpolled zones
        realized_mr = sum(true[i]['p'] for i in range(3) if i not in idx) / 3
        mr_hist.append(realized_mr)
        mean_aoi = sum(age) / 3; miss_rate_step = d / (a + d) if (a + d) else 0.0
        dual['bw'] = max(0.0, dual['bw'] + 0.010 * (B - 1.55))
        dual['aoi'] = max(0.0, dual['aoi'] + 0.006 * (mean_aoi - 1.60))
        dual['miss'] = max(0.0, dual['miss'] + 0.020 * (miss_rate_step - 0.008))
        if policy == 'rabs_cvar':
            # xi is the trailing alpha-quantile of realized miss-risk (top of loop).
            # Adapt the tail dual lam: raise it when realized miss-risk breaches
            # the VaR level (tail event), decay otherwise. Projected to [0,0.20].
            breach = 1.0 if realized_mr > xi else 0.0
            lam = min(0.20, max(0.0, lam + 0.010 * (breach - (1.0 - ALPHA))))
        aoi.extend(age)
    avgB = sum(bus) / len(bus)
    mean_loss = sum(loss) / len(loss)
    avg_aoi = sum(aoi) / len(aoi)
    objective = mean_loss + 0.04 * avgB + 0.015 * avg_aoi
    missed = 100 * fn / (tp + fn) if tp + fn else 0
    miss_risk_mean = fn / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    # tail metrics
    sl = sorted(loss, reverse=True)
    kk = max(1, int((1.0 - ALPHA) * len(sl)))
    cvar = sum(sl[:kk]) / kk
    return {'policy': policy, 'network': network, 'seed': seed,
            'avg_bandwidth': avgB, 'mean_loss': mean_loss, 'cvar': cvar,
            'miss_risk': miss_risk_mean, 'objective': objective,
            'missed_pct': missed}


def mean(xs): return sum(xs) / len(xs)
def ci(xs): return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0


def main():
    steps = read_steps()
    starts = [i * PER for i in range(16) if (i + 1) * PER <= len(steps)]
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
    with (OUT / 'rabs_cvar_raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with (OUT / 'rabs_cvar_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    print('net            pol         Bw     Loss    CVaR    MissRisk  Obj     Missed%')
    for net in NETWORKS:
        for pol in POLICIES:
            r = next(x for x in summ if x['network'] == net and x['policy'] == pol)
            print(f"{net:<14} {pol:<10} {r['avg_bandwidth_mean']:.3f}  {r['mean_loss_mean']:.4f}  "
                  f"{r['cvar_mean']:.4f}  {r['miss_risk_mean']:.4f}    {r['objective_mean']:.4f}  {r['missed_pct_mean']:.3f}")
    print('WROTE', OUT / 'rabs_cvar_summary.csv')


if __name__ == '__main__':
    main()
