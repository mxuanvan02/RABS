#!/usr/bin/env python3
"""RABS family: adaptive bandwidth scaling for smart greenhouse NCS.

Stdlib-only reproducible simulation using safety_probability_calibration_raw.csv.
The main contribution variant is RABS-PD, a primal-dual constrained
online selector that adapts safety, AoI, and bandwidth penalties over time.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/source/safety_probability_calibration_raw.csv'
OUT=ROOT/'outputs/rabs'; OUT.mkdir(parents=True,exist_ok=True)
OUTT=ROOT/'outputs/tables'; OUTT.mkdir(parents=True,exist_ok=True)
OUTF=ROOT/'outputs/figures'; OUTF.mkdir(parents=True,exist_ok=True)
NETWORKS={'burst':(0.06,0.035,0.65,0.22),'severe_burst':(0.08,0.055,0.82,0.15)}
POLICIES=['fixed_b1','fixed_b2','fixed_b3',
          'aoi_adaptive','risk_adaptive','burst_adaptive',
          'max_aoi','max_risk','voi_b2','event_triggered','channel_aware',
          'rabs_h','rabs_l','rabs_pd','oracle_b']
SAFE_MIN,SAFE_MAX=22.0,30.0
SEEDS=[7,11,17,23,29,31,41,43,53,59,61,67,71,73,79,83,89,97,101,103]


def read_steps():
    d={}
    with SRC.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            w=int(float(r['window'])); t=int(float(r['t'])); loop=int(float(r['loop']))
            rec={'window':w,'t':t,'loop':loop,'x':float(r['xt']),'mu':float(r['mu']),'p':float(r['p_gaussian']),'v':int(float(r['true_violation']))}
            d.setdefault((w,t),{})[loop]=rec
    steps=[]
    for key in sorted(d):
        if len(d[key])==3:
            steps.append([d[key][i] for i in range(3)])
    return steps


def channel(kind,rng,bad):
    gl,pgb,bl,pbg=NETWORKS[kind]
    if bad:
        ok=rng.random()>bl
        if rng.random()<pbg: bad=False
    else:
        ok=rng.random()>gl
        if rng.random()<pgb: bad=True
    return ok,bad


def jain(x):
    s=sum(x); ss=sum(v*v for v in x)
    return s*s/(len(x)*ss+1e-12) if s else 1.0


def proxy_error(records,hat,i):
    """Pre-transmission mismatch proxy; never uses current untransmitted x."""
    return min(abs(records[i]['mu']-hat[i])/8.0,1.0)


def risk_score(records,hat,age,pbad):
    ps=[r['p'] for r in records]
    err=[proxy_error(records,hat,i) for i in range(3)]
    maxp=max(ps); meanp=sum(ps)/3; meana=sum(age)/3
    return 0.45*maxp+0.20*meanp+0.18*min(meana/8.0,1.0)+0.10*pbad+0.07*max(err)


def choose_B_threshold(policy,true,hat,age,pbad):
    R=risk_score(true,hat,age,pbad)
    if policy=='fixed_b1': return 1,R
    if policy=='fixed_b2': return 2,R
    if policy=='fixed_b3': return 3,R
    if policy=='aoi_adaptive':
        m=sum(age)/3
        return (1 if m<2 else 2 if m<5 else 3),R
    if policy=='risk_adaptive':
        mp=max(r['p'] for r in true)
        return (1 if mp<0.55 else 2 if mp<0.75 else 3),R
    if policy=='burst_adaptive':
        return (1 if pbad<0.25 else 2 if pbad<0.55 else 3),R
    if policy in ['max_aoi','max_risk','voi_b2']:
        return 2,R
    if policy=='event_triggered':
        triggers=0
        for i in range(3):
            err=proxy_error(true,hat,i)
            if true[i]['p']>=0.55 or err>=0.25 or age[i]>=5:
                triggers+=1
        return max(1,min(3,triggers)),R
    if policy=='channel_aware':
        # Avoid spending many attempts during inferred bad-channel periods.
        return (3 if pbad<0.25 else 2 if pbad<0.55 else 1),R
    if policy=='rabs_h':
        return (1 if R<0.38 else 2 if R<0.54 else 3),R
    raise ValueError(policy)


def choose_sensors(true,hat,age,B,policy='rabs'):
    scores=[]
    for i in range(3):
        err=proxy_error(true,hat,i)
        age_score=min(age[i]/8.0,1.0)
        if policy=='max_aoi':
            score=age[i]
        elif policy=='max_risk':
            score=true[i]['p']
        elif policy=='voi_b2':
            # Deployable VoI-style benchmark: value rises with risk, state mismatch, and staleness.
            score=true[i]['p']*(0.65*err+0.35*age_score)
        elif policy=='event_triggered':
            triggered=1.0 if (true[i]['p']>=0.55 or err>=0.25 or age[i]>=5) else 0.0
            score=triggered*(0.50*true[i]['p']+0.30*err+0.20*age_score)
        elif policy=='channel_aware':
            score=0.45*true[i]['p']+0.35*err+0.20*age_score
        else:
            score=0.55*true[i]['p']+0.25*err+0.20*age_score
        scores.append(score)
    return sorted(range(3), key=lambda i:scores[i], reverse=True)[:B]


def eval_step(true,hat):
    loss=0; tp=fp=tn=fn=0
    for i in range(3):
        pred=1 if (hat[i]<SAFE_MIN or hat[i]>SAFE_MAX or true[i]['p']>=0.55) else 0
        y=true[i]['v']
        if pred and y: tp+=1
        elif pred and not y: fp+=1
        elif not pred and y: fn+=1
        else: tn+=1
        loss += (abs(hat[i]-true[i]['x'])/8.0)**2 + 5*(1 if y and not pred else 0) + 1*(1 if pred and not y else 0)
    return loss/3,tp,fp,tn,fn


def predict_candidate(true,hat,age,B):
    idx=choose_sensors(true,hat,age,B)
    h=hat[:]
    for i in idx:
        # Candidate scoring is pre-transmission: use baseline/proxy value, not current x.
        h[i]=true[i]['mu']
    l,tp,fp,tn,fn=eval_step(true,h)
    pred_aoi=sum((0 if i in idx else age[i]+1) for i in range(3))/3
    miss_risk=sum(true[i]['p'] for i in range(3) if i not in idx)/3
    return idx,l,pred_aoi,miss_risk


def oracle_candidate(true,hat,age,B):
    idx=choose_sensors(true,hat,age,B)
    h=hat[:]
    for i in idx:
        h[i]=true[i]['x']
    l,tp,fp,tn,fn=eval_step(true,h)
    pred_aoi=sum((0 if i in idx else age[i]+1) for i in range(3))/3
    miss_risk=sum(true[i]['p'] for i in range(3) if i not in idx)/3
    return idx,l,pred_aoi,miss_risk


def choose_B_rabs_family(policy,true,hat,age,pbad,dual):
    R=risk_score(true,hat,age,pbad)
    best=(1e18,1)
    for B in [1,2,3]:
        _,loss,aoi,miss_risk=predict_candidate(true,hat,age,B)
        if policy=='rabs_l':
            score=loss + 0.048*B + 0.016*aoi - 0.024*R*B
        elif policy=='rabs_pd':
            # Primal-dual selector: dual variables tighten safety/AoI/bandwidth constraints online.
            score=(loss
                   + (0.030+dual['bw'])*B
                   + (0.010+dual['aoi'])*aoi
                   + (0.035+dual['miss'])*miss_risk
                   - 0.030*R*B)
        else:
            raise ValueError(policy)
        if score<best[0]: best=(score,B)
    return best[1],R


def run_fixed(steps,policy,network,seed,start,end):
    rng=random.Random(seed); bad=False
    pbad=NETWORKS[network][1]/(NETWORKS[network][1]+NETWORKS[network][3])
    hat=[r['x'] for r in steps[start]]; age=[0,0,0]; counts=[0,0,0]
    dual={'bw':0.0,'aoi':0.0,'miss':0.0}
    loss=[]; tp=fp=tn=fn=0; bus=[]; aoi=[]; okn=att=0; switches=0; lastB=None; burst_recovery=[]; in_bad_run=0
    for k in range(start+1,end):
        true=steps[k]
        if policy=='oracle_b':
            best=(1e9,1)
            for B in [1,2,3]:
                _,l,ao,miss_risk=oracle_candidate(true,hat,age,B)
                obj=l+0.04*B+0.010*ao+0.030*miss_risk
                if obj<best[0]: best=(obj,B)
            B=best[1]; R=risk_score(true,hat,age,pbad)
        elif policy in ['rabs_l','rabs_pd']:
            B,R=choose_B_rabs_family(policy,true,hat,age,pbad,dual)
        else:
            B,R=choose_B_threshold(policy,true,hat,age,pbad)
        if lastB is not None and B!=lastB: switches+=1
        lastB=B
        idx=choose_sensors(true,hat,age,B,policy)
        bus.append(B); age=[aa+1 for aa in age]
        step_fail=False
        for i in idx:
            ok,bad=channel(network,rng,bad); att+=1; okn+=1 if ok else 0; counts[i]+=1
            step_fail = step_fail or (not ok)
            pbad=min(0.99,max(0.01,0.85*pbad + (0.15 if not ok else -0.05)))
            if ok: hat[i]=true[i]['x']; age[i]=0
        l,a,b,c,d=eval_step(true,hat); loss.append(l); tp+=a; fp+=b; tn+=c; fn+=d
        mean_aoi=sum(age)/3; miss_rate_step=d/(a+d) if (a+d) else 0.0
        if policy=='rabs_pd':
            dual['bw']=max(0.0, dual['bw'] + 0.010*(B-1.55))
            dual['aoi']=max(0.0, dual['aoi'] + 0.006*(mean_aoi-1.60))
            dual['miss']=max(0.0, dual['miss'] + 0.020*(miss_rate_step-0.008))
        if step_fail:
            in_bad_run+=1
        elif in_bad_run:
            burst_recovery.append(max(age)); in_bad_run=0
        aoi.extend(age)
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    avgB=sum(bus)/len(bus); saving=100*(3-avgB)/3; avg_aoi=sum(aoi)/len(aoi)
    objective=sum(loss)/len(loss)+0.04*avgB+0.015*avg_aoi
    missed=100*fn/(tp+fn) if tp+fn else 0
    return {'policy':policy,'network':network,'seed':seed,'window_start':start,'objective':objective,'loss_mean':sum(loss)/len(loss),'avg_bandwidth':avgB,'bandwidth_saving_vs_b3_pct':saving,'missed_pct':missed,'false_alarm_pct':100*fp/(fp+tn) if fp+tn else 0,'precision':precision,'recall':recall,'f1':f1,'avg_aoi':avg_aoi,'max_aoi':max(aoi),'fairness':jain(counts),'packet_success_pct':100*okn/att if att else 100,'constraint_violation_pct':missed,'switching_rate_pct':100*switches/max(1,len(bus)-1),'recovery_aoi':sum(burst_recovery)/len(burst_recovery) if burst_recovery else 0.0}


def mean(xs): return sum(xs)/len(xs)
def ci(xs): return 1.96*statistics.stdev(xs)/(len(xs)**0.5) if len(xs)>1 else 0

def tex_escape(x): return str(x).replace('_', '\\_')


def write_svg(summary):
    policies=['fixed_b2','fixed_b3','rabs_h','rabs_l','rabs_pd','oracle_b']
    W,H=820,360; margin=60
    rows=[r for r in summary if r['policy'] in policies and r['network']=='burst']
    maxv=max(r['objective_mean'] for r in rows)*1.1
    barw=48; gap=34; x=margin
    colors={'fixed_b2':'#8da0cb','fixed_b3':'#66c2a5','rabs_h':'#fc8d62','rabs_l':'#e78ac3','rabs_pd':'#a6d854','oracle_b':'#ffd92f'}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16">RABS family objective under burst packet loss</text>']
    parts.append(f'<line x1="{margin}" y1="{H-margin}" x2="{W-margin}" y2="{H-margin}" stroke="black"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{H-margin}" stroke="black"/>')
    for r in rows:
        h=(r['objective_mean']/maxv)*(H-2*margin); y=H-margin-h
        pol=r['policy']
        parts.append(f'<rect x="{x}" y="{y}" width="{barw}" height="{h}" fill="{colors[pol]}" stroke="#333"/>')
        parts.append(f'<text x="{x+barw/2}" y="{H-margin+16}" text-anchor="middle" font-size="10" transform="rotate(25 {x+barw/2},{H-margin+16})">{pol}</text>')
        parts.append(f'<text x="{x+barw/2}" y="{y-5}" text-anchor="middle" font-size="10">{r["objective_mean"]:.3f}</text>')
        x+=barw+gap
    parts.append('<text x="18" y="190" font-size="12" transform="rotate(-90 18,190)">Objective</text></svg>')
    (OUTF/'fig_rabs_family_objective.svg').write_text('\n'.join(parts),encoding='utf-8')


def summarize(raw,metrics,split=None):
    out=[]
    for net in NETWORKS:
        for pol in POLICIES:
            sub=[r for r in raw if r['network']==net and r['policy']==pol]
            if split=='windows_4_7':
                sub=[r for r in sub if int(r['window_start'])>=4*2015]
            rec={'network':net,'policy':pol,'n':len(sub)}
            if split: rec['split']=split
            for m in metrics:
                vals=[float(r[m]) for r in sub]; rec[m+'_mean']=mean(vals); rec[m+'_ci95']=ci(vals)
            out.append(rec)
    return out


def write_tables(summary,holdout):
    display=['fixed_b1','fixed_b2','fixed_b3','max_aoi','max_risk','voi_b2','event_triggered','channel_aware','rabs_h','rabs_l','rabs_pd','oracle_b']
    lines=['\\begin{tabular}{llrrrrrrrr}','\\toprule','Mạng & Chính sách & Obj. & Loss & $\\bar{B}$ & Save (\\%) & Missed (\\%) & AoI & Viol. (\\%) & Switch (\\%) \\\\','\\midrule']
    for r in summary:
        if r['policy'] in display:
            lines.append(f"{tex_escape(r['network'])} & {tex_escape(r['policy'])} & {r['objective_mean']:.4f} & {r['loss_mean_mean']:.4f} & {r['avg_bandwidth_mean']:.2f} & {r['bandwidth_saving_vs_b3_pct_mean']:.1f} & {r['missed_pct_mean']:.2f} & {r['avg_aoi_mean']:.2f} & {r['constraint_violation_pct_mean']:.2f} & {r['switching_rate_pct_mean']:.1f} "+r"\\")
    lines+=['\\bottomrule','\\end{tabular}']
    (OUTT/'table_rabs_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    hdisplay=['fixed_b2','fixed_b3','max_aoi','max_risk','voi_b2','event_triggered','channel_aware','rabs_h','rabs_l','rabs_pd','oracle_b']
    lines=['\\begin{tabular}{llrrrrrr}','\\toprule','Mạng & Chính sách & Obj. & $\\bar{B}$ & Save (\\%) & Missed (\\%) & AoI & Viol. (\\%) \\\\','\\midrule']
    for r in holdout:
        if r['policy'] in hdisplay:
            lines.append(f"{tex_escape(r['network'])} & {tex_escape(r['policy'])} & {r['objective_mean']:.4f} & {r['avg_bandwidth_mean']:.2f} & {r['bandwidth_saving_vs_b3_pct_mean']:.1f} & {r['missed_pct_mean']:.2f} & {r['avg_aoi_mean']:.2f} & {r['constraint_violation_pct_mean']:.2f} "+r"\\")
    lines+=['\\bottomrule','\\end{tabular}']
    (OUTT/'table_rabs_holdout_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def by(pol,net): return next(r for r in summary if r['policy']==pol and r['network']==net)
    gap=['\\begin{tabular}{lrrrrrr}','\\toprule','Kịch bản & Obj. RABS-H & Obj. RABS-PD & Obj. Oracle & Gap H (\\%) & Gap PD (\\%) & $\\bar{B}$ PD \\\\','\\midrule']
    for net in NETWORKS:
        h=by('rabs_h',net); pd=by('rabs_pd',net); o=by('oracle_b',net)
        gh=100*(h['objective_mean']-o['objective_mean'])/o['objective_mean']
        gp=100*(pd['objective_mean']-o['objective_mean'])/o['objective_mean']
        gap.append(f"{tex_escape(net)} & {h['objective_mean']:.4f} & {pd['objective_mean']:.4f} & {o['objective_mean']:.4f} & {gh:.1f} & {gp:.1f} & {pd['avg_bandwidth_mean']:.2f} "+r"\\")
    gap+=['\\bottomrule','\\end{tabular}']
    (OUTT/'table_rabs_oracle_gap.tex').write_text('\n'.join(gap)+'\n',encoding='utf-8')


def main():
    steps=read_steps(); per=1000; starts=[i*per for i in range(0,16) if (i+1)*per<=len(steps)]
    raw=[]
    for net in NETWORKS:
        for start in starts:
            for seed in SEEDS:
                for pol in POLICIES:
                    raw.append(run_fixed(steps,pol,net,seed,start,start+per))
    fields=list(raw[0].keys())
    with (OUT/'rabs_raw.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(raw)
    metrics=['objective','loss_mean','avg_bandwidth','bandwidth_saving_vs_b3_pct','missed_pct','false_alarm_pct','precision','recall','f1','avg_aoi','max_aoi','fairness','packet_success_pct','constraint_violation_pct','switching_rate_pct','recovery_aoi']
    summary=summarize(raw,metrics)
    with (OUT/'rabs_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    holdout=summarize(raw,metrics,'windows_4_7')
    with (OUT/'rabs_holdout_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(holdout[0].keys())); w.writeheader(); w.writerows(holdout)
    write_tables(summary,holdout); write_svg(summary)
    print(OUT/'rabs_summary.csv')
    print(OUT/'rabs_holdout_summary.csv')

if __name__=='__main__': main()
