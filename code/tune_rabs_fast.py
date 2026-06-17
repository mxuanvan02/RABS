#!/usr/bin/env python3
from __future__ import annotations
import csv, itertools, random, statistics
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'code/run_rabs_adaptive_bandwidth.py'
spec=importlib.util.spec_from_file_location('rabs',MOD); rabs=importlib.util.module_from_spec(spec); spec.loader.exec_module(rabs)
OUT=ROOT/'outputs/rabs_tuning'; OUT.mkdir(parents=True,exist_ok=True)
steps=rabs.read_steps(); per=2015; starts=[i*per for i in range(4,8) if (i+1)*per<=len(steps)]
seeds=[7,11,17,23,29,31,41,43,53,59]

def run_custom(t1,t2):
    # monkey patch choose_B only for rabs policy
    orig=rabs.choose_B
    def choose_B(policy,true,hat,age,pbad,network,bmin=1,bmax=3):
        if policy!='rabs': return orig(policy,true,hat,age,pbad,network,bmin,bmax)
        R=rabs.risk_score(true,hat,age,pbad)
        return (1 if R<t1 else 2 if R<t2 else 3),R
    rabs.choose_B=choose_B
    vals=[]
    for net in rabs.NETWORKS:
        for start in starts:
            for seed in seeds:
                vals.append(rabs.run_fixed(steps,'rabs',net,seed,start,start+per))
    rabs.choose_B=orig
    return vals

def avg(vals,key): return sum(float(v[key]) for v in vals)/len(vals)
records=[]
for t1,t2 in itertools.product([0.38,0.42,0.46,0.50,0.54,0.58,0.62],[0.54,0.58,0.62,0.66,0.70,0.74,0.78]):
    if t2<=t1+0.06: continue
    vals=run_custom(t1,t2)
    records.append({'t1':t1,'t2':t2,'objective_avg':avg(vals,'objective'),'loss_avg':avg(vals,'loss_mean'),'bandwidth_avg':avg(vals,'avg_bandwidth'),'missed_avg':avg(vals,'missed_pct'),'aoi_avg':avg(vals,'avg_aoi')})
records=sorted(records,key=lambda x:x['objective_avg'])
with (OUT/'rabs_threshold_grid_fast.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
best=records[0]
print('BEST',best)
# best per network summary rows
vals=run_custom(best['t1'],best['t2'])
summary=[]
for net in rabs.NETWORKS:
    sub=[v for v in vals if v['network']==net]
    rec={'network':net,'policy':'rabs_tuned','n':len(sub),'t1':best['t1'],'t2':best['t2']}
    for k in ['objective','loss_mean','avg_bandwidth','bandwidth_saving_vs_b3_pct','missed_pct','false_alarm_pct','precision','recall','f1','avg_aoi','max_aoi','fairness','packet_success_pct']:
        xs=[float(v[k]) for v in sub]
        rec[k+'_mean']=sum(xs)/len(xs)
        rec[k+'_ci95']=1.96*statistics.stdev(xs)/(len(xs)**0.5) if len(xs)>1 else 0
    summary.append(rec)
with (OUT/'rabs_tuned_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
