#!/usr/bin/env python3
from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
SUM=ROOT/'outputs/rabs/rabs_summary.csv'
FIG=ROOT/'outputs/figures'; FIG.mkdir(parents=True,exist_ok=True)
MAN=ROOT/'outputs/figures/figures_manifest_rabs.json'
rows=list(csv.DictReader(open(SUM,encoding='utf-8')))
colors={'fixed_b1':'#B8B8B8','fixed_b2':'#8DA0CB','fixed_b3':'#4C78A8','aoi_adaptive':'#72B7B2','risk_adaptive':'#F58518','burst_adaptive':'#ECA82C','rabs':'#54A24B','oracle_b':'#E45756'}
policies=['fixed_b1','fixed_b2','fixed_b3','aoi_adaptive','risk_adaptive','burst_adaptive','rabs','oracle_b']

def bar_svg(net, metric, ylabel, path, title):
    data=[r for r in rows if r['network']==net and r['policy'] in policies]
    W,H=900,420; m=70; bw=46; gap=24
    maxv=max(float(r[metric+'_mean']) for r in data)*1.15
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>']
    parts.append(f'<line x1="{m}" y1="{H-m}" x2="{W-m}" y2="{H-m}" stroke="#333"/>')
    parts.append(f'<line x1="{m}" y1="{m}" x2="{m}" y2="{H-m}" stroke="#333"/>')
    for i in range(6):
        val=maxv*i/5; y=H-m-(val/maxv)*(H-2*m)
        parts.append(f'<line x1="{m}" y1="{y:.1f}" x2="{W-m}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{m-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.2f}</text>')
    x=m+18
    for r in data:
        v=float(r[metric+'_mean']); h=(v/maxv)*(H-2*m); y=H-m-h; pol=r['policy']
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{h:.1f}" fill="{colors[pol]}"/>')
        parts.append(f'<text x="{x+bw/2}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{v:.2f}</text>')
        parts.append(f'<text x="{x+bw/2}" y="{H-m+16}" text-anchor="start" font-family="Arial" font-size="10" transform="rotate(35 {x+bw/2},{H-m+16})">{pol}</text>')
        x+=bw+gap
    parts.append(f'<text x="20" y="{H/2}" transform="rotate(-90 20,{H/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')

def scatter_svg(net,path,title):
    data=[r for r in rows if r['network']==net and r['policy'] in policies]
    W,H=720,480; m=70
    xs=[float(r['avg_bandwidth_mean']) for r in data]; ys=[float(r['loss_mean_mean']) for r in data]
    xmin,xmax=0.8,3.1; ymin=0; ymax=max(ys)*1.25
    def X(x): return m+(x-xmin)/(xmax-xmin)*(W-2*m)
    def Y(y): return H-m-(y-ymin)/(ymax-ymin)*(H-2*m)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',f'<line x1="{m}" y1="{H-m}" x2="{W-m}" y2="{H-m}" stroke="#333"/>',f'<line x1="{m}" y1="{m}" x2="{m}" y2="{H-m}" stroke="#333"/>']
    for r in data:
        pol=r['policy']; x=X(float(r['avg_bandwidth_mean'])); y=Y(float(r['loss_mean_mean']))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[pol]}" stroke="#333"/>')
        parts.append(f'<text x="{x+9:.1f}" y="{y+4:.1f}" font-family="Arial" font-size="11">{pol}</text>')
    parts.append(f'<text x="{W/2}" y="{H-18}" text-anchor="middle" font-family="Arial" font-size="13">Average bandwidth used</text>')
    parts.append(f'<text x="20" y="{H/2}" transform="rotate(-90 20,{H/2})" text-anchor="middle" font-family="Arial" font-size="13">Safety loss</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')

def overview(path):
    W,H=1100,620
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>',
           '<text x="550" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">RABS: Risk-Adaptive Bandwidth Scaling for Smart Greenhouse NCS</text>']
    boxes=[('Greenhouse sensors','x_i,t, risk p_i,t',70,110),('Network state','burst belief pi_bad,t',70,260),('Information age','AoI a_i,t',70,410),('System risk R_t','max risk + mean risk + AoI + burst + error',390,250),('Bandwidth scaling','B_t = 1 / 2 / 3',690,250),('Top-B update','send selected sensors; update x_hat and AoI',910,250)]
    for title,sub,x,y in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="190" height="88" rx="12" fill="#F7F9FC" stroke="#4C78A8" stroke-width="2"/>')
        parts.append(f'<text x="{x+95}" y="{y+30}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold">{title}</text>')
        parts.append(f'<text x="{x+95}" y="{y+58}" text-anchor="middle" font-family="Arial" font-size="12">{sub}</text>')
    def arrow(x1,y1,x2,y2,label=''):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
        if label:
            parts.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    arrow(260,154,390,280)
    arrow(260,304,390,294)
    arrow(260,454,390,308)
    arrow(580,294,690,294,'compute')
    arrow(880,294,910,294,'allocate')
    parts.append('<rect x="390" y="410" width="710" height="70" rx="10" fill="#FFF7E6" stroke="#F58518" stroke-width="1.5"/>')
    parts.append('<text x="745" y="438" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold">Online objective</text>')
    parts.append('<text x="745" y="463" text-anchor="middle" font-family="Arial" font-size="13">reduce safety loss while controlling bandwidth use and information staleness under bursty packet loss</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')

bar_svg('burst','objective','Objective',FIG/'fig_rabs_objective_burst.svg','Objective comparison under burst loss')
bar_svg('severe_burst','objective','Objective',FIG/'fig_rabs_objective_severe.svg','Objective comparison under severe burst loss')
bar_svg('burst','avg_bandwidth','Average bandwidth',FIG/'fig_rabs_bandwidth_burst.svg','Bandwidth usage under burst loss')
scatter_svg('burst',FIG/'fig_rabs_tradeoff_burst.svg','Safety-bandwidth trade-off under burst loss')
scatter_svg('severe_burst',FIG/'fig_rabs_tradeoff_severe.svg','Safety-bandwidth trade-off under severe burst loss')
overview(FIG/'fig_rabs_overview.svg')
manifest=[]
for fid,typ,purpose,files in [
 ('fig_rabs_overview','architecture','Overview of RABS data flow and decisions',['fig_rabs_overview.svg']),
 ('fig_rabs_objective_burst','data_plot','Objective comparison under burst loss',['fig_rabs_objective_burst.svg']),
 ('fig_rabs_objective_severe','data_plot','Objective comparison under severe burst loss',['fig_rabs_objective_severe.svg']),
 ('fig_rabs_bandwidth_burst','data_plot','Average bandwidth use under burst loss',['fig_rabs_bandwidth_burst.svg']),
 ('fig_rabs_tradeoff_burst','data_plot','Safety-bandwidth tradeoff under burst loss',['fig_rabs_tradeoff_burst.svg']),
 ('fig_rabs_tradeoff_severe','data_plot','Safety-bandwidth tradeoff under severe burst loss',['fig_rabs_tradeoff_severe.svg'])]:
    manifest.append({'figure_id':fid,'type':typ,'purpose':purpose,'source_data':['outputs/rabs/rabs_summary.csv'] if typ=='data_plot' else ['manuscript/method equations'], 'script':'code/make_rabs_figures.py','outputs':{'svg':[str(FIG/f) for f in files]},'verification':{'file_exists':all((FIG/f).exists() for f in files),'text_controlled_svg':True,'data_values_from_csv':typ=='data_plot','manuscript_consistency_checked':False}})
MAN.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('generated', len(manifest), 'figures')

# Oracle gap figure
import csv as _csv
_gap=list(_csv.DictReader(open(ROOT/'outputs/rabs/rabs_oracle_gap.csv',encoding='utf-8')))
W,H=680,380; m=70
maxv=max(float(r['rabs_oracle_gap_pct']) for r in _gap)*1.25
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<text x="340" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Online RABS-to-oracle objective gap</text>',f'<line x1="{m}" y1="{H-m}" x2="{W-m}" y2="{H-m}" stroke="#333"/>',f'<line x1="{m}" y1="{m}" x2="{m}" y2="{H-m}" stroke="#333"/>']
x=m+90
for r in _gap:
    v=float(r['rabs_oracle_gap_pct']); h=(v/maxv)*(H-2*m); y=H-m-h
    parts.append(f'<rect x="{x}" y="{y:.1f}" width="90" height="{h:.1f}" fill="#E45756"/>')
    parts.append(f'<text x="{x+45}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{v:.1f}%</text>')
    parts.append(f'<text x="{x+45}" y="{H-m+20}" text-anchor="middle" font-family="Arial" font-size="12">{r["network"]}</text>')
    x+=190
parts.append(f'<text x="24" y="{H/2}" transform="rotate(-90 24,{H/2})" text-anchor="middle" font-family="Arial" font-size="13">Objective gap (%)</text>')
parts.append('</svg>')
(FIG/'fig_rabs_oracle_gap.svg').write_text('\n'.join(parts),encoding='utf-8')
