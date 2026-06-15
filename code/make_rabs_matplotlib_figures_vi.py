#!/usr/bin/env python3
"""Generate Vietnamese publication figures for the RABS manuscript.
All plots are regenerated from source CSV/data; no hand-edited values.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(ROOT / 'outputs' / 'rabs' / 'rabs_summary.csv')
gap = pd.read_csv(ROOT / 'outputs' / 'rabs' / 'rabs_oracle_gap.csv')
raw_src = pd.read_csv(ROOT / 'data' / 'source' / 'safety_probability_calibration_raw.csv')

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.5,
    'axes.titlesize': 11,
    'axes.labelsize': 9.5,
    'legend.fontsize': 8.2,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

policies = ['fixed_b1','fixed_b2','fixed_b3','aoi_adaptive','risk_adaptive','burst_adaptive','rabs','oracle_b']
labels = {
    'fixed_b1':'Cố định-B1','fixed_b2':'Cố định-B2','fixed_b3':'Cố định-B3',
    'aoi_adaptive':'Thích nghi AoI','risk_adaptive':'Thích nghi rủi ro','burst_adaptive':'Thích nghi burst',
    'rabs':'RABS','oracle_b':'Oracle-B','lyapunov_dp':'Lyapunov-DP','primal_dual':'Primal-dual'
}
colors = {
    'fixed_b1':'#bdbdbd','fixed_b2':'#8da0cb','fixed_b3':'#4c78a8',
    'aoi_adaptive':'#72b7b2','risk_adaptive':'#f58518','burst_adaptive':'#eeca3b',
    'rabs':'#54a24b','oracle_b':'#e45756','lyapunov_dp':'#9467bd','primal_dual':'#ff9da6'
}
net_labels={'burst':'Mất gói burst','severe_burst':'Mất gói burst nghiêm trọng'}

def savefig(name):
    for ext in ['pdf','png','svg']:
        plt.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    plt.close()

# 1. Handwritten academic overview, but text-controlled vector/Matplotlib.
def draw_box(ax, xy, wh, title, lines, fc, ec):
    x,y=xy; w,h=wh
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.045',fc=fc,ec=ec,lw=1.7)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h-0.09,title,ha='center',va='top',fontsize=10.5,fontweight='bold',color='#111827')
    for i,line in enumerate(lines):
        ax.text(x+0.04,y+h-0.18-0.075*i,line,ha='left',va='top',fontsize=8.5,color='#334155')

def arrow(ax, start, end, rad=0.0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.5,color='#334155',connectionstyle=f'arc3,rad={rad}'))

fig, ax = plt.subplots(figsize=(10.8,5.6))
ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.5,0.96,'Tổng quan kiến trúc RABS cho hệ cảm biến nhà kính',ha='center',va='top',fontsize=14,fontweight='bold')
draw_box(ax,(0.04,0.66),(0.21,0.20),'Dữ liệu cảm biến',['nhiệt độ, độ ẩm','giá trị lưu giữ $\\hat{x}_{i,t}$','tuổi thông tin a_i,t'],'#e0f2fe','#0284c7')
draw_box(ax,(0.34,0.67),(0.25,0.19),'Xây dựng trạng thái rủi ro',['xác suất vi phạm p_i,t','rủi ro cực đại / trung bình','niềm tin kênh xấu pi_t'],'#fef3c7','#d97706')
draw_box(ax,(0.68,0.67),(0.25,0.19),'Quyết định băng thông',['R_t < tau_1: dùng B_min','tau_1 <= R_t < tau_2: dùng B_mid','R_t >= tau_2: dùng B_max'],'#dcfce7','#16a34a')
draw_box(ax,(0.68,0.34),(0.25,0.19),'Xếp hạng cảm biến',['tính điểm ưu tiên $S_{i,t}$','chọn Top-$B_t$ cảm biến','truyền qua kênh burst'],'#ede9fe','#7c3aed')
draw_box(ax,(0.34,0.34),(0.25,0.19),'Cập nhật mạng',['mô hình Gilbert--Elliott','thành công/thất bại gói','reset hoặc tăng AoI'],'#fee2e2','#dc2626')
draw_box(ax,(0.04,0.34),(0.21,0.19),'Đánh giá thực nghiệm',['tổn thất an toàn','băng thông trung bình','AoI, fairness, F1'],'#f8fafc','#475569')
arrow(ax,(0.25,0.76),(0.34,0.76)); arrow(ax,(0.59,0.76),(0.68,0.76)); arrow(ax,(0.81,0.67),(0.81,0.53)); arrow(ax,(0.68,0.43),(0.59,0.43)); arrow(ax,(0.34,0.43),(0.25,0.43)); arrow(ax,(0.145,0.53),(0.145,0.66),rad=-0.25)
ax.text(0.5,0.18,'Logic chính: tăng ngân sách truyền khi rủi ro, độ cũ thông tin hoặc trạng thái kênh xấu tăng; giảm ngân sách khi hệ ổn định.',ha='center',fontsize=9.5,color='#334155')
savefig('fig_rabs_architecture_vi')

# 2. Temperature/time series sample.
sample = raw_src[raw_src['window'] == 0].copy()
fig, ax = plt.subplots(figsize=(9.2,3.6))
for loop, df in sample.groupby('loop'):
    df = df.sort_values('t').head(350)
    ax.plot(df['t'], df['xt'], label=f'Cảm biến {loop}', linewidth=1.1)
ax.axhline(22, color='#d62728', linestyle='--', linewidth=1.0, label='Ngưỡng an toàn')
ax.axhline(30, color='#d62728', linestyle='--', linewidth=1.0)
ax.set_title('Ví dụ chuỗi nhiệt độ/vi khí hậu nhà kính trong một cửa sổ dữ liệu')
ax.set_xlabel('Chỉ số thời gian')
ax.set_ylabel('Giá trị cảm biến')
ax.legend(ncol=4, frameon=False)
savefig('fig_greenhouse_temperature_sample')

# 3. Risk heatmap.
pivot = raw_src.pivot_table(index='loop', columns='window', values='p_gaussian', aggfunc='mean')
fig, ax = plt.subplots(figsize=(6.8,3.0))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
ax.set_xlabel('Cửa sổ dữ liệu')
ax.set_ylabel('Vòng cảm biến')
ax.set_title('Ma trận xác suất rủi ro trung bình theo cảm biến và cửa sổ')
cb = fig.colorbar(im, ax=ax)
cb.set_label('Xác suất rủi ro trung bình')
savefig('fig_risk_heatmap_loop_window')

# 4. Objective comparison.
fig, axes = plt.subplots(1, 2, figsize=(10.8,3.8), sharey=True)
for ax, net in zip(axes, ['burst','severe_burst']):
    df = summary[(summary.network == net) & (summary.policy.isin(policies))].copy()
    df['policy'] = pd.Categorical(df['policy'], policies, ordered=True)
    df = df.sort_values('policy')
    ax.bar([labels[p] for p in df.policy.astype(str)], df.objective_mean, color=[colors[p] for p in df.policy.astype(str)])
    ax.set_title(net_labels[net])
    ax.set_ylabel('Giá trị hàm mục tiêu')
    ax.tick_params(axis='x', rotation=32)
fig.suptitle('So sánh hàm mục tiêu giữa các chính sách')
savefig('fig_rabs_objective_comparison')

# 5. Safety-bandwidth tradeoff.
fig, axes = plt.subplots(1, 2, figsize=(10.8,3.8), sharey=True)
for ax, net in zip(axes, ['burst','severe_burst']):
    df = summary[(summary.network == net) & (summary.policy.isin(policies))]
    for _, r in df.iterrows():
        ax.scatter(r.avg_bandwidth_mean, r.loss_mean_mean, s=70, color=colors[r.policy], edgecolor='black', linewidth=0.6)
        ax.text(r.avg_bandwidth_mean + 0.018, r.loss_mean_mean, labels[r.policy], fontsize=7.5)
    ax.set_title(net_labels[net])
    ax.set_xlabel('Băng thông trung bình')
    ax.set_ylabel('Tổn thất an toàn')
fig.suptitle('Đánh đổi giữa an toàn và chi phí băng thông')
savefig('fig_rabs_safety_bandwidth_tradeoff')

# 6. Oracle gap.
fig, ax = plt.subplots(figsize=(5.2,3.4))
ax.bar([net_labels[n] for n in gap.network], gap.rabs_oracle_gap_pct, color='#e45756')
ax.set_ylabel('Khoảng cách mục tiêu so với Oracle-B (%)')
ax.set_title('Khoảng cách tối ưu hóa của RABS so với Oracle-B')
for i, v in enumerate(gap.rabs_oracle_gap_pct):
    ax.text(i, v + 0.6, f'{v:.1f}%', ha='center', fontsize=9)
savefig('fig_rabs_oracle_gap')

manifest=[]
for fid, typ, purpose, data in [
    ('fig_rabs_architecture_vi','architecture','Tổng quan kiến trúc và luồng quyết định RABS',[]),
    ('fig_greenhouse_temperature_sample','data_plot','Minh họa chuỗi nhiệt độ/vi khí hậu và ngưỡng an toàn',['data/source/safety_probability_calibration_raw.csv']),
    ('fig_risk_heatmap_loop_window','data_plot','Ma trận xác suất rủi ro trung bình theo cảm biến và cửa sổ',['data/source/safety_probability_calibration_raw.csv']),
    ('fig_rabs_objective_comparison','data_plot','So sánh hàm mục tiêu giữa chính sách',['outputs/rabs/rabs_summary.csv']),
    ('fig_rabs_safety_bandwidth_tradeoff','data_plot','Đánh đổi giữa tổn thất an toàn và băng thông trung bình',['outputs/rabs/rabs_summary.csv']),
    ('fig_rabs_oracle_gap','data_plot','Khoảng cách giữa RABS và Oracle-B',['outputs/rabs/rabs_oracle_gap.csv']),
]:
    manifest.append({
        'figure_id': fid,
        'type': typ,
        'purpose': purpose,
        'source_data': data,
        'script': 'code/make_rabs_matplotlib_figures_vi.py',
        'outputs': {'pdf': str(OUT/f'{fid}.pdf'), 'png': str(OUT/f'{fid}.png'), 'svg': str(OUT/f'{fid}.svg')},
        'language': 'Vietnamese labels/captions',
        'font': 'DejaVu Sans, 9.5--14 pt controlled by Matplotlib',
        'verification': {'file_signature_checked': False, 'render_preview_checked': False, 'text_checked': True, 'data_values_from_source_script': typ=='data_plot'}
    })
(OUT/'figures_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote Vietnamese figures to {OUT}')
