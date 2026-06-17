#!/usr/bin/env python3
"""Publication-style Matplotlib figures for the RABS paper.
Requires numpy, pandas, matplotlib. Not runnable in the current minimal env.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'figures_matplotlib'
OUT.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(ROOT / 'outputs' / 'rabs' / 'rabs_summary.csv')
gap = pd.read_csv(ROOT / 'outputs' / 'rabs' / 'rabs_oracle_gap.csv')
raw_src = pd.read_csv(ROOT / 'data' / 'source' / 'safety_probability_calibration_raw.csv')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

policies = ['fixed_b1','fixed_b2','fixed_b3','aoi_adaptive','risk_adaptive','burst_adaptive','rabs','oracle_b']
colors = {
    'fixed_b1':'#bdbdbd','fixed_b2':'#8da0cb','fixed_b3':'#4c78a8',
    'aoi_adaptive':'#72b7b2','risk_adaptive':'#f58518','burst_adaptive':'#eeca3b',
    'rabs':'#54a24b','oracle_b':'#e45756'
}

def savefig(name):
    for ext in ['png','pdf','svg']:
        plt.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    plt.close()

# 1. Objective comparison, both networks
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, net in zip(axes, ['burst','severe_burst']):
    df = summary[(summary.network == net) & (summary.policy.isin(policies))].copy()
    df['policy'] = pd.Categorical(df['policy'], policies, ordered=True)
    df = df.sort_values('policy')
    ax.bar(df.policy.astype(str), df.objective_mean, color=[colors[p] for p in df.policy.astype(str)])
    ax.set_title(f'Objective under {net}')
    ax.set_ylabel('Objective')
    ax.tick_params(axis='x', rotation=35)
fig.suptitle('RABS objective comparison')
savefig('fig_rabs_objective_comparison')

# 2. Safety-bandwidth tradeoff
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, net in zip(axes, ['burst','severe_burst']):
    df = summary[(summary.network == net) & (summary.policy.isin(policies))]
    for _, r in df.iterrows():
        ax.scatter(r.avg_bandwidth_mean, r.loss_mean_mean, s=70, color=colors[r.policy], edgecolor='black')
        ax.text(r.avg_bandwidth_mean + 0.015, r.loss_mean_mean, r.policy, fontsize=8)
    ax.set_title(net)
    ax.set_xlabel('Average bandwidth')
    ax.set_ylabel('Safety loss')
fig.suptitle('Safety-bandwidth trade-off')
savefig('fig_rabs_safety_bandwidth_tradeoff')

# 3. Oracle gap
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.bar(gap.network, gap.rabs_oracle_gap_pct, color='#e45756')
ax.set_ylabel('Objective gap to Oracle-B (%)')
ax.set_title('RABS-to-oracle optimization gap')
for i, v in enumerate(gap.rabs_oracle_gap_pct):
    ax.text(i, v + 0.6, f'{v:.1f}%', ha='center')
savefig('fig_rabs_oracle_gap')

# 4. Temperature/time-series sample from full data
sample = raw_src[(raw_src['window'] == 0)].copy()
fig, ax = plt.subplots(figsize=(10, 4))
for loop, df in sample.groupby('loop'):
    df = df.sort_values('t').head(350)
    ax.plot(df['t'], df['xt'], label=f'Loop {loop}', linewidth=1.2)
ax.axhline(22, color='red', linestyle='--', linewidth=1, label='Safety bounds')
ax.axhline(30, color='red', linestyle='--', linewidth=1)
ax.set_title('Greenhouse sensor measurements, sample window')
ax.set_xlabel('Time index')
ax.set_ylabel('Measured value xt')
ax.legend(ncol=4)
savefig('fig_greenhouse_temperature_sample')

# 5. Risk/AoI heatmap-like matrix from processed data
pivot = raw_src.pivot_table(index='loop', columns='window', values='p_gaussian', aggfunc='mean')
fig, ax = plt.subplots(figsize=(7, 2.8))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
ax.set_xlabel('Window')
ax.set_ylabel('Sensor loop')
ax.set_title('Mean Gaussian safety-risk probability by loop/window')
fig.colorbar(im, ax=ax, label='Mean risk probability')
savefig('fig_risk_heatmap_loop_window')

print(f'Wrote figures to {OUT}')
