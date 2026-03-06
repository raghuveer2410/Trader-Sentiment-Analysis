"""
Primetrade.ai – Trader Performance vs Market Sentiment
Full analysis script
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import gzip, warnings, os

warnings.filterwarnings('ignore')
os.makedirs('/home/claude/charts', exist_ok=True)

# ─────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────
PALETTE = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#bcbd22',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#1f77b4',
}
SIMPLE = {'Fear-zone': '#d62728', 'Greed-zone': '#2ca02c'}

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
fg = pd.read_csv('/mnt/user-data/uploads/1772807974423_fear_greed_index.csv')
with gzip.open('/mnt/user-data/uploads/1772808509931_historical_data_csv.gz') as f:
    df = pd.read_csv(f)

print(f"Fear/Greed: {fg.shape[0]:,} rows × {fg.shape[1]} cols")
print(f"Trader data: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Missing values – Fear/Greed: {fg.isnull().sum().sum()} | Trader: {df.isnull().sum().sum()}")
print(f"Duplicates – Fear/Greed: {fg.duplicated().sum()} | Trader: {df.duplicated().sum()}")

# ─────────────────────────────────────────
# 2. CLEAN & ALIGN
# ─────────────────────────────────────────
fg['date'] = pd.to_datetime(fg['date']).dt.date
df['date'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True).dt.date

# Binary sentiment
def to_binary(c):
    return 'Fear-zone' if 'Fear' in str(c) else ('Greed-zone' if 'Greed' in str(c) else 'Neutral')

fg['binary'] = fg['classification'].apply(to_binary)

# Merge trader rows with sentiment
merged = df.merge(fg[['date','classification','binary','value']], on='date', how='inner')
print(f"\nAfter merge: {merged.shape[0]:,} rows | date range: {merged['date'].min()} → {merged['date'].max()}")
print(f"Unique accounts: {merged['Account'].nunique()}")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
# Identify closing trades (where PnL is realised)
close_dirs = {'Close Long', 'Close Short', 'Sell', 'Buy',
              'Short > Long', 'Long > Short', 'Settlement', 'Auto-Deleveraging'}
merged['is_close'] = merged['Direction'].isin(close_dirs)

# Long/Short flag on opening trades
merged['is_long_open']  = merged['Direction'].isin({'Open Long',  'Short > Long'})
merged['is_short_open'] = merged['Direction'].isin({'Open Short', 'Long > Short'})

# ── Daily account-level metrics ──────────────────────────────────────────────
daily = (merged.groupby(['date','Account','classification','binary','value'])
         .agg(
             total_pnl       = ('Closed PnL', 'sum'),
             trade_count     = ('Trade ID',   'count'),
             avg_size_usd    = ('Size USD',   'mean'),
             total_volume    = ('Size USD',   'sum'),
             long_opens      = ('is_long_open',  'sum'),
             short_opens     = ('is_short_open', 'sum'),
             close_trades    = ('is_close',      'sum'),
             win_trades      = ('Closed PnL',    lambda x: (x > 0).sum()),
             total_fee       = ('Fee',        'sum'),
         )
         .reset_index())

daily['win_rate']         = daily['win_trades'] / daily['close_trades'].clip(lower=1)
daily['long_short_ratio'] = daily['long_opens'] / (daily['short_opens'] + 1e-9)
daily['net_pnl_after_fee']= daily['total_pnl'] - daily['total_fee']

# ── Per-trade leverage proxy (Size USD / |Start Position| in USD equivalent) ─
# Leverage is not directly given; use Crossed flag as a proxy (cross = higher risk)
merged['is_cross'] = merged['Crossed'].astype(bool)

cross_rate = (merged.groupby(['date','Account'])['is_cross']
              .mean().reset_index().rename(columns={'is_cross':'cross_rate'}))
daily = daily.merge(cross_rate, on=['date','Account'], how='left')

print("\nDaily metrics sample:")
print(daily.head(3).to_string())

# ─────────────────────────────────────────
# 4. TRADER SEGMENTS
# ─────────────────────────────────────────
trader_stats = (merged.groupby('Account')
                .agg(
                    total_pnl     = ('Closed PnL', 'sum'),
                    trade_count   = ('Trade ID',   'count'),
                    avg_size_usd  = ('Size USD',   'mean'),
                    win_trades    = ('Closed PnL', lambda x: (x > 0).sum()),
                    close_trades  = ('is_close',   'sum'),
                    cross_rate    = ('is_cross',   'mean'),
                )
                .reset_index())

trader_stats['win_rate']   = trader_stats['win_trades'] / trader_stats['close_trades'].clip(1)
trader_stats['trades_per_day'] = trader_stats['trade_count'] / merged['date'].nunique()

# Segment 1: High vs Low leverage (cross margin usage)
lev_med = trader_stats['cross_rate'].median()
trader_stats['lev_seg'] = np.where(trader_stats['cross_rate'] >= lev_med,
                                   'High-Leverage', 'Low-Leverage')

# Segment 2: Frequent vs Infrequent traders
freq_med = trader_stats['trade_count'].median()
trader_stats['freq_seg'] = np.where(trader_stats['trade_count'] >= freq_med,
                                    'Frequent', 'Infrequent')

# Segment 3: Winners vs Losers
trader_stats['perf_seg'] = np.where(trader_stats['total_pnl'] > 0,
                                    'Net Winners', 'Net Losers')

print("\nSegment breakdown:")
for col in ['lev_seg','freq_seg','perf_seg']:
    print(f"  {col}: {trader_stats[col].value_counts().to_dict()}")

# Merge segments back
daily = daily.merge(trader_stats[['Account','lev_seg','freq_seg','perf_seg',
                                   'trades_per_day']], on='Account', how='left')

# ─────────────────────────────────────────
# 5. ANALYSIS
# ─────────────────────────────────────────

# ── A. PnL by sentiment classification ───────────────────────────────────────
pnl_by_sent = daily.groupby('classification')['total_pnl'].agg(['mean','median','std','count'])
print("\nMean daily PnL by sentiment:")
print(pnl_by_sent.sort_values('mean').to_string())

winrate_by_sent = daily.groupby('classification')['win_rate'].mean()
print("\nMean win-rate by sentiment:")
print(winrate_by_sent.sort_values())

# ── B. Behavior metrics by binary sentiment ───────────────────────────────────
beh = daily.groupby('binary').agg(
    avg_trades   = ('trade_count',    'mean'),
    avg_size_usd = ('avg_size_usd',   'mean'),
    avg_volume   = ('total_volume',   'mean'),
    avg_ls_ratio = ('long_short_ratio','mean'),
    avg_cross    = ('cross_rate',     'mean'),
    avg_win_rate = ('win_rate',       'mean'),
    avg_pnl      = ('total_pnl',      'mean'),
    n            = ('total_pnl',      'count'),
).reset_index()
print("\nBehavior by Fear vs Greed:")
print(beh.to_string())

# T-test: PnL Fear vs Greed
fear_pnl  = daily[daily['binary']=='Fear-zone']['total_pnl']
greed_pnl = daily[daily['binary']=='Greed-zone']['total_pnl']
t, p = stats.ttest_ind(fear_pnl, greed_pnl)
print(f"\nT-test PnL Fear vs Greed: t={t:.3f}, p={p:.4f}")

# ─────────────────────────────────────────
# 6. CHARTS
# ─────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
ORDER = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']

# ── Chart 1: PnL distribution by sentiment ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 1 – Daily PnL by Sentiment', fontsize=14, fontweight='bold', y=1.02)

# mean PnL bar
means = daily.groupby('classification')['total_pnl'].mean().reindex(ORDER)
colors = [PALETTE[c] for c in ORDER]
axes[0].bar(ORDER, means.values, color=colors, edgecolor='white', linewidth=0.5)
axes[0].axhline(0, color='black', lw=0.8, ls='--')
axes[0].set_title('Mean Daily PnL per Account')
axes[0].set_ylabel('USD')
axes[0].set_xticklabels(ORDER, rotation=15, ha='right')
for i, (v, l) in enumerate(zip(means.values, ORDER)):
    axes[0].text(i, v + (50 if v >= 0 else -120), f'${v:.0f}', ha='center', fontsize=8)

# win rate bar
wr = daily.groupby('classification')['win_rate'].mean().reindex(ORDER)
axes[1].bar(ORDER, wr.values*100, color=colors, edgecolor='white', linewidth=0.5)
axes[1].axhline(50, color='black', lw=0.8, ls='--')
axes[1].set_title('Mean Win-Rate per Account (%)')
axes[1].set_ylabel('%')
axes[1].set_xticklabels(ORDER, rotation=15, ha='right')
for i, v in enumerate(wr.values):
    axes[1].text(i, v*100+0.5, f'{v*100:.1f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/charts/chart1_pnl_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 1")

# ── Chart 2: Behavior – trades, size, long/short by binary sentiment ──────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Chart 2 – Trader Behavior: Fear vs Greed Days', fontsize=14, fontweight='bold')

metrics = [
    ('trade_count',      'Avg Daily Trades per Account', ''),
    ('avg_size_usd',     'Avg Trade Size (USD)', '$'),
    ('long_short_ratio', 'Avg Long/Short Ratio', 'x'),
]
bins_data = daily[daily['binary'].isin(['Fear-zone','Greed-zone'])]
for ax, (col, title, prefix) in zip(axes, metrics):
    grp = bins_data.groupby('binary')[col].mean()
    bars = ax.bar(grp.index, grp.values,
                  color=[SIMPLE[k] for k in grp.index], edgecolor='white')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(f'{prefix}' if prefix else '')
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+bar.get_height()*0.01,
                f'{prefix}{v:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/charts/chart2_behavior_fear_vs_greed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 2")

# ── Chart 3: Segment PnL – High/Low Leverage ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 3 – PnL by Trader Segments × Sentiment', fontsize=14, fontweight='bold')

seg_pairs = [
    ('lev_seg',  'lev_seg',  ['High-Leverage','Low-Leverage'],  '#d62728,#2ca02c'),
    ('freq_seg', 'freq_seg', ['Frequent','Infrequent'],         '#ff7f0e,#1f77b4'),
    ('perf_seg', 'perf_seg', ['Net Winners','Net Losers'],      '#2ca02c,#d62728'),
]

sub = daily[daily['binary'].isin(['Fear-zone','Greed-zone'])]
for ax, (seg_col, seg_col2, seg_cats, clrs) in zip(axes, seg_pairs):
    clr_list = clrs.split(',')
    pivot = (sub.groupby([seg_col,'binary'])['total_pnl']
               .mean().unstack('binary')
               .reindex(seg_cats))
    x = np.arange(len(seg_cats))
    w = 0.35
    b1 = ax.bar(x - w/2, pivot.get('Fear-zone', 0), w,
                label='Fear', color='#d62728', alpha=0.85)
    b2 = ax.bar(x + w/2, pivot.get('Greed-zone', 0), w,
                label='Greed', color='#2ca02c', alpha=0.85)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(seg_cats, fontsize=9)
    ax.set_title(f'By {seg_col.replace("_seg","")}', fontsize=10)
    ax.set_ylabel('Mean Daily PnL (USD)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/charts/chart3_segments_pnl.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 3")

# ── Chart 4: Cross-margin usage by sentiment ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 4 – Risk Proxies by Sentiment', fontsize=14, fontweight='bold')

cross_by_sent = daily.groupby('classification')['cross_rate'].mean().reindex(ORDER)
axes[0].bar(ORDER, cross_by_sent.values*100,
            color=colors, edgecolor='white')
axes[0].set_title('Cross-Margin Usage Rate (%)')
axes[0].set_ylabel('%')
axes[0].set_xticklabels(ORDER, rotation=15, ha='right')
for i,v in enumerate(cross_by_sent.values):
    axes[0].text(i, v*100+0.3, f'{v*100:.1f}%', ha='center', fontsize=8)

vol_by_sent = daily.groupby('classification')['total_volume'].mean().reindex(ORDER)
axes[1].bar(ORDER, vol_by_sent.values/1000,
            color=colors, edgecolor='white')
axes[1].set_title('Avg Daily Volume per Account ($K)')
axes[1].set_ylabel('$K')
axes[1].set_xticklabels(ORDER, rotation=15, ha='right')

plt.tight_layout()
plt.savefig('/home/claude/charts/chart4_risk_proxies.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 4")

# ── Chart 5: Time-series overlay ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
fig.suptitle('Chart 5 – Aggregate Daily PnL vs Fear/Greed Index', fontsize=14, fontweight='bold')

# aggregate PnL per day
agg_pnl = daily.groupby('date')['total_pnl'].sum().reset_index()
agg_pnl['date'] = pd.to_datetime(agg_pnl['date'])
fg2 = fg.copy(); fg2['date'] = pd.to_datetime(fg2['date'])
fg2 = fg2.sort_values('date')

ax1.bar(pd.to_datetime(agg_pnl['date']), agg_pnl['total_pnl'],
        color=['#d62728' if v < 0 else '#2ca02c' for v in agg_pnl['total_pnl']],
        width=1, alpha=0.7)
ax1.axhline(0, color='black', lw=0.7)
ax1.set_ylabel('Total Daily PnL (USD)')
ax1.set_title('Total Cohort PnL')

ax2.fill_between(fg2['date'], fg2['value'], alpha=0.4,
                 color='orange', label='FG Index')
ax2.plot(fg2['date'], fg2['value'], color='darkorange', lw=0.8)
ax2.axhline(50, color='gray', lw=0.8, ls='--', label='Neutral (50)')
ax2.set_ylabel('Fear/Greed Value')
ax2.set_xlabel('Date')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/charts/chart5_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 5")

# ── Chart 6: Winner vs Loser behavior heatmap ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 6 – Net Winners vs Losers: Behavior Profile', fontsize=14, fontweight='bold')

metrics_map = {
    'Avg Trade Count':       'trade_count',
    'Avg Size USD ($K)':     'avg_size_usd',
    'Avg Win Rate (%)':      'win_rate',
    'Avg L/S Ratio':         'long_short_ratio',
    'Cross Margin Rate (%)': 'cross_rate',
}
scale_map = {'avg_size_usd': 1/1000, 'win_rate': 100, 'cross_rate': 100}

heat_data = {}
for seg in ['Net Winners', 'Net Losers']:
    sub2 = daily[daily['perf_seg'] == seg]
    row = {}
    for label, col in metrics_map.items():
        val = sub2[col].mean()
        sc = scale_map.get(col, 1)
        row[label] = val * sc
    heat_data[seg] = row

heat_df = pd.DataFrame(heat_data).T
# normalise per column for display
heat_norm = heat_df.apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9))

sns.heatmap(heat_norm, annot=heat_df.round(2), fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalised score'})
ax.set_title('Raw values annotated | Color = relative strength', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/charts/chart6_winner_loser_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart 6")

# ─────────────────────────────────────────
# 7. PRINT KEY STATS FOR WRITE-UP
# ─────────────────────────────────────────
print("\n" + "="*60)
print("KEY NUMBERS FOR WRITE-UP")
print("="*60)
print(f"Total trade rows: {df.shape[0]:,}")
print(f"Unique accounts:  {df['Account'].nunique()}")
print(f"Date range:       {merged['date'].min()} → {merged['date'].max()}")
print()
print("Mean daily PnL per account:")
print(daily.groupby('classification')['total_pnl'].mean().reindex(ORDER).round(2).to_string())
print()
print("Mean win-rate by sentiment:")
print((daily.groupby('classification')['win_rate'].mean().reindex(ORDER)*100).round(2).to_string())
print()
print("Behavior (Fear vs Greed):")
print(beh[['binary','avg_trades','avg_size_usd','avg_ls_ratio','avg_cross','avg_win_rate','avg_pnl']].to_string())
print()
print("Segment PnL (all sentiment):")
for col, cats in [('lev_seg',['High-Leverage','Low-Leverage']),
                  ('freq_seg',['Frequent','Infrequent']),
                  ('perf_seg',['Net Winners','Net Losers'])]:
    g = daily.groupby(col)['total_pnl'].mean().reindex(cats)
    print(f"  {col}: {g.round(2).to_dict()}")
print()
print(f"T-test PnL (Fear vs Greed): t={t:.3f}, p={p:.4f}")
print("="*60)
print("DONE")
