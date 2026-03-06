# Trader Performance vs Market Sentiment — Primetrade.ai Assignment

**Author:** Data Science Intern Candidate  
**Date:** March 2026

---

## Setup & How to Run

### Requirements
```
pandas numpy matplotlib seaborn scipy
```
All are standard packages available in any Python 3.10+ environment.

### Steps
```bash
# 1. Clone / unzip the repo
cd trader-sentiment-analysis

# 2. Place datasets in the data/ folder:
#    data/fear_greed_index.csv
#    data/historical_data_csv.gz

# 3a. Run the standalone script (fastest):
python analysis.py

# 3b. Or open the notebook:
jupyter notebook trader_sentiment_analysis.ipynb
```

Charts are saved to `charts/`.

---

## Methodology

### Data Preparation
- **Fear/Greed Index**: 2,644 daily records (2018–2025). Zero missing values, zero duplicates.
- **Hyperliquid Trader Data**: 211,224 trade-level rows across 32 accounts and 246 coins (May 2023 – May 2025). Zero missing values, zero duplicates.
- Timestamps in the trader dataset used `dayfirst=True` parsing (DD-MM-YYYY HH:MM format).
- Inner join on `date` yielded 211,218 matched rows (99.997% retention).
- Sentiment mapped to a binary `Fear-zone` / `Greed-zone` / `Neutral` label in addition to the 5-class original.

### Feature Engineering (Daily Account Level)
| Feature | Definition |
|---------|-----------|
| `total_pnl` | Sum of `Closed PnL` per account per day |
| `win_rate` | Closing trades with PnL > 0 ÷ total closing trades |
| `trade_count` | Number of trade rows |
| `avg_size_usd` | Mean `Size USD` |
| `long_short_ratio` | Long opens ÷ Short opens |
| `cross_rate` | Fraction of trades using cross-margin (proxy for leverage risk) |
| `net_pnl_after_fee` | `total_pnl` − `total_fee` |

### Trader Segments
| Segment | Criterion |
|---------|-----------|
| High vs Low Leverage | Cross-margin rate ≥ median |
| Frequent vs Infrequent | Total trade count ≥ median |
| Net Winners vs Net Losers | Cumulative PnL > 0 |

---

## Key Insights

### Insight 1 — Fear Days Produce Higher Average PnL, But More Variance
Mean daily PnL on Fear days = **$5,329** vs $3,318 on Greed days.  
However a t-test (t=0.75, p=0.45) shows no statistically significant difference — variance is huge.  
*Median* PnL tells the opposite story: Fear=$108, Greed=$158, meaning Greed days produce more **consistent** mid-range gains.

### Insight 2 — Traders Trade More Aggressively During Fear
On Fear-zone days vs Greed-zone days:
- **+37% more trades** per account per day (105 vs 77)
- **+43% larger average position size** ($8,530 vs $5,955)
- Long/Short ratio stays elevated (> 2×), suggesting traders still lean long even in fear

### Insight 3 — Low Leverage Traders Dramatically Outperform
Low-Leverage traders earn **$6,765/day** vs $3,205/day for High-Leverage traders — a **2.1× edge**.  
Net Winners post $4,799/day; Net Losers lose $1,832/day.  
The biggest behavioral differentiator: Net Winners use *smaller* average trade sizes and have notably higher win rates (~57% vs ~46%).

---

## Strategy Recommendations

### Strategy 1 — "Fear Contrarian" (for Frequent, Low-Leverage, Net-Winner traders)
> During **Extreme Fear / Fear** days:
> - ✅ Increase trade frequency (market dislocations create more edge)
> - ✅ Position size up selectively (data supports larger sizes on Fear days)
> - ❌ Do NOT use cross-margin; stick to isolated margin
> - ❌ Avoid high-leverage setups entirely

### Strategy 2 — "Greed Consolidation" (for all traders)
> During **Greed / Extreme Greed** days:
> - ✅ Execute fewer, higher-conviction trades
> - ✅ Take profits earlier (median gains are better but mean is dragged by blow-ups)
> - ❌ Net Losers and High-Leverage traders should reduce or eliminate positions
> - ❌ Avoid cross-margin on Greed days (cross usage is highest then — wrong time to be aggressive)

### Quick Reference Table
| Segment | Fear Day | Greed Day |
|---------|----------|-----------|
| Low-Leverage + Frequent + Winner | ↑ frequency + ↑ size | Fewer, targeted trades |
| High-Leverage (any) | Reduce cross exposure | Avoid new positions |
| Net Losers | Stay flat | Do not trade |
| Infrequent | 1–2 selective entries only | Normal pace |

---

## Output Files
```
charts/
  chart1_pnl_by_sentiment.png       — PnL & win-rate by all 5 sentiment classes
  chart2_behavior_fear_vs_greed.png — Trade count, size, L/S ratio (Fear vs Greed)
  chart3_segments_pnl.png           — PnL by 3 segments × sentiment
  chart4_risk_proxies.png           — Cross-margin usage & volume by sentiment
  chart5_timeseries.png             — Cohort PnL overlaid with Fear/Greed index
  chart6_winner_loser_heatmap.png   — Behavior profile: Winners vs Losers
trader_sentiment_analysis.ipynb    — Full annotated notebook
analysis.py                        — Standalone Python script
README.md                          — This file
```
