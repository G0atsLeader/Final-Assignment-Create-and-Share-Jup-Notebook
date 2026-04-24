import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ── Load GDELT daily CSVs ─────────────────────────────────────────────────────
neg_counts                  = pd.read_csv('Articles_count_Negative_keywords_daily.csv', sep=';')
tone                        = pd.read_csv('Average_tone_Neutral_keywords_daily.csv', sep=';')
neu_counts                  = pd.read_csv('Articles_count_daily_Neutral_keywords_daily.csv', sep=';')
tone_ALL_neutral_keywords   = pd.read_csv('Average_tone_ALL_Neutral_keywords_daily.csv', sep=';')
neu_counts_ALL_neutral_keywords = pd.read_csv('Articles_count_daily_ALL_Neutral_keywords_daily.csv', sep=';')

for df in [neg_counts, tone, neu_counts, tone_ALL_neutral_keywords, neu_counts_ALL_neutral_keywords]:
    date_col = df.columns[0]
    df['date'] = pd.to_datetime(df[date_col], dayfirst=True)
    df.set_index('date', inplace=True)
    if date_col != 'date':
        df.drop(columns=[date_col], inplace=True)

neg_col      = neg_counts.columns[0]
tone_col     = tone.columns[0]
neu_col      = neu_counts.columns[0]
tone_all_col = tone_ALL_neutral_keywords.columns[0]
neu_all_col  = neu_counts_ALL_neutral_keywords.columns[0]
print(f"Columns: neg={neg_col!r}  tone={tone_col!r}  neu={neu_col!r}")
print(f"         tone_ALL={tone_all_col!r}  neu_ALL={neu_all_col!r}")

# ── Weekly aggregation — Sun→Sat (week-START, matching GT and Denton) ─────────
# resample('W-SAT') bins Sun→Sat, labels by Saturday; shift -6 days → Sunday
weekly_neg_counts = neg_counts[neg_col].resample('W-SAT').sum()
weekly_neg_counts.index = weekly_neg_counts.index - pd.Timedelta(days=6)

def weighted_weekly_tone(tone_df, tone_c, weight_df, weight_c):
    combined = pd.DataFrame({'tone': tone_df[tone_c], 'weight': weight_df[weight_c]}).dropna()
    combined['weighted_tone'] = combined['tone'] * combined['weight']
    ww = combined.resample('W-SAT').sum()
    ww.index = ww.index - pd.Timedelta(days=6)
    return ww['weighted_tone'] / ww['weight']

weekly_avg_tone     = weighted_weekly_tone(tone,                      tone_col,     neu_counts,                     neu_col)
weekly_avg_tone_ALL = weighted_weekly_tone(tone_ALL_neutral_keywords, tone_all_col, neu_counts_ALL_neutral_keywords, neu_all_col)

# ── Daily series ──────────────────────────────────────────────────────────────
daily_neg      = neg_counts[neg_col]
daily_tone     = tone[tone_col]
daily_tone_ALL = tone_ALL_neutral_keywords[tone_all_col]

# ── Ensure GT indices are Timestamps ─────────────────────────────────────────
naive_denton.index = pd.to_datetime(naive_denton.index)
weekly_final.index = pd.to_datetime(weekly_final.index)

# ── Align daily series to full date range ────────────────────────────────────
full_daily_range = pd.date_range(
    min(naive_denton.index.min(), daily_neg.index.min(),
        daily_tone.index.min(), daily_tone_ALL.index.min()),
    max(naive_denton.index.max(), daily_neg.index.max(),
        daily_tone.index.max(), daily_tone_ALL.index.max()),
    freq='D'
)
daily_all = pd.DataFrame({
    'gtrends':           naive_denton.reindex(full_daily_range),
    'neg_art_count':     daily_neg.reindex(full_daily_range),
    'avg_tone':          daily_tone.reindex(full_daily_range),
    'avg_tone_ALL':      daily_tone_ALL.reindex(full_daily_range),
}, index=full_daily_range)
daily_all.index.name = 'date'

# ── Align weekly series to full Sunday range ──────────────────────────────────
full_weekly_range = pd.date_range(
    min(weekly_final.index.min(), weekly_neg_counts.index.min(),
        weekly_avg_tone.index.min(), weekly_avg_tone_ALL.index.min()),
    max(weekly_final.index.max(), weekly_neg_counts.index.max(),
        weekly_avg_tone.index.max(), weekly_avg_tone_ALL.index.max()),
    freq='W-SUN'
)
weekly_all = pd.DataFrame({
    'gtrends':           weekly_final.reindex(full_weekly_range),
    'neg_art_count':     weekly_neg_counts.reindex(full_weekly_range),
    'avg_tone':          weekly_avg_tone.reindex(full_weekly_range),
    'avg_tone_ALL':      weekly_avg_tone_ALL.reindex(full_weekly_range),
}, index=full_weekly_range)
weekly_all.index.name = 'week_start'

# ── Gap report ────────────────────────────────────────────────────────────────
print("=== DAILY SERIES ===")
print(f"Full range: {daily_all.index.min().date()} → {daily_all.index.max().date()} ({len(daily_all)} days)")
for col in daily_all.columns:
    n_miss = daily_all[col].isna().sum()
    if n_miss > 0:
        miss_dates = daily_all.index[daily_all[col].isna()]
        print(f"  {col}: {n_miss} missing  ({miss_dates.min().date()} → {miss_dates.max().date()})")
    else:
        print(f"  {col}: no missing")

print(f"\n=== WEEKLY SERIES ===")
print(f"Full range: {weekly_all.index.min().date()} → {weekly_all.index.max().date()} ({len(weekly_all)} weeks)")
for col in weekly_all.columns:
    n_miss = weekly_all[col].isna().sum()
    if n_miss > 0:
        miss_dates = weekly_all.index[weekly_all[col].isna()]
        print(f"  {col}: {n_miss} missing  ({miss_dates.min().date()} → {miss_dates.max().date()})")
    else:
        print(f"  {col}: no missing")

# ── ADF stationarity tests ────────────────────────────────────────────────────
def run_adf(series, name):
    s = series.dropna()
    result = adfuller(s, autolag='AIC')
    stat_str = "STATIONARY" if result[1] < 0.05 else "NON-STATIONARY"
    print(f"  {name:<50s} ADF={result[0]:>8.3f}  p={result[1]:.4f}  → {stat_str}")

print("\n" + "="*80)
print("ADF STATIONARITY TESTS")
print("="*80)

print("\n── Weekly (Levels) ──────────────────────────────────────────────────────────")
run_adf(weekly_all['gtrends'],       "Google Trends")
run_adf(weekly_all['neg_art_count'], "Neg article count")
run_adf(weekly_all['avg_tone'],      "Avg tone (weighted, Neutral kw)")
run_adf(weekly_all['avg_tone_ALL'],  "Avg tone (weighted, ALL Neutral kw)")

print("\n── Weekly (First Differences) ───────────────────────────────────────────────")
run_adf(weekly_all['gtrends'].diff(),       "Google Trends (Δ)")
run_adf(weekly_all['neg_art_count'].diff(), "Neg article count (Δ)")
run_adf(weekly_all['avg_tone'].diff(),      "Avg tone (weighted, Neutral kw) (Δ)")
run_adf(weekly_all['avg_tone_ALL'].diff(),  "Avg tone (weighted, ALL Neutral kw) (Δ)")

print("\n── Daily (Levels) ───────────────────────────────────────────────────────────")
run_adf(daily_all['gtrends'],       "Google Trends (Denton)")
run_adf(daily_all['neg_art_count'], "Neg article count")
run_adf(daily_all['avg_tone'],      "Avg tone (Neutral kw)")
run_adf(daily_all['avg_tone_ALL'],  "Avg tone (ALL Neutral kw)")

print("\n── Daily (First Differences) ────────────────────────────────────────────────")
run_adf(daily_all['gtrends'].diff(),       "Google Trends (Denton) (Δ)")
run_adf(daily_all['neg_art_count'].diff(), "Neg article count (Δ)")
run_adf(daily_all['avg_tone'].diff(),      "Avg tone (Neutral kw) (Δ)")
run_adf(daily_all['avg_tone_ALL'].diff(),  "Avg tone (ALL Neutral kw) (Δ)")

print("\n── DataFrames ready ─────────────────────────────────────────────────────────")
print(f"  daily_all:  {daily_all.shape}  (use .dropna() for pairwise analysis)")
print(f"  weekly_all: {weekly_all.shape}  (use .dropna() for pairwise analysis)")
