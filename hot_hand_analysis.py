"""
Hot Hand Fallacy — Baseline Analysis & Cognitive Engines
CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS

Runs Tasks 7–12 of the research plan:
  - Task 7–8:  Data cleaning and baseline statistical analysis
  - Task 11:   Rational Engine (Bayesian) — Type 2 System
  - Task 12:   Cognitive Engine (Exponential Decay) — Type 1 System

Output: luka_2025_26_shots_with_beliefs.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

from rational_engine import RationalEngine
from cognitive_engine import CognitiveEngine

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 12

SHOT_DATA_PATH = 'luka_2025_26_shots.csv'
OUTPUT_PATH    = 'luka_2025_26_shots_with_beliefs.csv'

STRENGTH_LEVELS = {
    'low_bias'    : 0.3,
    'medium_bias' : 0.6,
    'high_bias'   : 0.9,
}

# ── 1. Load & Clean Data (Tasks 7–8) ──────────────────────────────────────

df = pd.read_csv(SHOT_DATA_PATH)
print(f'Raw shape: {df.shape}')

print('\n=== Missing Values ===')
print(df.isnull().sum()[df.isnull().sum() > 0])
print('\n=== shot_made distribution ===')
print(df['shot_made'].value_counts())
print('\n=== shot_type distribution ===')
print(df['shot_type'].value_counts())
print('\n=== Shots per game (first 5 games) ===')
print(df.groupby('game_date')['shot_made'].count().head())

before = len(df)
df = df.dropna(subset=['shot_made', 'loc_x', 'loc_y', 'game_id'])
print(f'\nDropped {before - len(df)} rows with null required fields')

df['shot_made'] = df['shot_made'].astype(int)
df['game_id']   = df['game_id'].astype(str)
df['game_date'] = pd.to_datetime(df['game_date'])

df = df.sort_values(['game_date', 'game_id', 'game_shot_number']).reset_index(drop=True)
df['shot_number'] = df.index + 1

print(f'\nFinal clean shape: {df.shape}')
print(f'Date range: {df["game_date"].min().date()} → {df["game_date"].max().date()}')
print(f'Games: {df["game_id"].nunique()}')
print(f'Total FGAs: {len(df)}')

# ── 2. Baseline Statistical Analysis (Task 8) ─────────────────────────────

total_fga  = len(df)
total_fgm  = df['shot_made'].sum()
overall_fg = total_fgm / total_fga

ci_low, ci_high = proportion_confint(total_fgm, total_fga, alpha=0.05, method='wilson')

print('\n' + '=' * 50)
print('  Luka Doncic — 2025-26 Regular Season')
print('=' * 50)
print(f'  Total FGA : {total_fga}')
print(f'  Total FGM : {total_fgm}')
print(f'  FG%       : {overall_fg:.4f}  ({overall_fg*100:.1f}%)')
print(f'  95% CI    : [{ci_low:.4f}, {ci_high:.4f}]')
print(f'\n  → Rational Prior for Bayesian Engine: α={total_fgm}, β={total_fga - total_fgm}')

by_type = df.groupby('shot_type')['shot_made'].agg(['sum', 'count', 'mean'])
by_type.columns = ['FGM', 'FGA', 'FG%']
by_type['FG%'] = by_type['FG%'].map('{:.1%}'.format)
print('\nFG% by Shot Type:')
print(by_type.to_string())

by_zone = df.groupby('shot_zone_basic')['shot_made'].agg(['sum', 'count', 'mean'])
by_zone.columns = ['FGM', 'FGA', 'FG%']
by_zone = by_zone.sort_values('FGA', ascending=False)
by_zone['FG%'] = by_zone['FG%'].map('{:.1%}'.format)
print('\nFG% by Zone:')
print(by_zone.to_string())

# Streak distribution plot
streak_counts = df['streak'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#d7191c' if x < 0 else '#2b83ba' if x > 0 else '#aaaaaa'
          for x in streak_counts.index]
axes[0].bar(streak_counts.index, streak_counts.values, color=colors, edgecolor='white')
axes[0].set_xlabel('Streak entering shot (+ = make streak, - = miss streak)')
axes[0].set_ylabel('Number of shots')
axes[0].set_title('Streak Distribution — All FGAs')
axes[0].axvline(0, color='black', linewidth=1, linestyle='--')

streak_fg = df.groupby('streak')['shot_made'].mean()
streak_fg_count = df.groupby('streak')['shot_made'].count()
streak_fg = streak_fg[streak_fg_count >= 15]

axes[1].bar(streak_fg.index, streak_fg.values * 100,
            color=['#d7191c' if x < 0 else '#2b83ba' if x > 0 else '#aaaaaa'
                   for x in streak_fg.index],
            edgecolor='white')
axes[1].axhline(overall_fg * 100, color='black', linewidth=2,
                linestyle='--', label=f'Season avg ({overall_fg*100:.1f}%)')
axes[1].set_xlabel('Streak entering shot')
axes[1].set_ylabel('FG% on this shot (%)')
axes[1].set_title('FG% Conditional on Prior Streak\n(Hot Hand test — rational view)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].legend()

plt.suptitle("Luka Doncic 2025-26 — Streak Analysis", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print('\nNote: if FG% after a hot streak is NOT significantly higher than baseline,')
print('that supports the Hot Hand being a fallacy (Gilovich et al., 1985)')

# ── 3. Run the Rational Engine — Type 2 System (Task 11) ──────────────────

rational = RationalEngine(
    prior_makes  = total_fgm,
    prior_misses = total_fga - total_fgm
)

rational_beliefs_before = []
rational_beliefs_after  = []
current_game = None

for _, row in df.iterrows():
    if row['game_id'] != current_game:
        rational.new_game()
        current_game = row['game_id']

    rational_beliefs_before.append(rational.current_belief)
    rational.update(int(row['shot_made']), metadata={'shot_number': row['shot_number']})
    rational_beliefs_after.append(rational.current_belief)

df['rational_belief_before'] = rational_beliefs_before
df['rational_belief_after']  = rational_beliefs_after

print('\nRational Engine complete.')
print(f"Belief range: [{df['rational_belief_before'].min():.4f}, {df['rational_belief_before'].max():.4f}]")
print(f"Mean belief : {df['rational_belief_before'].mean():.4f}  (should be ≈ season FG% = {overall_fg:.4f})")

# ── 4. Run the Cognitive Engine — Type 1 System (Task 12) ─────────────────

for label, strength in STRENGTH_LEVELS.items():
    engine = CognitiveEngine(
        prior_fg_pct       = overall_fg,
        heuristic_strength = strength,
        prior_anchor       = 0.3
    )

    beliefs_before = []
    current_game = None

    for _, row in df.iterrows():
        if row['game_id'] != current_game:
            engine.new_game()
            current_game = row['game_id']

        beliefs_before.append(engine.current_belief)
        engine.update(int(row['shot_made']))

    col_name = f'cognitive_belief_{label}'
    df[col_name] = beliefs_before

    print(f'{label} (strength={strength}): '
          f'mean={df[col_name].mean():.4f}, '
          f'std={df[col_name].std():.4f}, '
          f'range=[{df[col_name].min():.3f}, {df[col_name].max():.3f}]')

# ── Divergence metric ─────────────────────────────────────────────────────

for label in STRENGTH_LEVELS:
    col = f'cognitive_belief_{label}'
    df[f'divergence_{label}'] = (df[col] - df['rational_belief_before']).abs()

print('\nDivergence metrics added.')
print('\nMean divergence by heuristic strength:')
for label, strength in STRENGTH_LEVELS.items():
    mean_div = df[f'divergence_{label}'].mean()
    print(f'  {label} (strength={strength}): {mean_div:.4f}  ({mean_div*100:.2f} percentage points avg drift)')

# ── 5. Belief Comparison Visualization ────────────────────────────────────

game_max_streak = df.groupby('game_id')['streak'].max().idxmax()
game_df = df[df['game_id'] == game_max_streak].copy()
game_date_str = game_df['game_date'].iloc[0].strftime('%b %d, %Y')

print(f'\nShowing belief trace for game: {game_date_str} (highest make streak in season)')
print(f'Shots in game: {len(game_df)}')
print(f'Max streak in game: +{game_df["streak"].max()}')

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

shot_nums = game_df['game_shot_number']

axes[0].plot(shot_nums, game_df['rational_belief_before'] * 100,
             color='#2166ac', linewidth=2.5, label='Rational (Bayesian)', zorder=3)
axes[0].plot(shot_nums, game_df['cognitive_belief_high_bias'] * 100,
             color='#d7191c', linewidth=2.5, linestyle='--',
             label='Cognitive (High Bias, strength=0.9)', zorder=3)
axes[0].plot(shot_nums, game_df['cognitive_belief_medium_bias'] * 100,
             color='#f4a442', linewidth=1.5, linestyle=':',
             label='Cognitive (Medium Bias, strength=0.6)', zorder=2)
axes[0].axhline(overall_fg * 100, color='gray', linewidth=1,
                linestyle='-.', label=f'Season FG% ({overall_fg*100:.1f}%)')

for _, row in game_df.iterrows():
    color = '#27ae60' if row['shot_made'] == 1 else '#c0392b'
    axes[0].axvline(row['game_shot_number'], color=color, alpha=0.15, linewidth=1)

axes[0].set_ylabel('Estimated FG Probability (%)')
axes[0].set_title(f'Rational vs. Biased Belief — {game_date_str}\n'
                  '(green lines = makes, red lines = misses)')
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].legend(loc='upper right', fontsize=10)
axes[0].set_ylim(20, 80)

axes[1].fill_between(shot_nums, game_df['divergence_high_bias'] * 100,
                     alpha=0.6, color='#d7191c', label='High Bias divergence')
axes[1].fill_between(shot_nums, game_df['divergence_medium_bias'] * 100,
                     alpha=0.4, color='#f4a442', label='Medium Bias divergence')
axes[1].set_xlabel('Shot number within game')
axes[1].set_ylabel('|Bias Gap| (percentage points)')
axes[1].set_title('Divergence from Rational Baseline (Bias Gap)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# Season-level divergence by streak magnitude
streak_div = df.groupby('streak').agg(
    high_div   = ('divergence_high_bias',   'mean'),
    medium_div = ('divergence_medium_bias', 'mean'),
    low_div    = ('divergence_low_bias',    'mean'),
    count      = ('shot_made',              'count')
).query('count >= 15')

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(streak_div.index, streak_div['high_div']   * 100, 'o-', color='#d7191c',
        linewidth=2.5, markersize=7, label='High Bias (strength=0.9)')
ax.plot(streak_div.index, streak_div['medium_div'] * 100, 's-', color='#f4a442',
        linewidth=2.5, markersize=7, label='Medium Bias (strength=0.6)')
ax.plot(streak_div.index, streak_div['low_div']    * 100, '^-', color='#74add1',
        linewidth=2.5, markersize=7, label='Low Bias (strength=0.3)')

ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(3, color='purple', linewidth=1.5, linestyle=':', alpha=0.7,
           label='Hypothesized tipping point (+3 streak)')

ax.set_xlabel('Streak entering shot (+ = consecutive makes, - = consecutive misses)')
ax.set_ylabel('Mean |Bias Gap| (percentage points)')
ax.set_title('Divergence from Rational Baseline by Streak Magnitude\n'
             'Does the cognitive bias amplify with longer streaks?')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()

plt.tight_layout()
plt.show()

print('\nIf divergence increases monotonically as streak magnitude grows,')
print('that validates the Hot Hand cognitive model.')

# ── 6. Save Enriched CSV ──────────────────────────────────────────────────

df.to_csv(OUTPUT_PATH, index=False)

print(f'\nSaved: {OUTPUT_PATH}')
print(f'Shape: {df.shape}')
print('\nNew columns added:')
new_cols = [c for c in df.columns if any(x in c for x in
            ['rational', 'cognitive', 'divergence'])]
for col in new_cols:
    print(f'  {col}')
