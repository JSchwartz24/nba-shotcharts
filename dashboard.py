"""
dashboard.py
────────────
Hot Hand Fallacy — Interactive Dashboard
CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS

Run with:
    streamlit run dashboard.py

Requires:
    luka_2025_26_shots.csv  (from luka_hot_hand_data.py)
    rational_engine.py
    cognitive_engine.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, Arc
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from rational_engine import RationalEngine
from cognitive_engine import CognitiveEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hot Hand Fallacy — Luka Dončić",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --bg:        #0d0f14;
    --surface:   #161922;
    --border:    #252a36;
    --accent:    #00c2ff;
    --accent2:   #ff4b4b;
    --accent3:   #ffd166;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --rational:  #3b9eff;
    --cognitive: #ff5a5a;
    --make:      #22c55e;
    --miss:      #ef4444;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-val  { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; color: var(--accent); line-height: 1; }
  .metric-val.red   { color: var(--accent2); }
  .metric-val.green { color: var(--make); }
  .metric-val.gold  { color: var(--accent3); }
  .metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

  .section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 16px;
  }

  .theory-box {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #aab0be;
    margin-bottom: 16px;
    line-height: 1.6;
  }

  .shot-dot-made  { display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--make);  margin-right:4px; }
  .shot-dot-miss  { display:inline-block; width:10px; height:10px; background:var(--miss); margin-right:4px; clip-path:polygon(0 0,100% 0,50% 50%,100% 100%,0 100%,50% 50%); }

  [data-testid="stSlider"] > div > div { accent-color: var(--accent); }

  code, .dm-mono { font-family: 'DM Mono', monospace; }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Tab styling */
  [data-baseweb="tab-list"] { gap: 8px; }
  [data-baseweb="tab"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
  }
  [aria-selected="true"][data-baseweb="tab"] {
    background: var(--accent) !important;
    color: #000 !important;
    border-color: var(--accent) !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d0f14',
    'axes.facecolor':    '#161922',
    'axes.edgecolor':    '#252a36',
    'axes.labelcolor':   '#e8eaf0',
    'text.color':        '#e8eaf0',
    'xtick.color':       '#6b7280',
    'ytick.color':       '#6b7280',
    'grid.color':        '#252a36',
    'grid.linewidth':    0.6,
    'font.family':       'sans-serif',
    'legend.facecolor':  '#161922',
    'legend.edgecolor':  '#252a36',
})

RATIONAL_COLOR  = '#3b9eff'
COGNITIVE_COLOR = '#ff5a5a'
MAKE_COLOR      = '#22c55e'
MISS_COLOR      = '#ef4444'
GOLD_COLOR      = '#ffd166'


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & ENGINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['shot_made'] = df['shot_made'].astype(int)
    df['game_id']   = df['game_id'].astype(str)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_id', 'game_shot_number']).reset_index(drop=True)
    df['shot_number'] = df.index + 1
    return df


@st.cache_data
def run_engines(df_json: str, heuristic_strength: float, prior_anchor: float):
    """Run both engines across the full dataset. Cached by parameters."""
    df = pd.read_json(df_json)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_id', 'game_shot_number']).reset_index(drop=True)

    total_fgm = int(df['shot_made'].sum())
    total_fga = len(df)
    overall_fg = total_fgm / total_fga

    rational  = RationalEngine(prior_makes=total_fgm, prior_misses=total_fga - total_fgm)
    cognitive = CognitiveEngine(
        prior_fg_pct=overall_fg,
        heuristic_strength=heuristic_strength,
        prior_anchor=prior_anchor
    )

    r_before, r_after, c_before, c_after = [], [], [], []
    current_game = None

    for _, row in df.iterrows():
        if row['game_id'] != current_game:
            rational.new_game()
            cognitive.new_game()
            current_game = row['game_id']

        r_before.append(rational.current_belief)
        cognitive_b = cognitive.current_belief
        c_before.append(cognitive_b)

        rational.update(int(row['shot_made']))
        cognitive.update(int(row['shot_made']))

        r_after.append(rational.current_belief)
        c_after.append(cognitive.current_belief)

    df['rational_belief']  = r_before
    df['cognitive_belief'] = c_before
    df['divergence']       = (np.array(c_before) - np.array(r_before))

    return df, overall_fg, total_fgm, total_fga


def draw_court(ax, color='#3a4257', lw=1.5):
    elements = [
        Circle((0, 0),    radius=7.5,  lw=lw, color=color, fill=False),
        Rectangle((-30, -12.5), 60, 0, lw=lw, color=color),
        Rectangle((-80, -47.5), 160, 190, lw=lw, color=color, fill=False),
        Rectangle((-60, -47.5), 120, 190, lw=lw, color=color, fill=False),
        Arc((0, 142.5), 120, 120, theta1=0,   theta2=180, lw=lw, color=color, fill=False),
        Arc((0, 142.5), 120, 120, theta1=180, theta2=0,   lw=lw, color=color, linestyle='dashed'),
        Arc((0, 0),      80,  80, theta1=0,   theta2=180, lw=lw, color=color),
        Rectangle((-220, -47.5), 0, 140, lw=lw, color=color),
        Rectangle(( 220, -47.5), 0, 140, lw=lw, color=color),
        Arc((0, 0), 475, 475, theta1=22, theta2=158, lw=lw, color=color),
        Arc((0, 422.5), 120, 120, theta1=180, theta2=0, lw=lw, color=color),
        Arc((0, 422.5),  40,  40, theta1=180, theta2=0, lw=lw, color=color),
        Rectangle((-250, -47.5), 500, 470, lw=lw, color=color, fill=False),
    ]
    for e in elements:
        ax.add_patch(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.8rem;color:#00c2ff;letter-spacing:0.1em;">🏀 HOT HAND<br><span style="font-size:1rem;color:#6b7280;">FALLACY SIMULATOR</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:20px;">CS6795 · Georgia Tech OMSCS</div>', unsafe_allow_html=True)
    st.divider()

    data_path = st.text_input("Shot data CSV path", value="luka_2025_26_shots.csv")

    st.markdown("### ⚙️ Cognitive Engine Parameters")
    heuristic_strength = st.slider(
        "Heuristic Strength",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="How aggressively the biased observer overweights recent shots. 0 = rational, 1 = only last shot matters."
    )
    prior_anchor = st.slider(
        "Prior Anchor Weight",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="How much the biased observer still respects the long-term average. 0.3 = 30% anchored to season FG%."
    )

    st.divider()
    decay_lambda = 1.0 - heuristic_strength
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#6b7280;line-height:1.8;">
    <b style="color:#e8eaf0;">Decay λ</b> = {decay_lambda:.2f}<br>
    <b style="color:#e8eaf0;">Weight of last shot</b> = {decay_lambda**0:.2f}<br>
    <b style="color:#e8eaf0;">2 shots ago</b> = {decay_lambda**1:.2f}<br>
    <b style="color:#e8eaf0;">5 shots ago</b> = {decay_lambda**4:.3f}<br>
    <b style="color:#e8eaf0;">10 shots ago</b> = {decay_lambda**9:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#6b7280;">Gilovich, Vallone & Tversky (1985)<br>Miller & Sanjurjo (2018, 2019)<br>Bocskocsky, Ezekowitz & Stein (2014)</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

try:
    raw_df = load_data(data_path)
except FileNotFoundError:
    st.error(f"Could not find `{data_path}`. Make sure it's in the same directory as this script.")
    st.stop()

df, overall_fg, total_fgm, total_fga = run_engines(
    raw_df.assign(game_date=raw_df['game_date'].dt.strftime('%Y-%m-%d')).to_json(),
    heuristic_strength, prior_anchor
)

ci_low, ci_high = proportion_confint(total_fgm, total_fga, alpha=0.05, method='wilson')
n_games = df['game_id'].nunique()
mean_div = df['divergence'].abs().mean()
max_div  = df['divergence'].abs().max()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#e8eaf0;letter-spacing:0.05em;">LUKA DONČIĆ</span>
  <span style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#6b7280;letter-spacing:0.08em;">2025–26 REGULAR SEASON</span>
</div>
<div style="font-size:0.85rem;color:#6b7280;margin-bottom:24px;">
  Hot Hand Fallacy Simulator — Rational Bayesian Observer vs. Heuristic-Driven Cognitive Model
</div>
""", unsafe_allow_html=True)

# ── Top metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (f"{total_fga:,}", "Total FGAs", ""),
    (f"{overall_fg*100:.1f}%", "Season FG%", "green"),
    (f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]", "95% Confidence Interval", ""),
    (f"{n_games}", "Games Played", "gold"),
    (f"{mean_div*100:.1f}pp", f"Mean Bias Gap (strength={heuristic_strength})", "red"),
]
for col, (val, label, cls) in zip([c1,c2,c3,c4,c5], metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val {cls}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Belief Trace", "🎯 Game Explorer", "📊 Streak Analysis",
    "🏀 Shot Chart", "🧠 Model Theory", "🧪 Heuristic Strength Test"
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — SEASON-WIDE BELIEF TRACE
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Season Belief Trace — Rational vs. Cognitive</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box">Each point on the x-axis is a single field goal attempt in chronological order. The <b style="color:#3b9eff;">blue line</b> shows what a rational Bayesian observer believes about Luka\'s FG probability before each shot. The <b style="color:#ff5a5a;">red line</b> shows the biased cognitive observer, whose belief swings with recent streaks. The shaded region is the Bias Gap.</div>', unsafe_allow_html=True)

    # Rolling window for readability
    window = st.slider("Smoothing window (shots)", 1, 30, 10, key="smooth")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1.2]})

    r_smooth = df['rational_belief'].rolling(window, min_periods=1).mean() * 100
    c_smooth = df['cognitive_belief'].rolling(window, min_periods=1).mean() * 100
    d_smooth = df['divergence'].rolling(window, min_periods=1).mean() * 100

    x = df['shot_number']

    axes[0].plot(x, r_smooth, color=RATIONAL_COLOR,  lw=2,   label='Rational (Bayesian)', zorder=3)
    axes[0].plot(x, c_smooth, color=COGNITIVE_COLOR, lw=2,   label=f'Cognitive (strength={heuristic_strength})', zorder=3)
    axes[0].fill_between(x, r_smooth, c_smooth,
                         where=(c_smooth >= r_smooth), alpha=0.25, color=COGNITIVE_COLOR,
                         label='Cognitive overestimates')
    axes[0].fill_between(x, r_smooth, c_smooth,
                         where=(c_smooth < r_smooth), alpha=0.25, color=RATIONAL_COLOR,
                         label='Cognitive underestimates')
    axes[0].axhline(overall_fg * 100, color=GOLD_COLOR, lw=1.2, ls='--', alpha=0.7,
                    label=f'True FG% ({overall_fg*100:.1f}%)')
    axes[0].set_ylabel('Estimated FG Probability (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.8)
    axes[0].set_title('Belief Over the Season', fontsize=13, pad=10)
    axes[0].grid(True, alpha=0.3)

    # Divergence panel
    pos_div = np.where(d_smooth >= 0, d_smooth, 0)
    neg_div = np.where(d_smooth < 0, d_smooth, 0)
    axes[1].fill_between(x, pos_div, alpha=0.7, color=COGNITIVE_COLOR, label='Over-confident (hot hand)')
    axes[1].fill_between(x, neg_div, alpha=0.7, color=RATIONAL_COLOR,  label='Under-confident (cold hand)')
    axes[1].axhline(0, color='#6b7280', lw=0.8)
    axes[1].set_ylabel('Bias Gap (pp)')
    axes[1].set_xlabel('Shot Number (Season)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.8)
    axes[1].set_title('Divergence from Rational Baseline', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — GAME EXPLORER
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Game Explorer — Shot-by-Shot Belief</div>', unsafe_allow_html=True)

    games = df.groupby('game_id').agg(
        date=('game_date', 'first'),
        shots=('shot_made', 'count'),
        fgm=('shot_made', 'sum'),
        max_streak=('streak', 'max'),
        mean_div=('divergence', lambda x: x.abs().mean())
    ).reset_index()
    games['fg_pct'] = games['fgm'] / games['shots']
    games['label'] = games.apply(
        lambda r: f"{r['date'].strftime('%b %d')} — {r['fgm']}/{r['shots']} FG ({r['fg_pct']*100:.0f}%)", axis=1
    )
    games = games.sort_values('date')

    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        selected_label = st.selectbox("Select a game", games['label'].tolist())
    selected_game_id = games[games['label'] == selected_label]['game_id'].iloc[0]
    gdf = df[df['game_id'] == selected_game_id].copy()
    g_info = games[games['game_id'] == selected_game_id].iloc[0]

    with col_info:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;margin-top:8px;">
          <div style="font-size:0.8rem;color:#6b7280;">MAX STREAK IN GAME</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#ffd166;">+{int(g_info['max_streak'])}</div>
          <div style="font-size:0.8rem;color:#6b7280;margin-top:4px;">MEAN BIAS GAP</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#ff5a5a;">{g_info['mean_div']*100:.1f}pp</div>
        </div>
        """, unsafe_allow_html=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                              gridspec_kw={'height_ratios': [2.5, 1.2, 0.7]})

    shot_x = gdf['game_shot_number']

    # ── Belief trace ──────────────────────────────────────────────────────
    axes[0].plot(shot_x, gdf['rational_belief']  * 100, color=RATIONAL_COLOR,  lw=2.5,
                 label='Rational (Bayesian)', zorder=3)
    axes[0].plot(shot_x, gdf['cognitive_belief'] * 100, color=COGNITIVE_COLOR, lw=2.5,
                 linestyle='--', label=f'Cognitive (strength={heuristic_strength})', zorder=3)
    axes[0].axhline(overall_fg * 100, color=GOLD_COLOR, lw=1.2, ls=':', alpha=0.8,
                    label=f'Season FG% ({overall_fg*100:.1f}%)')

    for _, row in gdf.iterrows():
        c = MAKE_COLOR if row['shot_made'] == 1 else MISS_COLOR
        axes[0].axvline(row['game_shot_number'], color=c, alpha=0.12, lw=1)

    axes[0].set_ylim(15, 85)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylabel('Estimated FG Probability (%)')
    axes[0].set_title('Shot-by-Shot Belief Within Game', fontsize=12)
    axes[0].legend(fontsize=9, framealpha=0.8)
    axes[0].grid(True, alpha=0.3)

    # ── Divergence ────────────────────────────────────────────────────────
    div = gdf['divergence'] * 100
    axes[1].fill_between(shot_x, div, where=(div >= 0), alpha=0.8, color=COGNITIVE_COLOR)
    axes[1].fill_between(shot_x, div, where=(div < 0),  alpha=0.8, color=RATIONAL_COLOR)
    axes[1].axhline(0, color='#6b7280', lw=0.8)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_ylabel('Bias Gap (pp)')
    axes[1].set_title('Divergence from Rational Baseline', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # ── Shot outcomes timeline ─────────────────────────────────────────────
    axes[2].set_xlim(shot_x.min() - 0.5, shot_x.max() + 0.5)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_xlabel('Shot Number Within Game')
    axes[2].set_title('Shot Outcomes (● make / ✕ miss)', fontsize=10)
    for _, row in gdf.iterrows():
        if row['shot_made'] == 1:
            axes[2].scatter(row['game_shot_number'], 0.5, color=MAKE_COLOR,
                            s=80, zorder=3, marker='o')
        else:
            axes[2].scatter(row['game_shot_number'], 0.5, color=MISS_COLOR,
                            s=80, zorder=3, marker='X')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — STREAK ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Streak Analysis — Tipping Point Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box">The core prediction: the Cognitive model\'s divergence from rational should grow as streak magnitude increases. A <b style="color:#ffd166;">"tipping point"</b> around +3 consecutive makes is hypothesized — the point at which the Hot Hand fallacy causes the biased model to become statistically unreliable.</div>', unsafe_allow_html=True)

    min_sample = st.slider("Minimum shots per streak value (filter)", 5, 50, 15)

    streak_agg = df.groupby('streak').agg(
        count=('shot_made', 'count'),
        actual_fg=('shot_made', 'mean'),
        rational_mean=('rational_belief', 'mean'),
        cognitive_mean=('cognitive_belief', 'mean'),
        mean_div=('divergence', lambda x: x.abs().mean()),
    ).query(f'count >= {min_sample}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── 1. FG% conditional on streak (the empirical Hot Hand test) ────────
    colors = [MISS_COLOR if s < 0 else (GOLD_COLOR if s == 0 else MAKE_COLOR)
              for s in streak_agg.index]
    axes[0].bar(streak_agg.index, streak_agg['actual_fg'] * 100,
                color=colors, alpha=0.8, edgecolor='#0d0f14', lw=0.5)
    axes[0].axhline(overall_fg * 100, color=RATIONAL_COLOR, lw=2, ls='--',
                    label=f'Season avg ({overall_fg*100:.1f}%)')
    axes[0].axvline(3,  color=GOLD_COLOR, lw=1.5, ls=':', alpha=0.8, label='Tipping point (+3)')
    axes[0].axvline(-3, color=GOLD_COLOR, lw=1.5, ls=':', alpha=0.8)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_xlabel('Streak entering shot')
    axes[0].set_ylabel('Actual FG%')
    axes[0].set_title('Empirical FG%\nby Prior Streak', fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ── 2. Rational vs Cognitive mean beliefs by streak ────────────────────
    axes[1].plot(streak_agg.index, streak_agg['rational_mean']  * 100, 'o-',
                 color=RATIONAL_COLOR,  lw=2.5, ms=7, label='Rational')
    axes[1].plot(streak_agg.index, streak_agg['cognitive_mean'] * 100, 's-',
                 color=COGNITIVE_COLOR, lw=2.5, ms=7, label='Cognitive')
    axes[1].axhline(overall_fg * 100, color=GOLD_COLOR, lw=1.2, ls='--', alpha=0.7)
    axes[1].axvline(3,  color=GOLD_COLOR, lw=1.5, ls=':', alpha=0.7)
    axes[1].axvline(-3, color=GOLD_COLOR, lw=1.5, ls=':', alpha=0.7)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_xlabel('Streak entering shot')
    axes[1].set_ylabel('Mean predicted FG%')
    axes[1].set_title('Model Beliefs\nby Prior Streak', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # ── 3. Mean divergence by streak (the key bias gap chart) ─────────────
    bar_colors = [RATIONAL_COLOR if s < 0 else COGNITIVE_COLOR for s in streak_agg.index]
    axes[2].bar(streak_agg.index, streak_agg['mean_div'] * 100,
                color=bar_colors, alpha=0.85, edgecolor='#0d0f14', lw=0.5)
    axes[2].axvline(3,  color=GOLD_COLOR, lw=2, ls=':', alpha=0.9, label='Tipping point (+3)')
    axes[2].axvline(-3, color=GOLD_COLOR, lw=2, ls=':', alpha=0.9)
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].set_xlabel('Streak entering shot')
    axes[2].set_ylabel('Mean |Bias Gap| (pp)')
    axes[2].set_title('Bias Gap Magnitude\nby Prior Streak', fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Streak Analysis — Heuristic Strength = {heuristic_strength}',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Streak data table ─────────────────────────────────────────────────
    with st.expander("View raw streak table"):
        display = streak_agg.copy()
        display.index.name = 'Streak'
        display['actual_fg']      = display['actual_fg'].map('{:.1%}'.format)
        display['rational_mean']  = display['rational_mean'].map('{:.1%}'.format)
        display['cognitive_mean'] = display['cognitive_mean'].map('{:.1%}'.format)
        display['mean_div']       = display['mean_div'].map('{:.3f}'.format)
        st.dataframe(display, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — SHOT CHART
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Shot Chart — 2025-26 Season</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        chart_filter = st.selectbox("Filter shots", ["All", "Made only", "Missed only", "Hot streaks (≥3)", "Cold streaks (≤-3)"])
    with col_b:
        show_heatmap = st.checkbox("Show density heatmap", value=False)

    plot_df = df.copy()
    if chart_filter == "Made only":
        plot_df = plot_df[plot_df['shot_made'] == 1]
    elif chart_filter == "Missed only":
        plot_df = plot_df[plot_df['shot_made'] == 0]
    elif chart_filter == "Hot streaks (≥3)":
        plot_df = plot_df[plot_df['streak'] >= 3]
    elif chart_filter == "Cold streaks (≤-3)":
        plot_df = plot_df[plot_df['streak'] <= -3]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-250, 250)
    ax.set_ylim(422.5, -47.5)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_court(ax)

    if show_heatmap and len(plot_df) > 10:
        try:
            import seaborn as sns
            sns.kdeplot(x=plot_df['loc_x'], y=plot_df['loc_y'],
                        fill=True, cmap='inferno', ax=ax, alpha=0.6, bw_adjust=0.6)
        except Exception:
            pass

    made  = plot_df[plot_df['shot_made'] == 1]
    missed = plot_df[plot_df['shot_made'] == 0]

    ax.scatter(missed['loc_x'], missed['loc_y'], c=MISS_COLOR,
               marker='x', s=25, linewidths=1.2, alpha=0.6, label=f'Missed ({len(missed)})')
    ax.scatter(made['loc_x'],   made['loc_y'],   facecolors='none',
               edgecolors=MAKE_COLOR, marker='o', s=25, linewidths=1.2, alpha=0.7,
               label=f'Made ({len(made)})')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    fg_pct = len(made) / len(plot_df) * 100 if len(plot_df) > 0 else 0
    ax.set_title(f'Luka Dončić — {chart_filter}    {len(made)}/{len(plot_df)} FG ({fg_pct:.1f}%)',
                 fontsize=13, pad=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — MODEL THEORY
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Model Theory & Cognitive Science Framing</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔵 Type 2 System — Rational Bayesian Observer")
        st.markdown("""
        <div class="theory-box">
        Uses a <b>Beta-Binomial conjugate model</b>. Each shot is treated as a Bernoulli trial
        with unknown probability θ. The prior is seeded with the player's full season FG%,
        and the posterior is updated after every shot:
        <br><br>
        <code>Prior:     θ ~ Beta(α₀, β₀)</code><br>
        <code>Posterior: θ ~ Beta(α₀ + makes, β₀ + misses)</code><br>
        <code>Belief:    E[θ] = α / (α + β)</code>
        <br><br>
        The rational observer's belief moves very slowly — anchored to hundreds of prior shots —
        and resets to the season prior at each new game. This represents <b>Bounded Rationality</b>:
        the ideal statistical observer who processes all available data.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔴 Type 1 System — Cognitive Bias Simulator")
        st.markdown("""
        <div class="theory-box">
        Uses an <b>exponential decay weighted average</b>. Recent shots receive disproportionate
        weight, simulating the <b>Representativeness Heuristic</b> and <b>Recency Bias</b>:
        <br><br>
        <code>weight(k) = λᵏ   where λ = 1 − strength</code><br>
        <code>recency_est = Σ(wₖ × outcomeₖ) / Σ(wₖ)</code><br>
        <code>belief = anchor × prior + (1−anchor) × recency</code>
        <br><br>
        At strength=0.9: a shot from 5 attempts ago carries only
        <b>{:.1f}%</b> of the weight of the most recent shot.
        This reflects <b>Dual Process Theory</b> — the fast, intuitive System 1
        that sacrifices statistical accuracy for cognitive efficiency.
        </div>
        """.format((1 - heuristic_strength)**4 * 100), unsafe_allow_html=True)

    st.markdown("#### 📐 Divergence Metric")
    st.markdown("""
    <div class="theory-box">
    The <b>Bias Gap</b> at each shot is defined as:<br><br>
    <code>divergence(t) = cognitive_belief(t) − rational_belief(t)</code><br><br>
    A positive divergence means the biased observer is <i>over-confident</i> (Hot Hand effect — believes
    the player is "on fire"). A negative divergence means the biased observer is <i>under-confident</i>
    (Cold Hand effect). The magnitude of divergence after streak sequences of length ≥3 is the
    primary empirical test of the cognitive model's validity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📚 Key References")
    refs = [
        ("Gilovich, Vallone & Tversky (1985)", "The hot hand in basketball: On the misperception of random sequences. *Cognit. Psychol.*, 17, 295–314."),
        ("Miller & Sanjurjo (2018)", "Surprised by the hot hand fallacy? A truth in the law of small numbers. Univ. of Alicante Working Paper."),
        ("Miller & Sanjurjo (2019)", "A cold shower for the hot hand fallacy: Robust evidence that belief in the hot hand is justified."),
        ("Bocskocsky, Ezekowitz & Stein (2014)", "The hot hand: A new approach to an old 'fallacy'. *8th Annual MIT Sloan Sports Analytics Conference*."),
    ]
    for author, text in refs:
        st.markdown(f"- **{author}** — {text}")

    # ── Live parameter impact summary ─────────────────────────────────────
    st.markdown("#### 📊 Current Parameter Impact")
    current_strength_label = (
        "Mild bias — close to rational" if heuristic_strength < 0.4 else
        "Moderate bias — typical fan/observer" if heuristic_strength < 0.7 else
        "Strong bias — classic Hot Hand believer"
    )
    st.info(f"**Heuristic Strength = {heuristic_strength}** → {current_strength_label}  \n"
            f"Mean season bias gap: **{mean_div*100:.2f} percentage points**  \n"
            f"Max bias gap: **{max_div*100:.2f} percentage points**")


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — HEURISTIC STRENGTH TEST
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Heuristic Strength Test</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="theory-box">
    Watch a real Luka game shot-by-shot. After each shot is revealed, predict whether the <b>next shot</b> will go in.
    Before each guess you'll see the shot zone and Luka's season average from that area — the same information
    a rational observer would use. When the game is over, your prediction pattern is fitted to the
    Cognitive Engine to estimate your personal <b>heuristic strength</b>: how much your intuition was swayed
    by recent makes and misses versus the statistical baseline.
    </div>
    """, unsafe_allow_html=True)

    # Season averages for context cards (pre-computed once)
    _zr_fg = raw_df.groupby('shot_zone_range')['shot_made'].mean().to_dict()
    _st_fg = raw_df.groupby('shot_type')['shot_made'].mean().to_dict()

    # ── Session state init ────────────────────────────────────────────────
    if 'quiz' not in st.session_state:
        st.session_state.quiz = {
            'started':     False,
            'game_df':     None,
            'shot_idx':    1,
            'predictions': [],
            'complete':    False,
        }
    q = st.session_state.quiz

    # ══════════════════════════════════════════════════════════════════════
    # START SCREEN
    # ══════════════════════════════════════════════════════════════════════
    if not q['started']:
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st.markdown("""
**How it works:**
1. A random game from the 2025–26 season is selected
2. Shot 1's outcome is revealed; you then predict Shot 2
3. Each prediction shows the upcoming shot's **zone**, Luka's **season average from that zone**, and the **current streak**
4. After every guess the actual result is revealed, then the next shot is queued
5. At the end your predictions are fitted to the Cognitive Engine — your estimated heuristic strength is shown
            """)
        with c_right:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Pick a Random Game", type="primary", use_container_width=True):
                _gid = np.random.choice(raw_df['game_id'].unique())
                _gdf = (raw_df[raw_df['game_id'] == _gid]
                        .sort_values('game_shot_number')
                        .reset_index(drop=True))
                st.session_state.quiz = {
                    'started':     True,
                    'game_df':     _gdf.to_dict('records'),
                    'shot_idx':    1,
                    'predictions': [],
                    'complete':    False,
                }
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # ACTIVE QUIZ  /  RESULTS
    # ══════════════════════════════════════════════════════════════════════
    else:
        _gdf      = pd.DataFrame(q['game_df'])
        _n        = len(_gdf)
        _preds    = q['predictions']
        _shot_idx = q['shot_idx']
        _total_p  = _n - 1
        _date_str = pd.to_datetime(_gdf['game_date'].iloc[0]).strftime('%b %d, %Y')

        # ── Top bar: game label + reset ───────────────────────────────────
        _hcol, _rcol = st.columns([5, 1])
        with _hcol:
            st.markdown(f"**{_date_str}** — {_n} shots in this game — predicting shots 2 through {_n}")
            if not q['complete']:
                st.progress(len(_preds) / _total_p,
                            text=f"{len(_preds)} of {_total_p} predictions made")
        with _rcol:
            if st.button("🔄 New Game", use_container_width=True):
                st.session_state.quiz = {
                    'started': False, 'game_df': None,
                    'shot_idx': 1, 'predictions': [], 'complete': False,
                }
                st.rerun()

        # ══════════════════════════════════════════════════════════════════
        # ACTIVE: shot history + current prediction prompt
        # ══════════════════════════════════════════════════════════════════
        if not q['complete']:
            st.markdown("---")

            # Always show Shot 1 (never predicted)
            _s0 = _gdf.iloc[0]
            _r0 = "✅ **MADE**" if _s0['shot_made'] else "❌ **MISSED**"
            st.markdown(f"**Shot 1:** {_r0} — {_s0['shot_zone_range']} ({_s0['shot_type']})")

            # History of predictions already made
            for _p in _preds:
                _si   = _p['shot_idx']
                _row  = _gdf.iloc[_si]
                _ai   = "✅" if _row['shot_made'] else "❌"
                _pi   = "✅" if _p['pred'] else "❌"
                _ok   = _p['pred'] == _row['shot_made']
                _clr  = "#22c55e" if _ok else "#ef4444"
                _mark = "✓" if _ok else "✗"
                st.markdown(
                    f"**Shot {_si + 1}:** {_ai} {'MADE' if _row['shot_made'] else 'MISSED'} "
                    f"— your guess: {_pi} "
                    f'— <span style="color:{_clr};font-weight:700;">{_mark}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Current prediction prompt
            _cur         = _gdf.iloc[_shot_idx]
            _zone_range  = _cur['shot_zone_range']
            _shot_type   = _cur['shot_type']
            _zone_avg    = _zr_fg.get(_zone_range, overall_fg)
            _cur_streak  = int(_cur.get('streak', 0))
            _streak_str  = f"+{_cur_streak}" if _cur_streak > 0 else str(_cur_streak)
            _streak_cls  = "green" if _cur_streak > 0 else ("red" if _cur_streak < 0 else "")

            st.markdown(f"### Predict Shot {_shot_idx + 1} of {_n}")
            _c1, _c2, _c3, _c4 = st.columns(4)
            with _c1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val" style="font-size:1.3rem;line-height:1.2;">{_zone_range}</div>'
                    f'<div class="metric-label">Shot Zone</div></div>',
                    unsafe_allow_html=True,
                )
            with _c2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val green" style="font-size:1.3rem;">{_zone_avg*100:.1f}%</div>'
                    f'<div class="metric-label">Season Avg — This Zone</div></div>',
                    unsafe_allow_html=True,
                )
            with _c3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val {_streak_cls}" style="font-size:1.3rem;">{_streak_str}</div>'
                    f'<div class="metric-label">Streak Entering Shot</div></div>',
                    unsafe_allow_html=True,
                )
            with _c4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val gold" style="font-size:1.3rem;">{_shot_type}</div>'
                    f'<div class="metric-label">Shot Type</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            _b1, _b2, _b3 = st.columns([1, 1, 2])
            with _b1:
                if st.button("✅  YES — Goes In", type="primary",
                             use_container_width=True, key=f"yes_{_shot_idx}"):
                    _new_preds = _preds + [{'shot_idx': _shot_idx, 'pred': 1}]
                    _new_idx   = _shot_idx + 1
                    st.session_state.quiz['predictions'] = _new_preds
                    if _new_idx >= _n:
                        st.session_state.quiz['complete'] = True
                    else:
                        st.session_state.quiz['shot_idx'] = _new_idx
                    st.rerun()
            with _b2:
                if st.button("❌  NO — Misses",
                             use_container_width=True, key=f"no_{_shot_idx}"):
                    _new_preds = _preds + [{'shot_idx': _shot_idx, 'pred': 0}]
                    _new_idx   = _shot_idx + 1
                    st.session_state.quiz['predictions'] = _new_preds
                    if _new_idx >= _n:
                        st.session_state.quiz['complete'] = True
                    else:
                        st.session_state.quiz['shot_idx'] = _new_idx
                    st.rerun()

        # ══════════════════════════════════════════════════════════════════
        # RESULTS
        # ══════════════════════════════════════════════════════════════════
        else:
            _preds = q['predictions']

            # ── Accuracy ─────────────────────────────────────────────────
            _correct  = sum(1 for p in _preds
                            if p['pred'] == _gdf.iloc[p['shot_idx']]['shot_made'])
            _accuracy = _correct / len(_preds) if _preds else 0

            # ── Fit heuristic strength via log-likelihood ─────────────────
            _strengths = np.arange(0.0, 1.01, 0.05)
            _best_ll   = -np.inf
            _best_s    = 0.0
            _ll_curve  = []

            for _s in _strengths:
                _eng = CognitiveEngine(
                    prior_fg_pct=overall_fg,
                    heuristic_strength=float(_s),
                    prior_anchor=prior_anchor,
                )
                _eng.new_game()
                _beliefs = []
                for _rn, (_, _r) in enumerate(_gdf.iterrows()):
                    if _rn >= 1:
                        _beliefs.append(_eng.current_belief)
                    _eng.update(int(_r['shot_made']))

                _ll = 0.0
                for _k, _p in enumerate(_preds):
                    _prob = np.clip(_beliefs[_k], 1e-6, 1 - 1e-6)
                    _ll  += _p['pred'] * np.log(_prob) + (1 - _p['pred']) * np.log(1 - _prob)
                _ll_curve.append(_ll)
                if _ll > _best_ll:
                    _best_ll = _ll
                    _best_s  = float(_s)

            # ── Interpretation label ──────────────────────────────────────
            if _best_s < 0.3:
                _interp = "near-rational — your predictions closely tracked the statistical baseline"
                _scls   = "green"
            elif _best_s < 0.55:
                _interp = "mildly biased — slight sensitivity to recent streaks"
                _scls   = ""
            elif _best_s < 0.75:
                _interp = "moderately biased — typical Hot Hand believer"
                _scls   = "gold"
            else:
                _interp = "strongly biased — heavily swayed by recent makes and misses"
                _scls   = "red"

            # ── Summary metrics ───────────────────────────────────────────
            st.markdown("## Your Results")
            _naive_acc = max(overall_fg, 1 - overall_fg)
            _rm1, _rm2, _rm3, _rm4 = st.columns(4)
            with _rm1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val green">{_correct}/{len(_preds)}</div>'
                    f'<div class="metric-label">Correct Predictions</div></div>',
                    unsafe_allow_html=True,
                )
            with _rm2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{_accuracy*100:.0f}%</div>'
                    f'<div class="metric-label">Your Accuracy</div></div>',
                    unsafe_allow_html=True,
                )
            with _rm3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val {_scls}">{_best_s:.2f}</div>'
                    f'<div class="metric-label">Your Heuristic Strength</div></div>',
                    unsafe_allow_html=True,
                )
            with _rm4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{_naive_acc*100:.0f}%</div>'
                    f'<div class="metric-label">Naive Baseline (always guess majority)</div></div>',
                    unsafe_allow_html=True,
                )

            st.info(
                f"**Interpretation:** Your prediction pattern best matches a Cognitive Engine "
                f"with heuristic strength **{_best_s:.2f}** — {_interp}."
            )

            # ── Charts ────────────────────────────────────────────────────
            # Rebuild beliefs for the fitted engine and rational engine
            _fit_eng = CognitiveEngine(
                prior_fg_pct=overall_fg, heuristic_strength=_best_s, prior_anchor=prior_anchor
            )
            _fit_eng.new_game()
            _rat_eng = RationalEngine(prior_makes=total_fgm, prior_misses=total_fga - total_fgm)
            _rat_eng.new_game()

            _fit_beliefs = []
            _rat_beliefs = []
            for _rn, (_, _r) in enumerate(_gdf.iterrows()):
                if _rn >= 1:
                    _fit_beliefs.append(_fit_eng.current_belief)
                    _rat_beliefs.append(_rat_eng.current_belief)
                _fit_eng.update(int(_r['shot_made']))
                _rat_eng.update(int(_r['shot_made']))

            _shot_nums   = list(range(2, _n + 1))
            _user_pred_v = [p['pred'] for p in _preds]

            fig_r, (ax_ll, ax_tr) = plt.subplots(1, 2, figsize=(14, 4))

            # Left: log-likelihood curve
            ax_ll.plot(_strengths, _ll_curve, color=COGNITIVE_COLOR, lw=2.5, marker='o', ms=4)
            ax_ll.axvline(_best_s, color=GOLD_COLOR, lw=2, ls='--',
                          label=f'Your fit: {_best_s:.2f}')
            ax_ll.axvline(heuristic_strength, color=RATIONAL_COLOR, lw=1.5, ls=':',
                          alpha=0.7, label=f'Dashboard setting: {heuristic_strength}')
            ax_ll.set_xlabel('Heuristic Strength')
            ax_ll.set_ylabel('Log-Likelihood')
            ax_ll.set_title('Prediction Fit vs Heuristic Strength', fontsize=11)
            ax_ll.legend(fontsize=9)
            ax_ll.grid(True, alpha=0.3)

            # Right: belief traces + user predictions
            ax_tr.plot(_shot_nums, [b * 100 for b in _rat_beliefs],
                       color=RATIONAL_COLOR, lw=2, label='Rational (Bayesian)', zorder=3)
            ax_tr.plot(_shot_nums, [b * 100 for b in _fit_beliefs],
                       color=COGNITIVE_COLOR, lw=2, ls='--',
                       label=f'Cognitive (fitted s={_best_s:.2f})', zorder=3)
            ax_tr.axhline(overall_fg * 100, color=GOLD_COLOR, lw=1, ls=':', alpha=0.7,
                          label=f'Season FG% ({overall_fg*100:.1f}%)')

            for _k, (_sn, _up) in enumerate(zip(_shot_nums, _user_pred_v)):
                _actual = _gdf.iloc[_k + 1]['shot_made']
                _marker = '^' if _up == 1 else 'v'
                _clr    = MAKE_COLOR if _up == _actual else MISS_COLOR
                ax_tr.scatter(_sn, 50, marker=_marker, color=_clr, s=70, zorder=5, alpha=0.85)

            ax_tr.set_ylim(20, 80)
            ax_tr.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax_tr.set_xlabel('Shot Number in Game')
            ax_tr.set_ylabel('FG Probability (%)')
            ax_tr.set_title(
                'Your Predictions vs Engine Beliefs\n'
                '(▲ = predicted make, ▼ = predicted miss;  green = correct, red = wrong)',
                fontsize=10,
            )
            ax_tr.legend(fontsize=8)
            ax_tr.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close()

            # ── Full prediction breakdown ─────────────────────────────────
            with st.expander("View full prediction breakdown"):
                _rows = []
                for _p in _preds:
                    _si  = _p['shot_idx']
                    _r   = _gdf.iloc[_si]
                    _ok  = _p['pred'] == _r['shot_made']
                    _rows.append({
                        'Shot #':          _si + 1,
                        'Zone':            _r['shot_zone_range'],
                        'Type':            _r['shot_type'],
                        'Streak':          int(_r.get('streak', 0)),
                        'Zone Season Avg': f"{_zr_fg.get(_r['shot_zone_range'], overall_fg)*100:.1f}%",
                        'Your Prediction': '✅ Make' if _p['pred'] else '❌ Miss',
                        'Actual':          '✅ Made' if _r['shot_made'] else '❌ Missed',
                        'Correct':         '✓' if _ok else '✗',
                    })
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
