# Hot Hand Fallacy — Luka Doncic Shot Analysis

**CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS**

An interactive investigation into the Hot Hand Fallacy using Luka Doncic's 2025-26 shot chart data. The project models two competing observers — a rational Bayesian and a cognitively biased human — and compares their shot probability estimates across a full NBA season.

## Project Overview

The [Hot Hand Fallacy](https://en.wikipedia.org/wiki/Hot-hand_fallacy) refers to the belief that a player who has made several shots in a row is more likely to make the next one. Gilovich, Vallone & Tversky (1985) argued this is a cognitive bias; more recent work has complicated that finding. This project explores it empirically using real shot-by-shot data.

### Components

| File | Description |
|------|-------------|
| [luka_hot_hand_data.py](luka_hot_hand_data.py) | Data pipeline — pulls shot chart from NBA API, builds chronological shot sequence with streaks and rolling FG% |
| [rational_engine.py](rational_engine.py) | Type 2 system: Bayesian Beta-Binomial observer that updates belief rationally after each shot |
| [cognitive_engine.py](cognitive_engine.py) | Type 1 system: Biased observer using exponentially decaying recency weights (models representativeness heuristic) |
| [dashboard.py](dashboard.py) | Streamlit dashboard for interactive visualization and parameter exploration |
| [hot_hand_analysis.ipynb](hot_hand_analysis.ipynb) | Exploratory analysis notebook |

## Getting Started

### Prerequisites

- Python 3.8+
- A virtual environment (recommended)

### Setup

```bash
git clone <your-repo-url>
cd nba-shotcharts
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Collect Data

```bash
python luka_hot_hand_data.py
```

This pulls Luka's shot chart from the NBA API and writes `luka_2025_26_shots.csv` with per-shot streak and rolling FG% columns.

### Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard requires `luka_2025_26_shots.csv` to be present.

## Acknowledgments

- [hkair/nba-shotcharts](https://github.com/hkair/nba-shotcharts) — initial reference for working with the NBA API and drawing the shot chart court
- [swar/nba_api](https://github.com/swar/nba_api) — the Python client for stats.nba.com
- Gilovich, T., Vallone, R., & Tversky, A. (1985). *The hot hand in basketball: On the misperception of random sequences.*
