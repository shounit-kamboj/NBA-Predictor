# NBA Game Outcome Predictor (2019–2025)

## Overview
An end-to-end **NBA game outcome prediction system** built to demonstrate practical skills in data engineering, feature design, and machine learning. The project scrapes and processes real NBA game data, engineers time-aware features, and trains a supervised model to predict game outcomes on future data. Results are aggregated to produce **projected team win totals** by conference.

This project emphasizes **clean data pipelines, avoidance of data leakage, and evaluation on unseen data**, mirroring real-world ML workflows used in industry.

---

## What This Project Demonstrates
- Building a **real-world data pipeline** from raw data to model output  
- Time-aware **feature engineering using rolling statistics**  
- Training and evaluating ML models on **future test sets**  
- Debugging subtle issues caused by notebook state and aggregation logic  
- Producing interpretable, team-level outputs from per-game predictions  

---

## Project Structure
```
├── scraping_data.ipynb    # Collects raw NBA game data
├── parse_data.ipynb       # Cleans, merges, and validates datasets
├── nba-model.ipynb        # Feature engineering, modeling, and evaluation
├── nba_games_2019to2025.csv
└── README.md
```

---

## Data Pipeline
1. **Data Collection**
   - NBA game-level data from the 2019–2025 seasons
   - One row per team per game
   - Includes team stats, opponent stats, and game context

2. **Data Cleaning**
   - Datetime normalization
   - Categorical encoding for opponents
   - Removal of duplicate or invalid columns

3. **Feature Engineering**
   - Contextual features: home/away, opponent, month
   - **15-game rolling averages** for team and opponent performance
   - Rolling windows use only prior games to prevent leakage

---

## Model
- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Training period**: Games before `2024-09-01`
- **Evaluation period**: Games between `2024-09-01` and `2025-04-19`
- **Target**: Binary win / loss outcome

```python
RandomForestClassifier(
    n_estimators=5000,
    min_samples_split=100,
    random_state=1
)
```

---

## Results
The model was evaluated on **unseen future games** from the 2024–2025 season window.

**Performance (game-level):**
- **Accuracy**: ~53%
- **Precision**: ~53%
- Performance exceeds a naïve random baseline and demonstrates the value of contextual and rolling features.

**Observations:**
- Rolling averages significantly improved stability over raw per-game stats.
- The model captures team strength trends but remains conservative in predicting upsets.
- Aggregating predictions highlights how small per-game errors compound at the season level.

**Standings Projection:**
- Per-game predictions were summed to estimate **projected team win totals**.
- Standings were generated separately for **Eastern** and **Western Conferences**.
- All 30 teams are retained, including teams with zero predicted wins, to avoid aggregation bias.

---

## Evaluation & Validation
- Accuracy and precision were computed using `scikit-learn`
- Confusion matrices were used to inspect class imbalance effects
- Strict date-based splits ensure no future information leaks into training

---

## Key Technical Takeaways
- Rolling features are powerful but require careful handling to avoid leakage
- Aggregations should be performed via **summation**, not filtering, to preserve edge cases
- Notebook execution order can silently affect downstream results if artifacts are reused
- Simple, well-engineered features can rival more complex modeling approaches

---

## Technologies Used
- **Python**
- **pandas**, **NumPy**
- **scikit-learn**
- **Jupyter Notebook**

---

## Future Improvements
- Calibrated probability predictions using `predict_proba`
- Advanced team metrics (ELO, net rating)
- Player- or lineup-level features
- Comparison against betting market baselines
- Model calibration and uncertainty analysis

---

*Built as a portfolio project to demonstrate applied machine learning and data engineering skills for internship and entry-level roles.*
