# Project Summary: HCP Uplift Segmentation

## Overview
This project focuses on building an **Uplift Modeling Pipeline** to identify and segment Healthcare Professionals (HCPs) based on their incremental responsiveness (uplift) to targeted marketing or sales campaigns. The objective is to maximize the Return on Investment (ROI) of marketing efforts by targeting the HCPs who are most likely to be positively influenced, rather than those who would purchase anyway or react negatively.

## Development Journey & Milestones

### 1. Environment & Baseline Troubleshooting
- **Issue:** The initial `synthetic_data_generator.py` script failed to run due to missing packages (`numpy`, `pandas`), type conversion errors (`AttributeError: 'bool' object has no attribute 'astype'`), and interpreter misconfigurations.
- **Resolution:** 
  - Fixed syntax and type errors in the data generation code.
  - Activated the local Python virtual environment (`venv`).
  - Corrected and updated `requirements.txt` and successfully resolved missing module dependencies.
  - Automatically pushed the baseline fixes to the remote Git repository.

### 2. Developing the Uplift Modeling Architecture
We transformed a set of loose scripts into a highly professional, modularized machine learning architecture:

- **Central Configuration (`src/config.py`)**: Centralized global directory paths (data, models, reports) and hyperparameter configurations to keep code DRY and maintainable.
- **Feature Engineering (`src/data/feature_engineering.py`)**: Created functions to automatically encode categorical variables (specialty, region) and calculate derived features (e.g., `call_intensity`).
- **X-Learner Uplift Model (`src/modeling/uplift_xlearner.py`)**: 
  - Implemented the structural design of an X-Learner using `XGBoost`.
  - Used two distinct estimators (one for the treatment group, one for the control group) to isolate the treatment effect.
  - Added object persistence so models can be correctly saved and loaded via `joblib`.

### 3. Segmentation & Business Impact
- **Segmentation Strategy (`src/segmentation/segment_assign.py`)**: Converted continuous continuous uplift scores into actionable business tiers by ranking the population into `High`, `Medium`, and `Low` target segments.
- **ROI Simulation (`src/simulation/roi_simulator.py`)**: Built an ROI engine to simulate incremental value mathematically if the business only targets a top fraction, calculating net impact metrics to justify the model visually to stakeholders.

### 4. Evaluation Metrics & Visualizations
- **Advanced Metrics (`src/utils/metrics.py`)**: Programmed custom uplift metrics like the computationally complex **Qini curve** and **AUUC (Area Under the Uplift Curve)** over traditional accuracy models. Resolved `numpy 2.0` deprecated `trapz` bugs by upgrading code to use `np.trapezoid`.
- **Reporting (`src/utils/visualization.py`)**: Wired up `matplotlib` and `seaborn` to render clear distributions of predicted HCP segments alongside the Qini curve visualizations, saving the graphics natively to `/reports`.

### 5. Orchestration & Final Polish
- **Pipeline Runner (`src/modeling/train.py`)**: Unified every distinct module into a single elegant script that ingests the raw synthetic data, scales the features, trains the X-learner, outputs visualizations, logs terminal metrics and pickles the state space.
- **Git Hygiene (`.gitignore`)**: Set up a `.gitignore` to prevent polluting the codebase by keeping out heavy artifacts (`.csv`, cache, `.joblib`).
- **Final Deployment:** Successfully pushed the complete operational end-to-end framework to the main Git branch.

## How to Run End-to-End
With the environment activated with requisite requirements installed via `pip install -r requirements.txt`:
```bash
python -m src.modeling.train
```

*This will read data, execute the entire pipeline, yield metrics to the terminal, and save models/visualizations into root subdirectories.*
