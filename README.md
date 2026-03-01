# ml-hcp-uplift-segmentation

Initial project scaffold for uplift modeling and ROI-based segmentation.

## Project structure

```text
ml-hcp-uplift-segmentation/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic_data_generator.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_uplift_modeling.ipynb
│   ├── 04_evaluation_qini.ipynb
│   └── 05_roi_simulation.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── modeling/
│   │   ├── uplift_xlearner.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluation.py
│   ├── segmentation/
│   │   ├── roi_ranking.py
│   │   └── segment_assign.py
│   ├── simulation/
│   │   └── roi_simulator.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── models/
│   └── saved_models/
├── reports/
│   ├── figures/
│   └── final_presentation.pdf
└── tests/
    └── test_pipeline.py
```
